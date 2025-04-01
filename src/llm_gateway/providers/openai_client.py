# llm_gateway/providers/openai_client.py

"""
Provider implementation for OpenAI models (e.g., GPT-4, GPT-3.5) using the
official OpenAI Python SDK (v1.0+). Supports both standard OpenAI and Azure OpenAI.
"""
from enum import Enum
import asyncio
import base64
import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, cast

# --- OpenAI SDK Imports ---
try:
    import openai
    from openai import AsyncOpenAI, AsyncAzureOpenAI
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
        ChatCompletionAssistantMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionToolParam,
        ChatCompletionContentPartParam,
        ChatCompletionMessageToolCallParam,
    )
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall
    from openai.types.completion_usage import CompletionUsage
    from openai._exceptions import (
        APIError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
        AuthenticationError,
        PermissionDeniedError,
        BadRequestError, # Replaces InvalidRequestError in v1.x
        OpenAIError,
    )
except ImportError:
    raise ImportError(
        "OpenAI SDK not found. Please install it using 'pip install openai'"
    )

# Gateway imports
from llm_gateway.core.models import (
    ContentItem,
    ErrorDetails,
    ErrorLevel,
    FinishReason,
    GatewayConfig,
    InterventionContext,
    LLMConfig,
    LLMRequest,
    LLMResponse,
    MCPContentType as GatewayContentType,
    MCPRole as GatewayRole,
    PerformanceMetrics,
    ProviderConfig,
    StreamChunk,
    ToolDefinition,
    ToolFunction,
    ToolResult, # Need this if mapping tool results back
    ToolUseRequest,
    UsageStats,
)
from llm_gateway.providers.base import BaseProvider

class OpenAIRole(str, Enum):
    """Specific roles recognized by the OpenAI Chat Completions API."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

logger = logging.getLogger(__name__)

# --- Constants ---
# Mapping OpenAI finish reasons to Gateway FinishReason
# Reference: https://platform.openai.com/docs/api-reference/chat/object (finish_reason)
OPENAI_FINISH_REASON_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTERED,
    # "function_call" (legacy) is superseded by tool_calls
    # Others like "error" are usually indicated by API errors
}

# Roles mapping
OPENAI_ROLE_MAP = {
    GatewayRole.SYSTEM: "system",
    GatewayRole.USER: "user",
    GatewayRole.ASSISTANT: "assistant",
    GatewayRole.TOOL: "tool", # Used for providing tool results back to the model
}

class OpenAIClient(BaseProvider):
    """
    LLM Gateway provider for OpenAI models using the Chat Completions API.
    Supports standard OpenAI and Azure OpenAI endpoints.
    """

    def __init__(self, provider_config: ProviderConfig, gateway_config: GatewayConfig):
        """
        Initialize the OpenAI Client.

        Args:
            provider_config: Configuration specific to this OpenAI provider instance.
                             Handles standard API key or Azure parameters.
            gateway_config: Global gateway configuration.
        """
        super().__init__(provider_config)
        self.gateway_config = gateway_config
        self._client: Optional[Union[AsyncOpenAI, AsyncAzureOpenAI]] = None
        self.is_azure = provider_config.connection_params.get("is_azure", False)

        # Get retry/timeout settings from provider_config or fallback to gateway_config
        self._max_retries = provider_config.connection_params.get("max_retries", gateway_config.max_retries)
        self._timeout = provider_config.connection_params.get("timeout_seconds", gateway_config.default_timeout_seconds)
        self._retry_delay = provider_config.connection_params.get("retry_delay_seconds", gateway_config.retry_delay_seconds)


        try:
            if self.is_azure:
                # --- Azure OpenAI Configuration ---
                api_key_env_var = provider_config.connection_params.get("api_key_env_var", "AZURE_OPENAI_API_KEY")
                endpoint_env_var = provider_config.connection_params.get("endpoint_env_var", "AZURE_OPENAI_ENDPOINT")
                api_version_env_var = provider_config.connection_params.get("api_version_env_var", "AZURE_OPENAI_API_VERSION")

                api_key = os.environ.get(api_key_env_var)
                azure_endpoint = os.environ.get(endpoint_env_var)
                api_version = os.environ.get(api_version_env_var)

                if not all([api_key, azure_endpoint, api_version]):
                    raise ValueError(
                         "Azure OpenAI requires API key, endpoint, and API version environment variables. "
                        f"Checked: '{api_key_env_var}', '{endpoint_env_var}', '{api_version_env_var}'"
                    )

                self._client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=api_version,
                    max_retries=self._max_retries,
                    timeout=self._timeout,
                )
                logger.info(f"Initialized AzureOpenAIClient provider '{self.provider_id}' for endpoint {azure_endpoint}")

            else:
                # --- Standard OpenAI Configuration ---
                api_key_env_var = provider_config.connection_params.get("api_key_env_var", "OPENAI_API_KEY")
                org_id_env_var = provider_config.connection_params.get("org_id_env_var", "OPENAI_ORG_ID") # Optional

                api_key = os.environ.get(api_key_env_var)
                org_id = os.environ.get(org_id_env_var)

                if not api_key:
                    raise ValueError(
                        f"Standard OpenAI API key environment variable '{api_key_env_var}' not found for provider '{self.provider_id}'."
                    )

                self._client = AsyncOpenAI(
                    api_key=api_key,
                    organization=org_id, # SDK handles None if org_id is not set
                    max_retries=self._max_retries,
                    timeout=self._timeout,
                )
                logger.info(f"Initialized OpenAIClient provider '{self.provider_id}'")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI SDK client for provider {self.provider_id}: {e}", exc_info=True)
            raise ConnectionError(f"OpenAI SDK client initialization failed: {e}") from e


    async def cleanup(self):
        """Closes the OpenAI client."""
        if self._client:
            try:
                # httpx client used internally should manage connections.
                # await self._client.close() # Usually not required for OpenAI SDK v1+
                logger.info(f"OpenAI client for provider '{self.provider_id}' cleanup called (usually managed by httpx).")
            except Exception as e:
                logger.warning(f"Error during OpenAI client cleanup for '{self.provider_id}': {e}", exc_info=True)
        self._client = None


    # --- Core Generation Methods ---

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from OpenAI using the non-streaming Chat Completions API.
        """
        start_time = datetime.utcnow()
        llm_latency_ms: Optional[float] = None
        response_obj: Optional[ChatCompletion] = None
        error_details: Optional[ErrorDetails] = None

        if not self._client:
             raise ConnectionError("OpenAI client is not initialized.")

        try:
            # 1. Map Gateway Request to OpenAI API format
            # Azure requires deployment_id which is often the model name in the request
            model_id = request.config.model_identifier
            openai_params = self._map_request(request, model_id)

            # 2. Call OpenAI API
            logger.debug(f"Sending request to OpenAI model '{model_id}': {openai_params}")
            llm_call_start = datetime.utcnow()

            response_obj = await self._client.chat.completions.create(**openai_params)

            llm_latency_ms = (datetime.utcnow() - llm_call_start).total_seconds() * 1000
            logger.debug(f"Received response from OpenAI model '{model_id}'. ID: {response_obj.id}")

        except (APITimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"OpenAI request timed out for provider {self.provider_id}: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=True)
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded for provider {self.provider_id}: {e}", exc_info=True)
            retry_after = self._get_retry_after(e)
            error_details = self._map_error(e, retryable=True, retry_after=retry_after)
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication error for provider {self.provider_id}: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=False) # Not retryable
        except PermissionDeniedError as e:
             logger.error(f"OpenAI permission denied for provider {self.provider_id}: {e}", exc_info=True)
             error_details = self._map_error(e, retryable=False)
        except BadRequestError as e: # Catches 400 errors like invalid params, content policy violation
             logger.error(f"OpenAI bad request error for provider {self.provider_id}: {e.code} - {e.message}", exc_info=True)
             error_details = self._map_error(e, retryable=False) # Usually not retryable
             # Check for content filter specifically
             if e.code == 'content_filter':
                  error_details.code = "PROVIDER_CONTENT_FILTER" # More specific code
        except APIStatusError as e: # Other HTTP status errors
             logger.error(f"OpenAI API error for provider {self.provider_id}: Status={e.status_code}, Response={e.response.text}", exc_info=True)
             # Retry based on status code
             retryable = e.status_code in [429, 500, 502, 503, 504]
             error_details = self._map_error(e, retryable=retryable)
        except APIError as e: # Catch-all for other OpenAI API errors
            logger.error(f"OpenAI API error for provider {self.provider_id}: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=False) # Assume non-retryable unless specific
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI generate for provider {self.provider_id}: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=False)

        # 3. Map OpenAI Response back to Gateway LLMResponse
        total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        final_response = self._map_response(
            openai_response=response_obj,
            original_request=request,
            error_details=error_details,
            llm_latency_ms=llm_latency_ms,
            total_duration_ms=total_duration_ms,
        )

        return final_response

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response from OpenAI using the Chat Completions API.
        Yields gateway StreamChunk objects.
        """
        request_id = request.initial_context.request_id
        chunk_index = 0
        accumulated_content = ""
        accumulated_tool_calls: Dict[int, ChoiceDeltaToolCall] = {} # Accumulate tool call deltas by index
        final_usage: Optional[UsageStats] = None # OpenAI doesn't stream usage typically
        final_finish_reason: Optional[FinishReason] = None

        if not self._client:
            logger.error(f"OpenAI client not initialized for streaming request {request_id}")
            yield self._create_error_chunk(
                request_id, 0, self._map_error(ConnectionError("OpenAI client not initialized."))
            )
            return

        try:
            # 1. Map request
            model_id = request.config.model_identifier
            openai_params = self._map_request(request, model_id)
            openai_params["stream"] = True
            # stream_options needed for usage in stream (Azure supports, OpenAI might beta)
            openai_params["stream_options"] = {"include_usage": True}

            logger.debug(f"Starting stream request to OpenAI model '{model_id}'")

            # 2. Call OpenAI streaming API
            stream = await self._client.chat.completions.create(**openai_params)

            async for chunk in stream:
                chunk_start_time = datetime.utcnow()
                mapped_chunk = None
                delta: Optional[ChoiceDelta] = None
                finish_reason_str: Optional[str] = None

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    finish_reason_str = chunk.choices[0].finish_reason

                # Check for usage in the chunk (supported by Azure, maybe OpenAI beta)
                chunk_usage = getattr(chunk, 'usage', None)
                if chunk_usage:
                     final_usage = self._map_usage(chunk_usage) # Overwrite with latest usage chunk

                if delta:
                    # -- Text Delta --
                    if delta.content:
                        accumulated_content += delta.content
                        mapped_chunk = StreamChunk(
                            chunk_id=chunk_index,
                            request_id=request_id,
                            delta_text=delta.content,
                        )

                    # -- Tool Call Delta --
                    if delta.tool_calls:
                        # Accumulate tool call chunks
                        tool_deltas_to_yield = []
                        for tool_call_delta in delta.tool_calls:
                            index = tool_call_delta.index
                            if index not in accumulated_tool_calls:
                                # Start of a new tool call
                                accumulated_tool_calls[index] = tool_call_delta
                            else:
                                # Append function arguments chunk
                                if tool_call_delta.function and tool_call_delta.function.arguments:
                                     existing_args = accumulated_tool_calls[index].function.arguments or ""
                                     accumulated_tool_calls[index].function.arguments = existing_args + tool_call_delta.function.arguments
                                # Update other fields if necessary (id, name, type usually come first)
                                if tool_call_delta.id: accumulated_tool_calls[index].id = tool_call_delta.id
                                if tool_call_delta.type: accumulated_tool_calls[index].type = tool_call_delta.type
                                if tool_call_delta.function and tool_call_delta.function.name:
                                     accumulated_tool_calls[index].function.name = tool_call_delta.function.name

                        # We only yield the *complete* tool call info when the finish reason arrives.
                        # Alternatively, yield deltas as they come? Less useful for gateway.
                        # Let's yield a placeholder chunk indicating tool activity if needed.
                        # mapped_chunk = StreamChunk(...) # Indicate tool delta without full info

                # -- Finish Reason --
                if finish_reason_str:
                    final_finish_reason = self._map_finish_reason(finish_reason_str)
                    # If finished due to tool calls, map the accumulated calls now
                    final_tool_requests: Optional[List[ToolUseRequest]] = None
                    if final_finish_reason == FinishReason.TOOL_CALLS:
                         final_tool_requests = self._map_tool_call_deltas(accumulated_tool_calls, request.tools)

                    # Create final chunk with finish reason, usage (if available), and completed tool calls
                    mapped_chunk = StreamChunk(
                         chunk_id=chunk_index,
                         request_id=request_id,
                         finish_reason=final_finish_reason,
                         usage_update=final_usage, # Might be None
                         delta_tool_calls=final_tool_requests, # Attach final tools here
                         # Optionally add final accumulated text if finish reason != tool_calls?
                         # delta_text=accumulated_content if final_finish_reason != FinishReason.TOOL_CALLS else None
                    )
                    # Clear accumulated state after final chunk
                    accumulated_content = ""
                    accumulated_tool_calls = {}

                # Yield the mapped chunk if one was created
                if mapped_chunk:
                    yield mapped_chunk
                    chunk_index += 1

        except (APITimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"OpenAI stream timed out for request {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=True))
        except RateLimitError as e:
            logger.error(f"OpenAI stream rate limit for request {request_id}: {e}", exc_info=True)
            retry_after = self._get_retry_after(e)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=True, retry_after=retry_after))
        except AuthenticationError as e:
            logger.error(f"OpenAI stream authentication error for request {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))
        except PermissionDeniedError as e:
             logger.error(f"OpenAI stream permission denied for request {request_id}: {e}", exc_info=True)
             yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))
        except BadRequestError as e:
             logger.error(f"OpenAI stream bad request error for request {request_id}: {e.code} - {e.message}", exc_info=True)
             error_details = self._map_error(e, retryable=False)
             if e.code == 'content_filter': error_details.code = "PROVIDER_CONTENT_FILTER"
             yield self._create_error_chunk(request_id, chunk_index, error_details)
        except APIStatusError as e:
             logger.error(f"OpenAI stream API error for request {request_id}: Status={e.status_code}, Response={e.response.text if e.response else 'N/A'}", exc_info=True)
             retryable = e.status_code in [429, 500, 502, 503, 504]
             yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=retryable))
        except APIError as e: # Catch-all for other OpenAI API errors
            logger.error(f"OpenAI stream API error for request {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI stream for request {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))

        finally:
            logger.debug(f"OpenAI stream finished for request {request_id}.")


    # --- Mapping Helper Methods ---

    def _map_request(self, request: LLMRequest, model_id: str) -> Dict[str, Any]:
        """Maps the gateway LLMRequest to OpenAI's chat.completions.create parameters."""
        openai_params: Dict[str, Any] = {}

        # 1. Model Identifier (passed in for Azure compatibility)
        openai_params["model"] = model_id

        # 2. Messages (History + Current Prompt)
        messages: List[ChatCompletionMessageParam] = []

        # Handle System Prompt first
        system_prompt = request.config.system_prompt
        if system_prompt:
             messages.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

        # Map conversation history
        for turn in request.initial_context.conversation_history:
            # Skip system message if already handled above
            if turn.role == GatewayRole.SYSTEM.value and system_prompt:
                 continue

            role = self._map_role(turn.role)
            if not role: # Skip unmappable roles
                  logger.warning(f"Skipping history turn with unmappable role '{turn.role}'")
                  continue

            # Handle different content types and roles
            if role == "tool":
                 # Expect content to be ToolResult
                 if isinstance(turn.content, ToolResult):
                      messages.append(ChatCompletionToolMessageParam(
                           role="tool",
                           content=str(turn.content.output), # OpenAI expects string content for tool result
                           tool_call_id=turn.content.tool_call_id
                      ))
                 else:
                      logger.warning(f"History turn with role 'tool' has unexpected content type: {type(turn.content)}. Skipping.")
            else: # System (if not handled above), User, Assistant
                 content = self._map_content_to_openai(turn.content)
                 if content:
                      # Cast needed because mapped roles aren't specific enough for Param types
                      if role == "system": msg = ChatCompletionSystemMessageParam(role=role, content=content)
                      elif role == "user": msg = ChatCompletionUserMessageParam(role=role, content=content)
                      elif role == "assistant": msg = ChatCompletionAssistantMessageParam(role=role, content=content) # Handle potential tool_calls here if replaying history?
                      else: msg = None # Should not happen due to role mapping

                      if msg: messages.append(msg)
                 else:
                      logger.warning(f"Could not map content for history turn with role '{role}'. Type: {type(turn.content)}. Skipping.")


        # Map current prompt content (always as 'user' message)
        current_content = self._map_content_to_openai(request.prompt_content)
        if current_content:
            messages.append(ChatCompletionUserMessageParam(role="user", content=current_content))
        elif not messages:
            # Handle case where history is empty and prompt is empty/unmappable
            raise ValueError("Cannot send request to OpenAI with no valid message content.")

        openai_params["messages"] = messages

        # 3. Configuration Parameters
        if request.config.max_tokens is not None: openai_params["max_tokens"] = request.config.max_tokens
        if request.config.temperature is not None: openai_params["temperature"] = request.config.temperature
        if request.config.top_p is not None: openai_params["top_p"] = request.config.top_p
        if request.config.stop_sequences: openai_params["stop"] = request.config.stop_sequences # Renamed param
        if request.config.presence_penalty is not None: openai_params["presence_penalty"] = request.config.presence_penalty
        if request.config.frequency_penalty is not None: openai_params["frequency_penalty"] = request.config.frequency_penalty
        # Map other supported params like logit_bias, user, seed if needed

        # 4. Tools & Tool Choice
        if request.tools:
             mapped_tools = self._map_tools_to_openai(request.tools)
             if mapped_tools:
                  openai_params["tools"] = mapped_tools
                  # Handle tool_choice if needed
                  # Example: openai_params["tool_choice"] = "auto" # or {"type": "function", "function": {"name": "my_func"}}
                  # tool_choice_config = request.config.extra_params.get("tool_choice")
                  # if tool_choice_config: openai_params["tool_choice"] = tool_choice_config


        return openai_params

    def _map_role(self, gateway_role_str: str) -> Optional[OpenAIRole]:
        """
        Maps gateway role string to the OpenAIRole enum.

        Args:
            gateway_role_str: The role string (e.g., "user", "assistant") from gateway models.

        Returns:
            The corresponding OpenAIRole enum member, or None if mapping fails.
        """
        try:
            # Convert the input string to the GatewayRole Enum member
            gateway_role_enum = GatewayRole(gateway_role_str)
        except ValueError:
            logger.warning(f"Unknown gateway role '{gateway_role_str}' cannot be mapped to OpenAI role.")
            return None

        # Look up the corresponding OpenAIRole Enum member in the map
        openai_role_enum = OPENAI_ROLE_MAP.get(gateway_role_enum)

        if openai_role_enum is None:
             # This case should ideally not happen if OPENAI_ROLE_MAP is complete
             # relative to GatewayRole, but handles potential inconsistencies.
             logger.warning(f"Gateway role '{gateway_role_enum.value}' has no corresponding mapping in OPENAI_ROLE_MAP.")

        return openai_role_enum


    def _map_content_to_openai(self, content: Union[str, List[ContentItem], Any]) -> Union[str, List[ChatCompletionContentPartParam], None]:
         """Maps gateway content to OpenAI message content format (string or list of parts)."""
         if isinstance(content, str):
              return content # Simple text
         elif isinstance(content, list) and all(isinstance(item, ContentItem) for item in content):
              # Multimodal content or multiple text parts
              parts: List[ChatCompletionContentPartParam] = []
              has_image = False
              for item in content:
                   if item.type == GatewayContentType.TEXT and item.text_content:
                        parts.append({"type": "text", "text": item.text_content})
                   elif item.type == GatewayContentType.IMAGE:
                        try:
                             image_url_data = self._create_openai_image_url(item)
                             parts.append({"type": "image_url", "image_url": image_url_data})
                             has_image = True
                        except ValueError as e:
                             logger.warning(f"Skipping image content item due to error: {e}")
                   else:
                        logger.warning(f"Skipping unsupported content item type for OpenAI: {item.type}")

              if not parts: return None
              # OpenAI API limitation: If image_url is present, text content cannot be simple string, must be part list.
              # If only text parts exist, we *could* combine them into a single string, but list is safer.
              return parts
         else:
              logger.warning(f"Unsupported content type for OpenAI mapping: {type(content)}")
              return None

    def _create_openai_image_url(self, item: ContentItem) -> Dict[str, Any]:
         """Creates the image_url dictionary for OpenAI from a ContentItem."""
         # OpenAI accepts base64 encoded images OR public URLs. URLs preferred if available.
         # Format: {"url": "data:image/jpeg;base64,{base64_image_data}"} OR {"url": "http://...", "detail": "auto|low|high"}
         source = item.data.get("image", {}).get("source", {})
         source_type = source.get("type")

         if source_type == "url":
              url = source.get("url")
              if not url: raise ValueError("Image source type is 'url' but URL is missing.")
              # Optionally add detail level if needed
              return {"url": url} # Add "detail": "auto" if needed

         elif source_type == "base64":
              b64_data = source.get("data")
              mime_type = item.mime_type
              if not mime_type: raise ValueError("MIME type missing for base64 image")
              if not b64_data: raise ValueError("Base64 data missing")
              return {"url": f"data:{mime_type};base64,{b64_data}"}

         else:
              # Try to extract base64 from other fields if needed (e.g. direct item.data)
              raise ValueError(f"Unsupported image source type for OpenAI: {source_type}. Requires 'url' or 'base64'.")


    def _map_tools_to_openai(self, tools: List[ToolDefinition]) -> List[ChatCompletionToolParam]:
        """Maps gateway ToolDefinition list to OpenAI ChatCompletionToolParam list."""
        openai_tools: List[ChatCompletionToolParam] = []
        for tool in tools:
            try:
                func = tool.function
                # OpenAI expects 'function' type tools with parameters schema
                openai_tools.append(ChatCompletionToolParam(
                    type="function",
                    function={
                         "name": func.name,
                         "description": func.description or "", # Description recommended
                         "parameters": func.parameters or {"type": "object", "properties": {}}, # Schema required
                    }
                ))
            except Exception as e:
                logger.error(f"Failed to map tool '{getattr(tool.function, 'name', 'N/A')}' to OpenAI format: {e}", exc_info=True)
        return openai_tools

    def _map_response(
        self,
        openai_response: Optional[ChatCompletion],
        original_request: LLMRequest,
        error_details: Optional[ErrorDetails],
        llm_latency_ms: Optional[float],
        total_duration_ms: Optional[float],
    ) -> LLMResponse:
        """Maps the OpenAI ChatCompletion object back to the gateway's LLMResponse."""

        generated_content: Optional[Union[str, List[ContentItem]]] = None
        tool_use_requests: Optional[List[ToolUseRequest]] = None
        finish_reason: FinishReason = FinishReason.UNKNOWN # Default
        usage: Optional[UsageStats] = None
        first_choice: Optional[Choice] = None

        if error_details:
            finish_reason = FinishReason.ERROR
        elif openai_response and openai_response.choices:
             first_choice = openai_response.choices[0]
             message = first_choice.message

             # Map Finish Reason
             finish_reason = self._map_finish_reason(first_choice.finish_reason)

             # Map Content / Tool Calls from assistant message
             if message.content:
                  generated_content = message.content # Usually string content

             if message.tool_calls:
                  tool_use_requests = self._map_tool_calls(message.tool_calls, original_request.tools)
                  # Per OpenAI spec, content is null if tool_calls are present
                  generated_content = None

             # Map Usage
             usage = self._map_usage(openai_response.usage)

        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            total_duration_ms=total_duration_ms,
            llm_latency_ms=llm_latency_ms,
        )

        # Final context state might be enriched by interventions later
        final_context_state = original_request.initial_context

        return LLMResponse(
            version=original_request.version,
            request_id=original_request.initial_context.request_id,
            generated_content=generated_content,
            finish_reason=finish_reason,
            tool_use_requests=tool_use_requests,
            usage=usage,
            compliance_result=None,  # Filled by interventions layer
            final_context=final_context_state,
            error_details=error_details,
            performance_metrics=perf_metrics,
            # Optionally store raw response for debugging
            raw_provider_response=openai_response.model_dump() if openai_response else None,
        )

    def _map_finish_reason(self, openai_reason: Optional[str]) -> FinishReason:
        """Maps OpenAI finish reason string to gateway FinishReason."""
        if not openai_reason:
            return FinishReason.UNKNOWN
        return OPENAI_FINISH_REASON_MAP.get(openai_reason, FinishReason.UNKNOWN)

    def _map_usage(self, openai_usage: Optional[CompletionUsage]) -> Optional[UsageStats]:
        """Maps OpenAI CompletionUsage object to gateway UsageStats."""
        if not openai_usage:
            return None
        # Ensure tokens > 0 before creating stats object
        prompt_tokens = openai_usage.prompt_tokens
        completion_tokens = openai_usage.completion_tokens
        if prompt_tokens > 0 or completion_tokens > 0:
             return UsageStats(
                  prompt_tokens=prompt_tokens,
                  completion_tokens=completion_tokens,
                  total_tokens=openai_usage.total_tokens
             )
        return None

    def _map_tool_calls(self, tool_calls: List[ChatCompletionMessageToolCallParam], original_tools: Optional[List[ToolDefinition]]) -> List[ToolUseRequest]:
         """Maps OpenAI tool_calls list to gateway ToolUseRequest list."""
         tool_requests = []
         for call in tool_calls:
              # Only handle function calls for now
              if call.type == "function":
                   try:
                        # Find original tool definition for potential schema/description enrichment
                        original_tool_def_func = next((t.function for t in original_tools or [] if t.function.name == call.function.name), None)

                        tool_requests.append(ToolUseRequest(
                             id=call.id, # OpenAI provides the ID for the tool result message
                             type="function",
                             function=ToolFunction(
                                  name=call.function.name,
                                  description=original_tool_def_func.description if original_tool_def_func else None,
                                  # OpenAI provides arguments as a string needing parsing
                                  parameters={"arguments": call.function.arguments}, # Store raw string args
                             )
                        ))
                   except Exception as e:
                        logger.error(f"Failed to map OpenAI function tool call (id: {call.id}, name: {getattr(call.function, 'name', 'N/A')}): {e}", exc_info=True)
              else:
                   logger.warning(f"Unsupported OpenAI tool call type: {call.type}")
         return tool_requests

    def _map_tool_call_deltas(self, deltas: Dict[int, ChoiceDeltaToolCall], original_tools: Optional[List[ToolDefinition]]) -> List[ToolUseRequest]:
         """Maps accumulated OpenAI stream tool deltas to gateway ToolUseRequest list."""
         tool_requests = []
         for index, call_delta in deltas.items():
              if call_delta.type == "function" and call_delta.function:
                   try:
                        # Find original tool definition
                        original_tool_def_func = next((t.function for t in original_tools or [] if t.function.name == call_delta.function.name), None)

                        tool_requests.append(ToolUseRequest(
                             id=call_delta.id or f"tool_{index}", # Use index if ID missing in delta (shouldn't happen?)
                             type="function",
                             function=ToolFunction(
                                  name=call_delta.function.name or "",
                                  description=original_tool_def_func.description if original_tool_def_func else None,
                                  parameters={"arguments": call_delta.function.arguments or ""},
                             )
                        ))
                   except Exception as e:
                        logger.error(f"Failed to map accumulated OpenAI function tool call delta (index: {index}): {e}", exc_info=True)
              else:
                   logger.warning(f"Unsupported accumulated OpenAI tool call delta type: {call_delta.type}")
         return tool_requests

    def _map_error(self, error: Exception, retryable: Optional[bool] = None, retry_after: Optional[int] = None) -> ErrorDetails:
        """Maps various exceptions to the gateway's ErrorDetails model."""
        code = "PROVIDER_ERROR"
        message = f"OpenAI provider '{self.provider_id}' encountered an error: {str(error)}"
        level = ErrorLevel.ERROR
        provider_details: Optional[Dict[str, Any]] = None
        is_retryable = retryable if retryable is not None else False # Default to False

        if isinstance(error, APITimeoutError):
            code = "PROVIDER_TIMEOUT"
            message = f"OpenAI request timed out after {self._timeout}s."
            is_retryable = True
        elif isinstance(error, RateLimitError):
            code = "PROVIDER_RATE_LIMIT"
            message = f"OpenAI rate limit exceeded. {getattr(error, 'message', '')}"
            is_retryable = True
            provider_details = error.body # Contains details like type, code
        elif isinstance(error, AuthenticationError):
            code = "PROVIDER_AUTH_ERROR"
            message = "OpenAI authentication failed. Check API key or Azure credentials."
            is_retryable = False
        elif isinstance(error, PermissionDeniedError):
             code = "PROVIDER_PERMISSION_ERROR"
             message = "OpenAI permission denied. Check model access or organization settings."
             is_retryable = False
        elif isinstance(error, BadRequestError): # Includes InvalidRequestError, ContentPolicyViolation
             code = f"PROVIDER_BAD_REQUEST_{getattr(error, 'code', 'UNKNOWN')}"
             message = f"OpenAI rejected the request as invalid ({getattr(error, 'code', 'UNKNOWN')}): {error.message}"
             is_retryable = False
             provider_details = error.body
             if getattr(error, 'code', None) == 'content_filter':
                  code = "PROVIDER_CONTENT_FILTER"
                  level = ErrorLevel.WARNING # Content filter isn't necessarily a system error
        elif isinstance(error, APIStatusError):
             code = f"PROVIDER_HTTP_{error.status_code}"
             message = f"OpenAI API returned status {error.status_code}: {error.message or error.response.text if error.response else ''}"
             if error.status_code == 429: is_retryable = True; code = "PROVIDER_RATE_LIMIT"
             elif error.status_code >= 500: is_retryable = True # Server errors are potentially retryable
             provider_details = error.body if hasattr(error, 'body') else {"raw_response": error.response.text if error.response else None}
        elif isinstance(error, APIError): # General API error
            code = "PROVIDER_API_ERROR"
            message = f"OpenAI API Error ({getattr(error, 'code', 'UNKNOWN')}): {str(error)}"
            provider_details = error.body if hasattr(error, 'body') else None
        elif isinstance(error, ConnectionError): # Includes network errors from httpx
            code = "PROVIDER_CONNECTION_ERROR"
            message = f"Could not connect to OpenAI API: {str(error)}"
            is_retryable = True
        elif isinstance(error, ValueError): # e.g., mapping errors
             code = "PROVIDER_MAPPING_ERROR"
             message = f"Data mapping error for OpenAI: {str(error)}"
             level = ErrorLevel.WARNING

        # Add body/details if not already captured
        if not provider_details and hasattr(error, 'body'):
             provider_details = getattr(error, 'body', None)
        if not provider_details:
             provider_details = {"exception_type": type(error).__name__}

        return ErrorDetails(
            code=code,
            message=message,
            level=level,
            provider_error_details=provider_details,
            retryable=is_retryable,
            retry_after_seconds=retry_after
        )

    def _create_error_chunk(self, request_id: str, chunk_id: int, error_details: ErrorDetails) -> StreamChunk:
         """Helper to create a StreamChunk representing an error."""
         return StreamChunk(
              chunk_id=chunk_id,
              request_id=request_id,
              finish_reason=FinishReason.ERROR,
              provider_specific_data={"error": error_details.model_dump()}
         )

    def _get_retry_after(self, error: Union[RateLimitError, APIStatusError]) -> Optional[int]:
         """Extract retry-after header if available."""
         # OpenAI SDK v1+ parses headers into the error object sometimes
         headers = getattr(error, 'response', None) and getattr(error.response, 'headers', None)
         if headers and 'retry-after' in headers:
              try:
                   return int(headers['retry-after'])
              except (ValueError, TypeError): pass
         if headers and 'retry-after-ms' in headers: # Some APIs use ms
               try:
                    return int(headers['retry-after-ms']) // 1000
               except (ValueError, TypeError): pass
         # Check body for some specific rate limit errors (less common now)
         body = getattr(error, 'body', None)
         if isinstance(body, dict) and 'error' in body and isinstance(body['error'], dict):
             message = body['error'].get('message', '')
             import re
             match = re.search(r"retry after (\d+) seconds", message, re.IGNORECASE)
             if match:
                  return int(match.group(1))
         return None

# --- End of OpenAIClient class ---