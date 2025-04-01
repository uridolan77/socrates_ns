# llm_gateway/providers/anthropic_client.py

"""
Provider implementation for Anthropic models (e.g., Claude) using the official
Anthropic Python SDK and the Messages API.
"""
from enum import Enum
import asyncio
import base64
import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, cast

# --- Anthropic SDK Imports ---
try:
    import anthropic
    from anthropic import AsyncAnthropic
    from anthropic.types import Message, MessageParam, TextBlockParam, ImageBlockParam, ToolParam, MessageStreamEvent
    from anthropic.types.message import Usage as AnthropicUsage
    from anthropic.types.message_stream_event import (
        ContentBlockDeltaEvent,
        ContentBlockStartEvent,
        ContentBlockStopEvent,
        MessageStartEvent,
        MessageStopEvent,
        MessageDeltaEvent, # Includes usage in delta
    )
    from anthropic._exceptions import (
        APIError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
        AnthropicError,
    )
except ImportError:
    raise ImportError(
        "Anthropic SDK not found. Please install it using 'pip install anthropic'"
    )

# Gateway imports
from llm_gateway.core.models import (
    BaseGatewayModel, # Use for internal mapping if needed
    ContentItem,
    ErrorDetails,
    ErrorLevel,
    FinishReason,
    GatewayConfig,
    InterventionContext,
    LLMConfig,
    LLMRequest,
    LLMResponse,
    MCPContentType as GatewayContentType, # Renamed for clarity
    MCPRole as GatewayRole,               # Renamed for clarity
    PerformanceMetrics,
    ProviderConfig,
    StreamChunk,
    ToolDefinition,
    ToolFunction,
    ToolUseRequest,
    UsageStats,
)
from llm_gateway.providers.base import BaseProvider # Assuming this exists

logger = logging.getLogger(__name__)

# --- Constants ---
# Mapping Anthropic stop reasons to Gateway FinishReason
# Reference: https://docs.anthropic.com/claude/reference/messages-streaming (stop_reason)
ANTHROPIC_STOP_REASON_MAP = {
    "end_turn": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.TOOL_CALLS,
    # Others like "error", "content_filtered" might not be direct stop_reasons
    # but indicated by API errors or specific stream events.
}

ANTHROPIC_ROLE_MAP = {
    GatewayRole.USER: "user",
    GatewayRole.ASSISTANT: "assistant",
    # System role is handled differently in Anthropic's Messages API (top-level `system` param)
    # Tool role messages are constructed from tool results by the client
}

class AnthropicRole(str, Enum):
    """Specific roles recognized within the Anthropic Messages API message list."""
    USER = "user"
    ASSISTANT = "assistant"

class AnthropicClient(BaseProvider):
    """
    LLM Gateway provider for Anthropic models using the Messages API.
    """

    def __init__(self, provider_config: ProviderConfig, gateway_config: GatewayConfig):
        """
        Initialize the Anthropic Client.

        Args:
            provider_config: Configuration specific to this Anthropic provider instance.
                             Expects 'api_key_env_var' in connection_params.
            gateway_config: Global gateway configuration.
        """
        super().__init__(provider_config)
        self.gateway_config = gateway_config
        self._client: Optional[AsyncAnthropic] = None

        # --- Configuration ---
        api_key_env_var = provider_config.connection_params.get("api_key_env_var", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key environment variable '{api_key_env_var}' not found for provider '{self.provider_id}'."
            )

        # Get retry/timeout settings from provider_config or fallback to gateway_config
        self._max_retries = provider_config.connection_params.get("max_retries", gateway_config.max_retries)
        self._timeout = provider_config.connection_params.get("timeout_seconds", gateway_config.default_timeout_seconds)
        self._retry_delay = provider_config.connection_params.get("retry_delay_seconds", gateway_config.retry_delay_seconds)

        # --- Initialize Anthropic SDK Client ---
        try:
            self._client = AsyncAnthropic(
                api_key=api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
            logger.info(f"Initialized AnthropicClient provider '{self.provider_id}'")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic SDK client for provider {self.provider_id}: {e}", exc_info=True)
            raise ConnectionError(f"Anthropic SDK client initialization failed: {e}") from e

    async def cleanup(self):
        """Closes the Anthropic client."""
        if self._client:
            try:
                # The AsyncAnthropic client uses httpx internally, which should manage connections.
                # Explicit close might be needed depending on SDK version and usage patterns.
                # await self._client.close() # Uncomment if SDK requires explicit close
                logger.info(f"Anthropic client for provider '{self.provider_id}' cleanup called (usually managed by httpx).")
            except Exception as e:
                logger.warning(f"Error during Anthropic client cleanup for '{self.provider_id}': {e}", exc_info=True)
        self._client = None

    # --- Core Generation Methods ---

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from Anthropic using the non-streaming Messages API.
        """
        start_time = datetime.utcnow()
        llm_latency_ms: Optional[float] = None
        response_obj: Optional[Message] = None
        error_details: Optional[ErrorDetails] = None

        if not self._client:
             raise ConnectionError("Anthropic client is not initialized.")

        try:
            # 1. Map Gateway Request to Anthropic API format
            anthropic_params = self._map_request(request)

            # 2. Call Anthropic API
            logger.debug(f"Sending request to Anthropic model '{anthropic_params.get('model')}': {anthropic_params}")
            llm_call_start = datetime.utcnow()

            response_obj = await self._client.messages.create(**anthropic_params) # type: ignore

            llm_latency_ms = (datetime.utcnow() - llm_call_start).total_seconds() * 1000
            logger.debug(f"Received response from Anthropic model '{anthropic_params.get('model')}'.")

        except (APITimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"Anthropic request timed out for provider {self.provider_id}: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=True)
        except RateLimitError as e:
            logger.error(f"Anthropic rate limit exceeded for provider {self.provider_id}: {e}", exc_info=True)
            retry_after = self._get_retry_after(e)
            error_details = self._map_error(e, retryable=True, retry_after=retry_after)
        except APIStatusError as e: # Specific HTTP status errors
             logger.error(f"Anthropic API error for provider {self.provider_id}: Status={e.status_code}, Response={e.response.text}", exc_info=True)
             # Retry based on status code
             retryable = e.status_code in [429, 500, 502, 503, 504]
             error_details = self._map_error(e, retryable=retryable)
        except APIError as e: # Catch-all for other Anthropic API errors
            logger.error(f"Anthropic API error for provider {self.provider_id}: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=False) # Assume non-retryable unless specific
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic generate for provider {self.provider_id}: {e}", exc_info=True)
            error_details = self._map_error(e, retryable=False)

        # 3. Map Anthropic Response back to Gateway LLMResponse
        total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        final_response = self._map_response(
            anthropic_response=response_obj,
            original_request=request,
            error_details=error_details,
            llm_latency_ms=llm_latency_ms,
            total_duration_ms=total_duration_ms,
        )

        return final_response

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response from Anthropic using the Messages API.
        Yields gateway StreamChunk objects.
        """
        start_time = datetime.utcnow() # For potential final chunk metrics
        request_id = request.initial_context.request_id
        chunk_index = 0
        accumulated_usage: Optional[UsageStats] = None
        final_finish_reason: Optional[FinishReason] = None
        last_tool_use_requests: Optional[List[ToolUseRequest]] = None

        if not self._client:
            logger.error(f"Anthropic client not initialized for streaming request {request_id}")
            yield self._create_error_chunk(
                request_id, 0, self._map_error(ConnectionError("Anthropic client not initialized."))
            )
            return

        try:
            # 1. Map request
            anthropic_params = self._map_request(request)
            model_id = anthropic_params.get("model", "unknown_model")
            logger.debug(f"Starting stream request to Anthropic model '{model_id}'")

            # 2. Call Anthropic streaming API
            async with self._client.messages.stream(**anthropic_params) as stream: # type: ignore
                async for event in stream:
                    chunk_start_time = datetime.utcnow()
                    mapped_chunk = None

                    # 3. Map stream events to StreamChunk
                    if isinstance(event, ContentBlockDeltaEvent):
                        # Text delta
                        mapped_chunk = StreamChunk(
                            chunk_id=chunk_index,
                            request_id=request_id,
                            delta_text=event.delta.text,
                        )
                    elif isinstance(event, ContentBlockStartEvent):
                         # Check if it's a tool use block starting
                         if event.content_block.type == "tool_use":
                              # We accumulate tool info until MessageStop gives the final list
                              tool_use_info = event.content_block # Keep this info if needed
                              logger.debug(f"Stream: Tool use block started: {tool_use_info.name}")
                              # Don't yield a chunk for start event, wait for delta/stop
                         # else: text block started, ignore - wait for delta

                    elif isinstance(event, MessageStartEvent):
                        # Initial message metadata, potentially includes usage estimate
                        if hasattr(event.message, 'usage'):
                             # Anthropic includes input tokens here
                             initial_usage = self._map_usage(event.message.usage)
                             if initial_usage:
                                  accumulated_usage = initial_usage # Initialize usage
                                  mapped_chunk = StreamChunk(
                                       chunk_id=chunk_index, request_id=request_id, usage_update=initial_usage
                                  )

                    elif isinstance(event, MessageDeltaEvent):
                         # Contains stop_reason and potentially *output* token usage delta
                         # Note: This structure varies; check SDK/API docs
                         if event.delta and hasattr(event.delta, 'stop_reason'):
                              final_finish_reason = self._map_stop_reason(event.delta.stop_reason)
                         if hasattr(event, 'usage') and event.usage: # Contains output tokens for the delta
                              delta_output_tokens = event.usage.output_tokens
                              if accumulated_usage and delta_output_tokens > 0:
                                   # Update completion tokens (total will be calculated by model)
                                   # Create a *new* UsageStats object for the update
                                   updated_usage = UsageStats(
                                        prompt_tokens=accumulated_usage.prompt_tokens, # Keep prompt tokens
                                        completion_tokens=accumulated_usage.completion_tokens + delta_output_tokens
                                   )
                                   accumulated_usage = updated_usage # Update state
                                   # Yield separate usage chunk if needed, or attach to next content chunk?
                                   # For simplicity, let's yield it separately for now.
                                   mapped_chunk = StreamChunk(
                                        chunk_id=chunk_index, request_id=request_id, usage_update=updated_usage
                                   )

                    elif isinstance(event, MessageStopEvent):
                         # End of stream. Get final usage, stop reason, and map final tool calls.
                         # Note: Final usage might be in the Message object from the 'message' event context manager provides
                         final_message = await stream.get_final_message()
                         final_finish_reason = self._map_stop_reason(final_message.stop_reason)
                         accumulated_usage = self._map_usage(final_message.usage)

                         # Map final tool calls if the reason was tool_use
                         if final_finish_reason == FinishReason.TOOL_CALLS:
                              last_tool_use_requests = self._map_tool_calls(final_message.content, request.tools)

                         # Yield final chunk with finish reason and potentially final usage/tool calls
                         mapped_chunk = StreamChunk(
                              chunk_id=chunk_index,
                              request_id=request_id,
                              finish_reason=final_finish_reason,
                              usage_update=accumulated_usage,
                              delta_tool_calls=last_tool_use_requests # Attach final tool calls here
                         )

                    # elif isinstance(event, ContentBlockStopEvent):
                    #     # Usually not needed unless tracking block boundaries specifically
                    #     pass

                    # Yield the mapped chunk if one was created
                    if mapped_chunk:
                         # Optional: add timing info to chunk's provider_specific_data
                         # chunk_proc_time = (datetime.utcnow() - chunk_start_time).total_seconds() * 1000
                         # mapped_chunk.provider_specific_data = {"processing_time_ms": chunk_proc_time}
                         yield mapped_chunk
                         chunk_index += 1

        except (APITimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"Anthropic stream timed out for request {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=True))
        except RateLimitError as e:
            logger.error(f"Anthropic stream rate limit for request {request_id}: {e}", exc_info=True)
            retry_after = self._get_retry_after(e)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=True, retry_after=retry_after))
        except APIStatusError as e:
             logger.error(f"Anthropic stream API error for request {request_id}: Status={e.status_code}, Response={e.response.text if e.response else 'N/A'}", exc_info=True)
             retryable = e.status_code in [429, 500, 502, 503, 504]
             yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=retryable))
        except APIError as e: # Catch-all for other Anthropic API errors
            logger.error(f"Anthropic stream API error for request {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic stream for request {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, chunk_index, self._map_error(e, retryable=False))

        finally:
             # Ensure a final chunk with finish reason is yielded if stream ended abruptly without MessageStopEvent
             # This part is tricky; relies on state captured during iteration.
             # If the loop finished normally, MessageStopEvent should have handled the final chunk.
             # If an error occurred, the error chunk is yielded.
             logger.debug(f"Anthropic stream finished for request {request_id}.")


    # --- Mapping Helper Methods ---

    def _map_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Maps the gateway LLMRequest to Anthropic's messages.create parameters."""
        anthropic_params: Dict[str, Any] = {}

        # 1. Model Identifier
        # Model name might be directly in config or need mapping from a generic ID
        anthropic_params["model"] = request.config.model_identifier # Assuming it's the correct Anthropic model ID

        # 2. Messages (History + Current Prompt)
        messages: List[MessageParam] = []
        # Handle System Prompt (must be top-level)
        system_prompt = request.config.system_prompt
        # Check if last history turn was system prompt (less common for messages API)
        if not system_prompt and request.initial_context.conversation_history:
            first_turn = request.initial_context.conversation_history[0]
            if first_turn.role == GatewayRole.SYSTEM.value and isinstance(first_turn.content, str):
                 system_prompt = first_turn.content # Use first system message if no explicit system prompt
                 # Don't add this turn to messages list below

        if system_prompt:
            anthropic_params["system"] = system_prompt

        # Map conversation history
        for turn in request.initial_context.conversation_history:
             role = self._map_role(turn.role)
             if not role: # Skip system (handled above) or unmappable roles
                  continue
             
             content = self._map_content_to_anthropic(turn.content)
             if content:
                  messages.append(MessageParam(role=role, content=content)) # type: ignore

        # Map current prompt content
        current_content = self._map_content_to_anthropic(request.prompt_content)
        if current_content:
            # Ensure last message is from 'user'
            if messages and messages[-1]["role"] == "user":
                 # Append content to the last user message if possible (if both are text)
                 # This is complex; simpler to just add a new user message turn
                 logger.debug("Multiple consecutive user messages detected. Appending as new turn.")
            messages.append(MessageParam(role="user", content=current_content)) # type: ignore
        elif not messages:
             # Handle case where history is empty and prompt is empty/unmappable
             raise ValueError("Cannot send request to Anthropic with no valid user message content.")

        # Ensure the conversation starts with 'user' if history was empty
        if messages and messages[0]["role"] != "user":
             # This indicates an issue, likely history started with 'assistant' without preceding 'user'
             # Anthropic API requires alternating user/assistant roles
             logger.warning("Anthropic messages do not start with 'user' role. API may reject.")
             # Depending on strictness, might need to insert a dummy user message or raise error

        anthropic_params["messages"] = messages

        # 3. Configuration Parameters
        # Ensure max_tokens is set, as it's required by Anthropic
        anthropic_params["max_tokens"] = request.config.max_tokens or 4096 # Default if not set
        if request.config.temperature is not None: anthropic_params["temperature"] = request.config.temperature
        if request.config.top_p is not None: anthropic_params["top_p"] = request.config.top_p
        if request.config.stop_sequences: anthropic_params["stop_sequences"] = request.config.stop_sequences
        # Map other supported params like top_k if needed

        # 4. Tools
        if request.tools:
             mapped_tools = self._map_tools_to_anthropic(request.tools)
             if mapped_tools:
                  anthropic_params["tools"] = mapped_tools
                  # Add tool_choice if specified (e.g., {"type": "auto"} or {"type": "tool", "name": "..."})
                  # anthropic_params["tool_choice"] = {"type": "auto"} # Example

        # Add metadata if needed (Anthropic supports a metadata field)
        # anthropic_params["metadata"] = {"gateway_request_id": request.initial_context.request_id}

        return anthropic_params

    def _map_role(self, gateway_role_str: str) -> Optional[AnthropicRole]:
        """
        Maps gateway role string to the AnthropicRole enum.

        Returns AnthropicRole.USER or AnthropicRole.ASSISTANT, or None if the
        gateway role shouldn't be directly included in the Anthropic 'messages' list
        (e.g., 'system', 'tool').

        Args:
            gateway_role_str: The role string (e.g., "user", "assistant", "system", "tool")
                              from gateway models.

        Returns:
            The corresponding AnthropicRole enum member, or None.
        """
        try:
            # Convert the input string to the GatewayRole Enum member
            gateway_role_enum = GatewayRole(gateway_role_str)
        except ValueError:
            logger.warning(f"Unknown gateway role '{gateway_role_str}' cannot be mapped to an Anthropic message role.")
            return None

        # Map to Anthropic roles
        if gateway_role_enum == GatewayRole.USER:
            return AnthropicRole.USER
        elif gateway_role_enum == GatewayRole.ASSISTANT:
            return AnthropicRole.ASSISTANT
        elif gateway_role_enum == GatewayRole.SYSTEM:
            # System role is handled by the top-level 'system' parameter.
            logger.debug("Skipping mapping for 'system' role in _map_role (handled separately).")
            return None
        elif gateway_role_enum == GatewayRole.TOOL:
            # Tool results need to be formatted into a 'user' message with tool_result content.
            logger.debug("Skipping mapping for 'tool' role in _map_role (handled during tool result construction).")
            return None
        else:
            logger.warning(f"Gateway role '{gateway_role_enum.value}' has no defined mapping to an Anthropic message role.")
            return None
        
    def _map_content_to_anthropic(self, content: Union[str, List[ContentItem], Dict[str, Any]]) -> Union[str, List[Union[TextBlockParam, ImageBlockParam]], None]:
         """Maps gateway content to Anthropic message content format."""
         if isinstance(content, str):
              return content # Simple text
         elif isinstance(content, list):
              # Multimodal content
              blocks: List[Union[TextBlockParam, ImageBlockParam]] = []
              for item in content:
                   if item.type == GatewayContentType.TEXT and item.text_content:
                        blocks.append(TextBlockParam(type="text", text=item.text_content))
                   elif item.type == GatewayContentType.IMAGE:
                        try:
                             img_data, mime_type = self._extract_image_data(item)
                             blocks.append(ImageBlockParam(
                                  type="image",
                                  source={
                                       "type": "base64",
                                       "media_type": mime_type,
                                       "data": img_data,
                                  }
                             ))
                        except ValueError as e:
                             logger.warning(f"Skipping image content item due to error: {e}")
                   # Add other content types if Anthropic supports them (e.g., audio)
                   else:
                        logger.warning(f"Skipping unsupported content item type for Anthropic: {item.type}")
              return blocks if blocks else None
         elif isinstance(content, dict):
              # Handle tool results if they are passed as dict content in TOOL role turns
              # Anthropic expects tool results as a separate user message containing a tool_result block
              logger.warning(f"Mapping dict content is ambiguous for Anthropic. Type: {content.get('type', 'N/A')}. Skipping.")
              return None # Or try to serialize if appropriate context allows
         else:
              logger.warning(f"Unsupported content type for Anthropic mapping: {type(content)}")
              return None

    def _extract_image_data(self, item: ContentItem) -> Tuple[str, str]:
        """Extracts base64 data and MIME type from an Image ContentItem."""
        mime_type = item.mime_type
        source = item.data.get("image", {}).get("source", {})
        source_type = source.get("type")

        if source_type == "base64":
            b64_data = source.get("data")
            if not mime_type: raise ValueError("MIME type missing for base64 image")
            if not b64_data: raise ValueError("Base64 data missing")
            return b64_data, mime_type
        elif source_type == "url":
             # Fetching URL data is generally discouraged in providers for security/performance.
             # If needed, implement carefully with timeouts and error handling.
             raise ValueError("Image URLs are not directly supported; provide base64 data.")
        # Add handling for other potential sources if needed (e.g., file paths - requires reading)
        else:
            raise ValueError(f"Unsupported image source type: {source_type}")

    def _map_tools_to_anthropic(self, tools: List[ToolDefinition]) -> List[ToolParam]:
        """Maps gateway ToolDefinition list to Anthropic ToolParam list."""
        anthropic_tools: List[ToolParam] = []
        for tool in tools:
            try:
                func = tool.function
                # Anthropic requires 'input_schema' following JSON Schema format
                input_schema = func.parameters or {"type": "object", "properties": {}} # Default if no params
                
                anthropic_tools.append(ToolParam(
                    name=func.name,
                    description=func.description,
                    input_schema=cast(Dict[str, Any], input_schema), # Cast needed as ToolParam expects specific dict type
                ))
            except Exception as e:
                logger.error(f"Failed to map tool '{getattr(tool, 'name', 'N/A')}' to Anthropic format: {e}", exc_info=True)
        return anthropic_tools

    def _map_response(
        self,
        anthropic_response: Optional[Message],
        original_request: LLMRequest,
        error_details: Optional[ErrorDetails],
        llm_latency_ms: Optional[float],
        total_duration_ms: Optional[float],
    ) -> LLMResponse:
        """Maps the Anthropic Message object back to the gateway's LLMResponse."""

        generated_content: Optional[Union[str, List[ContentItem]]] = None
        tool_use_requests: Optional[List[ToolUseRequest]] = None
        finish_reason: Optional[FinishReason] = FinishReason.UNKNOWN # Default
        usage: Optional[UsageStats] = None

        if error_details:
            finish_reason = FinishReason.ERROR
        elif anthropic_response:
            # Map Stop Reason
            finish_reason = self._map_stop_reason(anthropic_response.stop_reason)

            # Map Content / Tool Calls
            if finish_reason == FinishReason.TOOL_CALLS:
                 tool_use_requests = self._map_tool_calls(anthropic_response.content, original_request.tools)
            else:
                 # Extract text content (assuming primarily text responses for now)
                 text_content = ""
                 for block in anthropic_response.content:
                      if block.type == "text":
                           text_content += block.text
                 generated_content = text_content if text_content else None
                 # TODO: Handle potential non-text output blocks if needed

            # Map Usage
            usage = self._map_usage(anthropic_response.usage)

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
            raw_provider_response=anthropic_response.model_dump() if anthropic_response else None,
        )

    def _map_stop_reason(self, anthropic_reason: Optional[str]) -> FinishReason:
        """Maps Anthropic stop reason string to gateway FinishReason."""
        if not anthropic_reason:
            return FinishReason.UNKNOWN
        return ANTHROPIC_STOP_REASON_MAP.get(anthropic_reason, FinishReason.UNKNOWN)

    def _map_usage(self, anthropic_usage: Optional[AnthropicUsage]) -> Optional[UsageStats]:
        """Maps Anthropic Usage object to gateway UsageStats."""
        if not anthropic_usage:
            return None
        # Ensure tokens > 0 before creating stats object
        if anthropic_usage.input_tokens > 0 or anthropic_usage.output_tokens > 0:
             return UsageStats(
                  prompt_tokens=anthropic_usage.input_tokens,
                  completion_tokens=anthropic_usage.output_tokens,
                  # Total calculated automatically
             )
        return None

    def _map_tool_calls(self, content_blocks: List[Any], original_tools: Optional[List[ToolDefinition]]) -> List[ToolUseRequest]:
         """Maps Anthropic tool_use content blocks to gateway ToolUseRequest list."""
         tool_requests = []
         for block in content_blocks:
              if block.type == "tool_use":
                   try:
                        # Find original tool definition for potential schema/description enrichment
                        original_tool_def_func = next((t.function for t in original_tools or [] if t.function.name == block.name), None)

                        tool_requests.append(ToolUseRequest(
                             id=block.id,
                             type="function", # Assuming function calls
                             function=ToolFunction(
                                  name=block.name,
                                  description=original_tool_def_func.description if original_tool_def_func else None,
                                  parameters=block.input, # Anthropic provides the input args here
                             )
                        ))
                   except Exception as e:
                        logger.error(f"Failed to map Anthropic tool_use block (id: {getattr(block, 'id', 'N/A')}, name: {getattr(block, 'name', 'N/A')}): {e}", exc_info=True)
         return tool_requests

    def _map_error(self, error: Exception, retryable: Optional[bool] = None, retry_after: Optional[int] = None) -> ErrorDetails:
         """Maps various exceptions to the gateway's ErrorDetails model."""
         code = "PROVIDER_ERROR"
         message = f"Anthropic provider '{self.provider_id}' encountered an error: {str(error)}"
         level = ErrorLevel.ERROR
         provider_details: Optional[Dict[str, Any]] = None
         is_retryable = retryable if retryable is not None else False # Default to False

         if isinstance(error, APITimeoutError):
              code = "PROVIDER_TIMEOUT"
              message = f"Anthropic request timed out after {self._timeout}s."
              is_retryable = True # Often retryable
         elif isinstance(error, RateLimitError):
              code = "PROVIDER_RATE_LIMIT"
              message = "Anthropic rate limit exceeded."
              is_retryable = True
         elif isinstance(error, APIStatusError):
              code = f"PROVIDER_HTTP_{error.status_code}"
              message = f"Anthropic API returned status {error.status_code}: {error.message or error.response.text}"
              if error.status_code == 400: level = ErrorLevel.WARNING # Bad request often user error
              elif error.status_code == 401: level = ErrorLevel.ERROR # Auth error
              elif error.status_code == 403: level = ErrorLevel.ERROR # Permissions
              elif error.status_code == 429: is_retryable = True; code = "PROVIDER_RATE_LIMIT"
              elif error.status_code >= 500: is_retryable = True # Server errors are potentially retryable
              # Extract details if available
              try:
                  provider_details = error.response.json() if error.response else None
              except: # nosec
                  provider_details = {"raw_response": error.response.text if error.response else None}
         elif isinstance(error, APIError):
              code = "PROVIDER_API_ERROR"
              message = f"Anthropic API Error: {str(error)}"
         elif isinstance(error, ConnectionError):
              code = "PROVIDER_CONNECTION_ERROR"
              message = f"Could not connect to Anthropic API: {str(error)}"
              is_retryable = True # Network issues often retryable
         elif isinstance(error, ValueError): # e.g., mapping errors
             code = "PROVIDER_MAPPING_ERROR"
             message = f"Data mapping error for Anthropic: {str(error)}"
             level = ErrorLevel.WARNING # Often indicates bad input format

         return ErrorDetails(
              code=code,
              message=message,
              level=level,
              provider_error_details=provider_details or {"exception_type": type(error).__name__},
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

    def _get_retry_after(self, error: RateLimitError) -> Optional[int]:
         """Extract retry-after header if available from RateLimitError."""
         # The SDK might parse this, check its attributes or the raw response headers
         if error.response and 'retry-after' in error.response.headers:
              try:
                   return int(error.response.headers['retry-after'])
              except (ValueError, TypeError):
                   logger.warning("Could not parse 'retry-after' header value.")
         # Check if SDK exposes it directly (hypothetical)
         # if hasattr(error, 'retry_after_seconds'):
         #    return error.retry_after_seconds
         return None

# --- End of AnthropicClient class ---