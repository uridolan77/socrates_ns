# llm_gateway/providers/mcp_provider.py

"""
Provider implementation using the external Model Context Protocol (MCP) standard,
based on the modelcontextprotocol/python-sdk.
"""

import asyncio
import logging
import os
import json
import uuid
import shutil
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast
from src.llm_gateway.providers.base import BaseProvider

# --- MCP SDK Imports ---
try:
    from mcp import ClientSession, StdioServerParameters
    import mcp.types as mcp_types
    from mcp.client.stdio import stdio_client
    from mcp.shared.exceptions import McpError
except ImportError:
    # Provide dummy types if MCP SDK is not installed, to allow basic loading
    # In a real scenario, this dependency should be mandatory
    logging.warning("MCP SDK not found, using placeholder types.")
    class ClientSession: pass
    class StdioServerParameters: pass
    class mcp_types: # type: ignore
        class SamplingMessage: pass
        class TextContent: pass
        class ImageContent: pass
        class Role: pass
        class CreateMessageResult: pass
        class Tool: pass
        class CallToolResult: pass
        class StopReason: pass
# --- End MCP SDK Imports ---

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
    MCPContentType as GatewayMCPContentType,
    MCPMetadata,
    MCPRole as GatewayMCPRole,
    PerformanceMetrics,
    ProviderConfig,
    StreamChunk,
    ToolFunction,
    ToolDefinition,
    ToolUseRequest,
    UsageStats,
    MCPUsage
)
from llm_gateway.providers.base import BaseProvider

logger = logging.getLogger(__name__)

class MCPProvider(BaseProvider):
    """
    LLM Gateway provider implementation using the MCP standard via the official SDK.
    Connects to an MCP server process using stdio.
    """

    def __init__(self, provider_config: ProviderConfig, gateway_config: GatewayConfig):
        """
        Initialize the MCP Provider.

        Args:
            provider_config: Configuration specific to this provider instance
                             (must contain connection details like command/args).
            gateway_config: Global gateway configuration.
        """
        super().__init__(provider_config)
        self.gateway_config = gateway_config
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Extract connection params from ProviderConfig
        self._server_command = provider_config.connection_params.get("command", "npx")
        self._server_args = provider_config.connection_params.get("args", [])
        self._server_env = provider_config.connection_params.get("env", {})
        
        # Additional options
        self._enable_streaming = provider_config.connection_params.get("enable_streaming", False)
        self._max_retry_attempts = provider_config.connection_params.get("max_retry_attempts", 2)
        self._retry_delay_seconds = provider_config.connection_params.get("retry_delay_seconds", 1.0)
        
        logger.info(f"Initialized MCP provider '{self.provider_id}' with command: {self._server_command}")

    def _initialize_client(self) -> None:
        # Session management happens on demand via _ensure_session
        logger.debug("MCPProvider client initialization deferred to session management.")
        pass

    @asynccontextmanager
    async def _ensure_session(self) -> AsyncGenerator[ClientSession, None]:
        """Ensures an active MCP session is available, establishing one if needed."""
        async with self._session_lock:
            if self._session is None:
                logger.info(f"No active MCP session for {self.provider_id}. Initializing...")
                try:
                    # Resolve command path
                    command = (
                        shutil.which(self._server_command)
                        if self._server_command == "npx"
                        else self._server_command
                    )
                    if command is None:
                         raise ValueError(f"MCP server command not found: {self._server_command}")

                    server_params = StdioServerParameters(
                        command=command,
                        args=self._server_args,
                        # Combine OS env, configured env, and API keys
                        env={**os.environ, **self._server_env, **self._get_auth_env()},
                    )

                    # Enter contexts using the exit stack
                    stdio_transport = await self._exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    read, write = stdio_transport
                    session = await self._exit_stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    self._session = session
                    logger.info(f"MCP session initialized successfully for {self.provider_id}.")
                except Exception as e:
                     logger.error(f"Failed to initialize MCP session for {self.provider_id}: {e}", exc_info=True)
                     # Ensure stack is cleaned up if init fails
                     await self._exit_stack.aclose()
                     self._session = None
                     raise

            # Yield the active session
            yield self._session

    async def cleanup(self):
         """Cleans up the MCP session and transport."""
         logger.info(f"Cleaning up MCP session for {self.provider_id}...")
         async with self._session_lock:
              await self._exit_stack.aclose()
              self._session = None
         logger.info(f"MCP session cleanup complete for {self.provider_id}.")

    def _get_auth_env(self) -> Dict[str, str]:
         """Get API keys/auth tokens for the MCP server environment."""
         # Check for provider-specific API key first, then fallback to generic MCP key
         api_key = os.environ.get(f"PROVIDER_API_KEY_{self.provider_id.upper()}") or os.environ.get("MCP_SERVER_API_KEY")
         if api_key:
              # Key name depends on what the specific MCP server process expects
              return {"MCP_SERVER_API_KEY": api_key, "LLM_API_KEY": api_key}
         return {}

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using MCP non-streaming `create_message`."""
        start_time = datetime.utcnow()
        llm_latency_ms = None
        mcp_result: Optional[mcp_types.CreateMessageResult] = None
        error_details: Optional[ErrorDetails] = None
        session_instance: Optional[ClientSession] = None

        try:
            # 1. Ensure active session
            async with self._ensure_session() as session:
                session_instance = session

                # 2. Map gateway request to MCP SamplingMessages
                mcp_messages = self._map_to_mcp_sampling_messages(request)

                # 3. Prepare MCP parameters from LLMConfig
                mcp_params = self._prepare_mcp_sampling_params(request.config)

                # 4. Call MCP API (sampling/createMessage)
                logger.debug(f"Sending createMessage to MCP model {request.config.model_identifier}...")
                llm_call_start = datetime.utcnow()
                
                # Try with retries for transient errors
                attempt = 0
                while True:
                    try:
                        mcp_result = await session.create_message(
                            messages=mcp_messages,
                            max_tokens=mcp_params.pop("max_tokens"),
                            **mcp_params
                        )
                        break  # Success, exit retry loop
                    except McpError as e:
                        attempt += 1
                        if attempt <= self._max_retry_attempts and self._is_retryable_error(e):
                            logger.warning(
                                f"Retryable error on attempt {attempt}/{self._max_retry_attempts}: {e}. "
                                f"Retrying in {self._retry_delay_seconds}s..."
                            )
                            await asyncio.sleep(self._retry_delay_seconds)
                        else:
                            # Max retries reached or non-retryable error
                            raise
                            
                llm_latency_ms = (datetime.utcnow() - llm_call_start).total_seconds() * 1000
                logger.debug(f"Received createMessage result from MCP model {request.config.model_identifier}.")

        except McpError as e:
             logger.error(f"MCPError during generate for provider {self.provider_id}: {e.error}", exc_info=True)
             error_details = ErrorDetails(
                  code=str(e.error.code),
                  message=e.error.message,
                  level=ErrorLevel.ERROR,
                  provider_error_details=e.error.data,
                  retryable=self._is_retryable_error(e)
             )
        except Exception as e:
            logger.error(f"Error during generate for provider {self.provider_id}: {e}", exc_info=True)
            error_details = ErrorDetails(
                code="PROVIDER_REQUEST_FAILED",
                message=f"MCP provider '{self.provider_id}' request failed: {str(e)}",
                level=ErrorLevel.ERROR,
                retryable=False
            )

        # Map MCP result back to gateway LLMResponse
        total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        final_context = request.initial_context

        response = self._map_from_mcp_create_message_result(
            mcp_result=mcp_result,
            original_request=request,
            final_context_state=final_context,
            error_details=error_details,
            llm_latency_ms=llm_latency_ms,
            total_duration_ms=total_duration_ms,
            mcp_session=session_instance
        )

        return response

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Generate response using MCP streaming."""
        # MCP SDK examples primarily show request/response via create_message
        # Streaming implementation would depend on MCP spec details
        
        if not self._enable_streaming:
            logger.warning(f"Streaming requested but disabled for {self.provider_id}. Using non-streaming fallback.")
            
            # Fallback: Call non-streaming and yield single chunk
            response = await self.generate(request)
            if response.error_details:
                # Yield an error chunk
                yield StreamChunk(
                     chunk_id=0,
                     request_id=request.initial_context.request_id,
                     finish_reason=FinishReason.ERROR,
                     provider_specific_data={"error": response.error_details.model_dump()}
                )
            else:
                yield StreamChunk(
                     chunk_id=0,
                     request_id=request.initial_context.request_id,
                     delta_text=cast(Optional[str], response.generated_content),
                     delta_tool_calls=None,
                     finish_reason=response.finish_reason,
                     usage_update=response.usage,
                     provider_specific_data=response.mcp_metadata.model_dump() if response.mcp_metadata else None
                )
        else:
            # This is a placeholder for actual MCP streaming implementation
            # The real implementation would depend on the MCP spec and SDK capabilities
            raise NotImplementedError(
                "Native MCP streaming not implemented. Set enable_streaming=False to use fallback mode."
            )

    def _is_retryable_error(self, error: McpError) -> bool:
        """Determine if an MCP error is retryable."""
        # Example of error code classification - adapt based on actual MCP error codes
        retryable_codes = [
            "RATE_LIMIT_EXCEEDED",
            "TEMPORARY_SERVER_ERROR", 
            "SERVICE_UNAVAILABLE",
            "TIMEOUT",
            429,  # Rate limit
            500,  # Server error
            503,  # Service unavailable
            504,  # Gateway timeout
        ]
        
        if hasattr(error, 'error') and hasattr(error.error, 'code'):
            return error.error.code in retryable_codes
        return False

    # --- Mapping Functions ---

    def _map_to_mcp_sampling_messages(self, request: LLMRequest) -> List[mcp_types.SamplingMessage]:
        """Convert gateway request history and prompt to MCP SamplingMessage list."""
        mcp_messages: List[mcp_types.SamplingMessage] = []

        # Map history turns
        for turn in request.initial_context.conversation_history:
            try:
                # Map role
                role = self._map_gateway_role_to_mcp(turn.role)
                if role is None:
                    logger.warning(f"Skipping history turn with unmappable role '{turn.role}': {turn.turn_id}")
                    continue

                # Map content - handle potential multimodality / complexity
                content_obj: Optional[Union[mcp_types.TextContent, mcp_types.ImageContent]] = None
                
                if isinstance(turn.content, str):
                    content_obj = mcp_types.TextContent(type="text", text=turn.content)
                elif isinstance(turn.content, list) and len(turn.content) > 0:
                    # Handle first item only for simplicity
                    first_item = turn.content[0]
                    if first_item.type == GatewayMCPContentType.TEXT:
                        content_obj = mcp_types.TextContent(type="text", text=cast(str, first_item.text_content))
                    elif first_item.type == GatewayMCPContentType.IMAGE:
                        content_obj = self._map_gateway_image_to_mcp(first_item)
                    else:
                        logger.warning(f"Skipping unsupported content type '{first_item.type}' in history turn {turn.turn_id}")
                elif isinstance(turn.content, dict) and turn.role == GatewayMCPRole.TOOL.value:
                    # Skip tool results for create_message context
                    logger.debug(f"Skipping TOOL history turn {turn.turn_id} for create_message context.")
                    continue
                else:
                    logger.warning(f"Cannot map history content type {type(turn.content)} for turn {turn.turn_id}")

                if content_obj:
                    mcp_messages.append(mcp_types.SamplingMessage(role=role, content=content_obj))

            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping history turn due to mapping error: {e} - Turn: {turn.turn_id}")
                continue

        # Map current prompt content
        current_content_items: List[ContentItem]
        if isinstance(request.prompt_content, str):
            current_content_items = [ContentItem(type=GatewayMCPContentType.TEXT, data={"text": request.prompt_content})]
        elif isinstance(request.prompt_content, list):
            current_content_items = request.prompt_content
        else:
            raise ValueError(f"Invalid prompt_content type: {type(request.prompt_content)}")

        for item in current_content_items:
            try:
                content_obj: Optional[Union[mcp_types.TextContent, mcp_types.ImageContent]] = None
                if item.type == GatewayMCPContentType.TEXT:
                    content_obj = mcp_types.TextContent(type="text", text=cast(str, item.text_content))
                elif item.type == GatewayMCPContentType.IMAGE:
                    content_obj = self._map_gateway_image_to_mcp(item)
                else:
                    logger.warning(f"Skipping unsupported prompt content type '{item.type}'")

                if content_obj:
                    # Assume current prompt maps to USER role
                    mcp_messages.append(mcp_types.SamplingMessage(role="user", content=content_obj))
            except ValueError as e:
                logger.warning(f"Skipping prompt content item due to mapping error: {e}")
                continue

        if not mcp_messages:
            # Ensure there's at least one message
            logger.warning("Mapped MCP message list is empty. Adding empty user message as fallback.")
            mcp_messages.append(mcp_types.SamplingMessage(role="user", content=mcp_types.TextContent(type="text", text="")))
        elif mcp_messages[-1].role != "user":
            # Ensure the last message is from the user to follow MCP conventions
            logger.warning("Last message in MCP sequence isn't from user. This may not be supported by all MCP services.")

        return mcp_messages

    def _map_gateway_role_to_mcp(self, gateway_role: str) -> Optional[str]:
        """Map gateway role to MCP role."""
        role_mapping = {
            "user": "user",
            "assistant": "assistant",
            "system": "system" if hasattr(mcp_types.Role, "SYSTEM") else None,
            "tool": None,  # Most MCP implementations don't have a tool role
        }
        
        return role_mapping.get(gateway_role.lower())

    def _map_gateway_image_to_mcp(self, item: ContentItem) -> mcp_types.ImageContent:
        """Maps a gateway ContentItem (Image) to mcp.types.ImageContent."""
        b64_data: Optional[str] = None
        mime_type = item.mime_type or "image/png"

        if item.uri:
            # Handle URI-based images
            if item.uri.startswith("data:"):
                # Extract base64 from data URI
                _, b64_data = item.uri.split(',', 1)
            elif item.uri.startswith("file://"):
                # Read local file
                try:
                    import base64
                    from pathlib import Path
                    file_path = item.uri.replace("file://", "")
                    b64_data = base64.b64encode(Path(file_path).read_bytes()).decode()
                except Exception as e:
                    raise ValueError(f"Failed to read image file URI {item.uri}: {e}") from e
            else:
                raise ValueError(f"Cannot map non-file image URI to base64: {item.uri}")

        elif isinstance(item.data, dict):
            if "base64" in item.data:
                b64_data = item.data["base64"]
            elif "text" in item.data and item.data["text"].startswith("data:"):
                # Extract base64 from data URI in text field
                _, b64_data = item.data["text"].split(',', 1)
        elif isinstance(item.data, bytes):
            import base64
            b64_data = base64.b64encode(item.data).decode()

        if not b64_data:
            raise ValueError(f"Could not extract base64 data for image ContentItem: {item}")

        return mcp_types.ImageContent(type="image", data=b64_data, mimeType=mime_type)

    def _prepare_mcp_sampling_params(self, config: LLMConfig) -> Dict[str, Any]:
        """Maps gateway LLMConfig to MCP create_message parameters."""
        params = {}
        
        # Handle required max_tokens parameter
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        else:
            # Default to a reasonable value if not specified
            params["max_tokens"] = 1024
            logger.warning(f"No max_tokens specified, defaulting to {params['max_tokens']}")

        # Map other parameters that MCP supports
        if config.temperature is not None:
            params["temperature"] = config.temperature
            
        if config.stop_sequences is not None:
            params["stop_sequences"] = config.stop_sequences
            
        # System prompt is handled separately in some MCP implementations
        if hasattr(config, "system_prompt") and getattr(config, "system_prompt"):
            params["systemPrompt"] = getattr(config, "system_prompt")
            
        # Add model identifier if provided
        if config.model_identifier:
            # This might be handled differently depending on MCP implementation
            # Some might use metadata, others might need specific parameters
            params["metadata"] = params.get("metadata", {})
            params["metadata"]["model"] = config.model_identifier

        # Log unmapped parameters for debugging
        unmapped_params = []
        if config.top_p is not None: unmapped_params.append("top_p")
        if config.presence_penalty is not None: unmapped_params.append("presence_penalty")
        if config.frequency_penalty is not None: unmapped_params.append("frequency_penalty")
        
        if unmapped_params:
            logger.debug(f"LLMConfig parameters not mapped to MCP: {', '.join(unmapped_params)}")

        return params

    def _map_from_mcp_create_message_result(
        self,
        mcp_result: Optional[mcp_types.CreateMessageResult],
        original_request: LLMRequest,
        final_context_state: InterventionContext,
        error_details: Optional[ErrorDetails],
        llm_latency_ms: Optional[float],
        total_duration_ms: Optional[float],
        mcp_session: Optional[ClientSession]
    ) -> LLMResponse:
        """Maps the mcp.types.CreateMessageResult back to the gateway's LLMResponse."""

        finish_reason: Optional[FinishReason] = None
        gateway_usage: Optional[UsageStats] = None
        mcp_meta: Optional[MCPMetadata] = None
        generated_content: Optional[Union[str, List[ContentItem]]] = None
        tool_use_reqs: Optional[List[ToolUseRequest]] = None

        if mcp_result:
            # Map Stop Reason
            finish_reason = self._map_mcp_stop_reason_to_gateway(mcp_result.stopReason)

            # Map Content
            if hasattr(mcp_result, 'content'):
                content_block = mcp_result.content
                gateway_items = self._map_mcp_content_block_to_gateway(content_block)
                if len(gateway_items) == 1 and gateway_items[0].type == GatewayMCPContentType.TEXT:
                    generated_content = gateway_items[0].text_content
                elif gateway_items:
                    generated_content = gateway_items

            # Create usage stats if available
            # MCP doesn't appear to return token counts directly in CreateMessageResult
            gateway_usage = None

            # Create MCP Metadata
            mcp_meta = MCPMetadata(
                model_version_reported=getattr(mcp_result, 'model', None),
                context_id=None,  # Not directly available in CreateMessageResult
                provider_usage=None  # Set if available
            )

            # Check for tool calls finish reason
            if finish_reason == FinishReason.TOOL_CALLS:
                # This is a placeholder - MCP spec needed for actual implementation
                tool_use_reqs = []
                logger.debug("Tool calls detected in MCP response, but mapping is not implemented")

        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            total_duration_ms=total_duration_ms,
            llm_latency_ms=llm_latency_ms,
        )

        return LLMResponse(
            version=original_request.version,
            request_id=original_request.initial_context.request_id,
            generated_content=generated_content,
            finish_reason=finish_reason,
            tool_use_requests=tool_use_reqs,
            usage=gateway_usage,
            compliance_result=None,  # Filled by interventions
            final_context=final_context_state,
            error_details=error_details,
            performance_metrics=perf_metrics,
            mcp_metadata=mcp_meta,
        )
        
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Generate response using MCP streaming."""
        if not self._enable_streaming:
            logger.warning(f"Streaming requested but disabled for {self.provider_id}. Using non-streaming fallback.")
            # Fallback: Call non-streaming and yield single chunk or error chunk
            response = await self.generate(request)
            if response.error_details:
                yield StreamChunk(
                    chunk_id=0,
                    request_id=request.initial_context.request_id,
                    finish_reason=FinishReason.ERROR,
                    provider_specific_data={"error": response.error_details.model_dump()}
                )
                return # Ensure generator stops after error
            else:
                # Determine content type for delta
                delta_text = None
                delta_content_items = None
                if isinstance(response.generated_content, str):
                     delta_text = response.generated_content
                elif isinstance(response.generated_content, list):
                     delta_content_items = response.generated_content
                
                yield StreamChunk(
                    chunk_id=0,
                    request_id=request.initial_context.request_id,
                    delta_text=delta_text,
                    delta_content_items=delta_content_items,
                    delta_tool_calls=response.tool_use_requests, # Include tool calls if any
                    finish_reason=response.finish_reason,
                    usage_update=response.usage,
                    provider_specific_data=response.mcp_metadata.model_dump() if response.mcp_metadata else None
                )
                return # Ensure generator stops after yielding final chunk

        else:
            # --- Placeholder for Actual MCP Streaming Implementation ---
            # This would involve:
            # 1. Ensuring the session is active.
            # 2. Mapping the request to MCP format (messages, params, tools).
            # 3. Calling a hypothetical `session.stream_message` or similar method.
            # 4. Asynchronously iterating through the chunks yielded by the MCP SDK.
            # 5. Mapping each MCP stream chunk (delta text, tool call chunk, usage update, finish reason)
            #    to the gateway's `StreamChunk` model.
            # 6. Handling errors during streaming.
            # 7. Yielding the mapped `StreamChunk`.
            logger.warning("Native MCP streaming is enabled but not implemented. No chunks will be yielded.")
            # Example of yielding an error chunk to indicate lack of implementation
            yield StreamChunk(
                 chunk_id=0,
                 request_id=request.initial_context.request_id,
                 finish_reason=FinishReason.ERROR,
                 provider_specific_data={
                     "error": ErrorDetails(
                         code="NOT_IMPLEMENTED",
                         message="Native MCP streaming is not yet implemented for this provider.",
                         level=ErrorLevel.ERROR
                     ).model_dump()
                 }
            )
            # The following line is needed because an async generator must yield something
            # if it doesn't raise an exception. In a real implementation, you would
            # yield actual data or raise upon error.
            if False: # pragma: no cover
                 yield # type: ignore

    def _is_retryable_error(self, error: McpError) -> bool:
        """Determine if an MCP error is retryable."""
        # Refine based on actual MCP error codes and standard HTTP codes
        # Use codes defined in the MCP spec if available
        retryable_codes = [
            # Standard HTTP-like codes
            429,  # Too Many Requests
            500,  # Internal Server Error (sometimes retryable)
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
            # Custom MCP codes (hypothetical examples)
            "RATE_LIMIT_EXCEEDED",
            "TEMPORARY_SERVER_ERROR",
            "SERVICE_UNAVAILABLE",
            "TIMEOUT",
            "RESOURCE_EXHAUSTED",
        ]

        if hasattr(error, 'error') and hasattr(error.error, 'code'):
             # Check if code is string or int
             error_code = error.error.code
             # Normalize common string codes if needed
             if isinstance(error_code, str):
                  error_code = error_code.upper()

             return error_code in retryable_codes
        # Add checks for network errors if McpError wraps them
        elif isinstance(error.__cause__, (asyncio.TimeoutError, ConnectionRefusedError)):
             return True # Network errors are often retryable

        return False

    # --- Mapping Functions ---

    def _map_to_mcp_sampling_messages(self, request: LLMRequest) -> List[mcp_types.SamplingMessage]:
        """Convert gateway request history and prompt to MCP SamplingMessage list."""
        # (Implementation as provided before - can be enhanced to handle multiple content items per turn)
        # ... (Keep the existing implementation) ...
        # Ensure the return type matches List[mcp_types.SamplingMessage]
        # Example minimal return for testing:
        mcp_messages: List[mcp_types.SamplingMessage] = []
        # ... (Add the actual mapping logic from the previous snippet here) ...
        # Map history
        # Map prompt
        # Handle edge cases (empty list, last message role)
        # For now, returning a simple structure to allow compilation
        if isinstance(request.prompt_content, str):
             mcp_messages.append(mcp_types.SamplingMessage(role="user", content=mcp_types.TextContent(type="text", text=request.prompt_content)))
        elif isinstance(request.prompt_content, list):
            # Simplified: take the first text item if available
            text_item = next((item for item in request.prompt_content if item.type == GatewayMCPContentType.TEXT), None)
            if text_item and text_item.text_content:
                 mcp_messages.append(mcp_types.SamplingMessage(role="user", content=mcp_types.TextContent(type="text", text=text_item.text_content)))
            else: # Fallback if no text found
                 mcp_messages.append(mcp_types.SamplingMessage(role="user", content=mcp_types.TextContent(type="text", text="")))

        return mcp_messages # Return the fully mapped list

    # ... (_map_gateway_role_to_mcp, _map_gateway_image_to_mcp - Keep existing implementations) ...

    def _prepare_mcp_sampling_params(self, config: LLMConfig, tools: Optional[List[ToolDefinition]] = None) -> Dict[str, Any]:
        """Maps gateway LLMConfig and Tools to MCP create_message parameters."""
        params = {}

        # Handle required max_tokens parameter
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        else:
            params["max_tokens"] = 1024 # Default if not specified
            logger.warning(f"No max_tokens specified, defaulting to {params['max_tokens']}")

        # Map other parameters that MCP supports
        if config.temperature is not None: params["temperature"] = config.temperature
        if config.stop_sequences is not None: params["stop_sequences"] = config.stop_sequences
        if config.top_p is not None: params["top_p"] = config.top_p # Assume MCP supports topP

        # System prompt handling might vary - assume it's a top-level param or message
        if config.system_prompt:
            # Option 1: Top-level parameter (if supported by MCP spec)
            # params["systemPrompt"] = config.system_prompt
            # Option 2: Prepend as a system message (handled in _map_to_mcp_sampling_messages if needed)
            pass # Assuming handled in message mapping if necessary

        # --- Tool Mapping ---
        if tools:
            mcp_tools: List[mcp_types.Tool] = []
            try:
                for tool_def in tools:
                    # Map ToolDefinition -> mcp_types.Tool (Hypothetical structure)
                    if hasattr(mcp_types, 'Tool') and hasattr(mcp_types.Tool, 'from_dict'): # Check if SDK has helper
                         mcp_tool = mcp_types.Tool.from_dict({
                             "name": tool_def.function.name,
                             "description": tool_def.function.description,
                             "input_schema": tool_def.function.parameters # Assume MCP takes schema directly
                         })
                         mcp_tools.append(mcp_tool)
                    else: # Manual mapping if no helper
                         mcp_tools.append(mcp_types.Tool(
                              name=tool_def.function.name,
                              description=tool_def.function.description,
                              input_schema=tool_def.function.parameters
                         ))
                if mcp_tools:
                    params["tools"] = mcp_tools
                    # Potentially add tool_choice parameter if needed based on MCP spec
                    # params["tool_choice"] = "auto" # or specific tool name
            except AttributeError:
                 logger.warning("MCP SDK types do not seem to support 'Tool'. Skipping tool mapping.")
            except Exception as e:
                 logger.error(f"Error mapping tools to MCP format: {e}", exc_info=True)


        # Add model identifier metadata
        params["metadata"] = params.get("metadata", {})
        if config.model_identifier:
            params["metadata"]["model"] = config.model_identifier
        # Add other potential metadata if needed

        # Log unmapped parameters
        unmapped_params = []
        # if config.top_p is not None and "top_p" not in params: unmapped_params.append("top_p") # Already mapped above
        if config.presence_penalty is not None: unmapped_params.append("presence_penalty")
        if config.frequency_penalty is not None: unmapped_params.append("frequency_penalty")

        if unmapped_params:
            logger.debug(f"LLMConfig parameters not directly mapped to MCP: {', '.join(unmapped_params)}")

        return params

    def _map_from_mcp_create_message_result(
        self,
        mcp_result: Optional[mcp_types.CreateMessageResult],
        original_request: LLMRequest,
        final_context_state: InterventionContext,
        error_details: Optional[ErrorDetails],
        llm_latency_ms: Optional[float],
        total_duration_ms: Optional[float],
        mcp_session: Optional[ClientSession] # Keep session for potential future use (e.g. context id)
    ) -> LLMResponse:
        """Maps the mcp.types.CreateMessageResult back to the gateway's LLMResponse."""

        finish_reason: Optional[FinishReason] = None
        gateway_usage: Optional[UsageStats] = None
        mcp_meta: Optional[MCPMetadata] = None
        generated_content: Optional[Union[str, List[ContentItem]]] = None
        tool_use_reqs: Optional[List[ToolUseRequest]] = None

        if error_details: # If an error occurred before getting a result
             finish_reason = FinishReason.ERROR
        elif mcp_result:
            # Map Stop Reason
            finish_reason = self._map_mcp_stop_reason_to_gateway(mcp_result.stopReason)

            # Map Content and Tool Calls based on stop reason
            if finish_reason == FinishReason.TOOL_CALLS:
                 # --- Tool Call Mapping ---
                 # Assume mcp_result.content contains tool call info when stopReason is tool_use
                 # Hypothetical structure: mcp_result.content might be a list of mcp_types.CallToolResult
                 tool_use_reqs = []
                 if isinstance(mcp_result.content, list):
                      for tool_call_block in mcp_result.content:
                           # Check if it resembles a tool call structure based on MCP SDK/spec
                           if hasattr(tool_call_block, 'type') and tool_call_block.type == "tool_use": # Hypothetical type
                                try:
                                     # Map mcp_types.CallToolResult -> gateway ToolUseRequest
                                     # Example structure based on Anthropic/OpenAI
                                     tool_name = getattr(tool_call_block, 'name', None)
                                     tool_input = getattr(tool_call_block, 'input', {})
                                     tool_call_id = getattr(tool_call_block, 'tool_use_id', str(uuid.uuid4())) # Get ID or generate

                                     if tool_name:
                                          # Find the original tool definition to include schema etc. (Optional but good practice)
                                          original_tool_def = next((t.function for t in original_request.tools or [] if t.function.name == tool_name), None)

                                          tool_use_reqs.append(ToolUseRequest(
                                               id=tool_call_id,
                                               type="function", # Assuming function calls
                                               function=ToolFunction(
                                                    name=tool_name,
                                                    # Include description/schema if found
                                                    description=original_tool_def.description if original_tool_def else None,
                                                    parameters=tool_input # MCP likely returns the arguments/input directly
                                               )
                                          ))
                                     else:
                                          logger.warning(f"MCP tool call block missing name: {tool_call_block}")

                                except AttributeError as e:
                                     logger.error(f"Error mapping MCP tool call block: Missing expected attribute {e}", exc_info=True)
                                except Exception as e:
                                     logger.error(f"Error mapping MCP tool call block: {e}", exc_info=True)
                           else:
                                logger.warning(f"Expected tool_use content block with TOOL_CALLS stop reason, but got type {getattr(tool_call_block, 'type', 'unknown')}")
                 elif mcp_result.content:
                      logger.warning(f"Expected list of tool calls with TOOL_CALLS stop reason, but got single content block: {type(mcp_result.content)}")

                 # Set generated_content to None if the primary reason was tool calls
                 generated_content = None

            elif hasattr(mcp_result, 'content') and mcp_result.content:
                 # Map regular Content (Text, Image, etc.)
                 content_block = mcp_result.content
                 # Assume CreateMessageResult returns a single content block for non-tool responses
                 gateway_items = self._map_mcp_content_block_to_gateway(content_block)
                 if len(gateway_items) == 1 and gateway_items[0].type == GatewayMCPContentType.TEXT:
                      generated_content = gateway_items[0].text_content
                 elif gateway_items:
                      generated_content = gateway_items # Store as list for multimodal

            # --- Usage Mapping ---
            # Check if usage info is available (e.g., in metadata or specific field)
            mcp_usage_data = getattr(mcp_result, 'usage', None) or getattr(mcp_result, 'metadata', {}).get('usage')
            if mcp_usage_data and isinstance(mcp_usage_data, dict):
                 # Map from MCP usage structure (e.g., {'input_tokens': N, 'output_tokens': M})
                 prompt_tokens = mcp_usage_data.get('input_tokens', 0)
                 completion_tokens = mcp_usage_data.get('output_tokens', 0)
                 if prompt_tokens > 0 or completion_tokens > 0:
                      gateway_usage = UsageStats(
                           prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens
                           # Total is calculated automatically by the model
                      )
            elif mcp_usage_data:
                 logger.warning(f"MCP usage data found but in unexpected format: {type(mcp_usage_data)}")

            # --- MCP Metadata ---
            provider_usage_mcp = None
            if gateway_usage: # Create MCPUsage if we mapped gateway stats
                 provider_usage_mcp = MCPUsage(input_tokens=gateway_usage.prompt_tokens, output_tokens=gateway_usage.completion_tokens)

            mcp_meta = MCPMetadata(
                # Assuming session or result might hold version/context info
                mcp_version=getattr(mcp_session, 'mcp_version', 'unknown') if mcp_session else 'unknown',
                model_version_reported=getattr(mcp_result, 'model', None) or getattr(mcp_result, 'metadata', {}).get('model_version'),
                context_id=getattr(mcp_result, 'context_id', None), # Hypothetical field for stateful context
                provider_usage=provider_usage_mcp
            )

        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            total_duration_ms=total_duration_ms,
            llm_latency_ms=llm_latency_ms,
            # Other timings (pre/post processing) would be calculated higher up
        )

        return LLMResponse(
            version=original_request.version,
            request_id=original_request.initial_context.request_id,
            generated_content=generated_content,
            finish_reason=finish_reason if finish_reason is not None else FinishReason.UNKNOWN, # Ensure finish_reason is set
            tool_use_requests=tool_use_reqs,
            usage=gateway_usage,
            compliance_result=None,  # Filled by interventions layer
            final_context=final_context_state,
            error_details=error_details,
            performance_metrics=perf_metrics,
            mcp_metadata=mcp_meta,
            # raw_provider_response=mcp_result # Optionally store raw response for debugging
        )


    def _map_mcp_stop_reason_to_gateway(self, mcp_reason: Optional[str]) -> FinishReason:
        """
        Maps MCP stop reason string to the gateway's FinishReason enum.

        Args:
            mcp_reason: The stop reason string provided by the MCP server.

        Returns:
            The corresponding gateway FinishReason enum member.
        """
        if not mcp_reason or not isinstance(mcp_reason, str):
            logger.debug("MCP stop reason is None or not a string, mapping to UNKNOWN.")
            return FinishReason.UNKNOWN

        reason_lower = mcp_reason.lower()

        # Mapping based on potential MCP spec values (case-insensitive)
        # This should be updated based on the official MCP specification
        stop_reason_mapping = {
            "endturn": FinishReason.STOP,
            "maxtokens": FinishReason.LENGTH,
            "stopsequence": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,        # Common convention
            "calltoolresult": FinishReason.TOOL_CALLS,  # Alternative convention?
            "contentfiltered": FinishReason.CONTENT_FILTERED,
            "error": FinishReason.ERROR,
        }

        mapped_reason = stop_reason_mapping.get(reason_lower)

        if mapped_reason:
            logger.debug(f"Mapped MCP stop reason '{mcp_reason}' to Gateway '{mapped_reason.value}'")
            return mapped_reason
        else:
            # Log the unmapped reason clearly
            logger.warning(
                f"Unmapped MCP stop reason string encountered: '{mcp_reason}'. "
                f"Mapping to Gateway '{FinishReason.UNKNOWN.value}'."
            )
            return FinishReason.UNKNOWN

    def _map_mcp_content_block_to_gateway(self, mcp_content_block: Any) -> List[ContentItem]:
        """
        Maps a single MCP content block back to a list of gateway ContentItems.
        Usually, this will be a list containing one item, but future-proofing for
        potential scenarios where one MCP block might map to multiple gateway items.

        Args:
            mcp_content_block: The content block object received from the MCP server.
                               Can be various types based on mcp.types (e.g., TextContent,
                               ImageContent, potentially CallToolResult, etc.).

        Returns:
            A list of gateway ContentItem objects derived from the MCP block.
        """
        gateway_items: List[ContentItem] = []

        if not mcp_content_block:
            logger.warning("Received empty or None MCP content block, cannot map.")
            return gateway_items

        try:
            # Get the type attribute safely
            content_type_str = getattr(mcp_content_block, 'type', None)
            if not content_type_str or not isinstance(content_type_str, str):
                 logger.warning(f"MCP content block is missing 'type' attribute or it's not a string: {type(mcp_content_block)}. Storing raw.")
                 content_type_str = "unknown_structure" # Assign a type for fallback

            content_type_lower = content_type_str.lower()

            # --- Handle Known/Expected Types ---

            if content_type_lower == "text":
                text_value = getattr(mcp_content_block, 'text', '')
                gateway_items.append(ContentItem(
                    type=GatewayMCPContentType.TEXT,
                    data={"text": text_value},
                    text_content=text_value
                ))
                logger.debug("Mapped MCP 'text' content block.")

            elif content_type_lower == "image":
                # Assumes image data is base64 encoded in 'data' attribute
                b64_data = getattr(mcp_content_block, 'data', None)
                mime_type = getattr(mcp_content_block, 'mimeType', "image/png") # Default MIME if missing
                if b64_data:
                     gateway_items.append(ContentItem(
                         type=GatewayMCPContentType.IMAGE,
                         # MCP schema might differ, adjust data structure as needed
                         data={"image": {"source": {"type": "base64", "data": b64_data}}},
                         mime_type=mime_type
                     ))
                     logger.debug(f"Mapped MCP 'image' content block (mime: {mime_type}).")
                else:
                     logger.warning("MCP 'image' content block missing 'data' attribute.")

            # --- Handle Tool-Related Types (Placeholder/Logging) ---
            # These might be handled elsewhere if they appear outside the main 'content'
            # when stop_reason is TOOL_CALLS, but mapping here handles cases where they
            # might appear as regular content blocks.

            elif content_type_lower == "tool_use": # Hypothetical type name
                 logger.info("Encountered MCP 'tool_use' content block. Storing raw - expected handling via TOOL_CALLS stop reason.")
                 # Store raw for inspection, specific mapping depends on structure
                 raw_data = self._safe_dump_mcp_object(mcp_content_block)
                 gateway_items.append(ContentItem(
                     type=GatewayMCPContentType.TOOL_USE, # Map to gateway enum
                     data={"raw_mcp_tool_use": raw_data},
                     mime_type="application/json"
                 ))

            elif content_type_lower == "tool_result": # Hypothetical type name
                 logger.info("Encountered MCP 'tool_result' content block. Storing raw.")
                 raw_data = self._safe_dump_mcp_object(mcp_content_block)
                 gateway_items.append(ContentItem(
                     type=GatewayMCPContentType.TOOL_RESULT, # Map to gateway enum
                     data={"raw_mcp_tool_result": raw_data},
                     mime_type="application/json"
                 ))


            # --- Fallback for Unknown Types ---
            else:
                logger.warning(f"Unsupported or unexpected MCP content block type encountered: '{content_type_str}'. Storing raw.")
                raw_data = self._safe_dump_mcp_object(mcp_content_block)

                # Try to map to a known gateway enum value if the string matches, otherwise use FILE
                try:
                     gateway_type_enum = GatewayMCPContentType(content_type_lower)
                except ValueError:
                     gateway_type_enum = GatewayMCPContentType.FILE # Default fallback type

                gateway_items.append(ContentItem(
                    type=gateway_type_enum,
                    data={"raw_mcp_block": raw_data, "original_mcp_type": content_type_str},
                    mime_type="application/json" # Assuming raw data is JSON-like
                ))

        except Exception as e:
            # Catch unexpected errors during mapping
            logger.error(f"Failed to map MCP content block due to unexpected error: {e}", exc_info=True)
            # Optionally, store the raw block as an error item
            try:
                 raw_data_on_error = self._safe_dump_mcp_object(mcp_content_block)
            except Exception:
                 raw_data_on_error = f"Failed to serialize MCP block: {type(mcp_content_block)}"

            gateway_items.append(ContentItem(
                 type=GatewayMCPContentType.FILE, # Or a specific ERROR type if defined
                 data={"mapping_error": str(e), "raw_mcp_block": raw_data_on_error},
                 mime_type="application/json"
            ))

        return gateway_items

    def _safe_dump_mcp_object(self, mcp_obj: Any) -> Dict[str, Any]:
         """Safely attempts to serialize an MCP object to a dict."""
         if hasattr(mcp_obj, 'model_dump'):
              try:
                   return mcp_obj.model_dump()
              except Exception as dump_err:
                   logger.warning(f"model_dump failed for {type(mcp_obj)}: {dump_err}")
         # Fallback: try converting to dict directly or getting __dict__
         try:
              if hasattr(mcp_obj, '__dict__'):
                   # Be cautious with __dict__, might include private attrs
                   return {k: v for k, v in vars(mcp_obj).items() if not k.startswith('_')}
              else:
                   return {"raw_string_representation": str(mcp_obj)}
         except Exception as str_err:
              logger.error(f"Failed to serialize MCP object {type(mcp_obj)}: {str_err}")
              return {"serialization_error": str(str_err)}

    
# --- End of MCPProvider class ---        