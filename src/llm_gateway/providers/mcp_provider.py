# llm_gateway/providers/mcp_provider.py

import asyncio
import logging
import os
import json
import uuid
import copy
import shutil
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

# --- MCP SDK Imports ---
# Keep the try/except block for graceful handling if SDK is missing initially
try:
    from mcp import ClientSession, StdioServerParameters
    import mcp.types as mcp_types
    from mcp.client.stdio import stdio_client
    from mcp.shared.exceptions import McpError, ErrorCode # Import ErrorCode if available
except ImportError:
    logging.warning("MCP SDK not found, using placeholder types. MCPProvider will not function.")
    # Define dummy classes to allow module import
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
        class Error: pass # Dummy Error type
        class ToolResultContent: pass # Dummy ToolResult type
    class McpError(Exception):
         def __init__(self, error=None, *args):
              self.error = error or mcp_types.Error() # Provide a dummy error
              super().__init__(*args)
    class ErrorCode: # Dummy ErrorCode enum/class
        # Add known/expected error codes if possible, otherwise leave empty
        TEMPORARY_SERVER_ERROR = "TEMPORARY_SERVER_ERROR"
        RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
        # ... other potential codes

# --- Gateway Imports ---
# Use absolute paths assuming standard project structure
from llm_gateway.core.models import (
    ContentItem, ErrorDetails, ErrorLevel, FinishReason, GatewayConfig,
    InterventionContext, LLMConfig, LLMRequest, LLMResponse,
    MCPContentType as GatewayMCPContentType, MCPMetadata, MCPRole as GatewayMCPRole,
    PerformanceMetrics, ProviderConfig, StreamChunk, ToolDefinition, ToolFunction,
    ToolResult as GatewayToolResult, # Renamed to avoid conflict
    ToolUseRequest, UsageStats, MCPUsage
)
from llm_gateway.providers.base import BaseProvider # Use absolute path

logger = logging.getLogger(__name__)

# Define known retryable MCP error codes (adjust based on actual MCP spec)
MCP_RETRYABLE_ERROR_CODES = {
    getattr(ErrorCode, "TEMPORARY_SERVER_ERROR", "TEMPORARY_SERVER_ERROR"),
    getattr(ErrorCode, "SERVICE_UNAVAILABLE", "SERVICE_UNAVAILABLE"),
    getattr(ErrorCode, "RATE_LIMIT_EXCEEDED", "RATE_LIMIT_EXCEEDED"),
    getattr(ErrorCode, "TIMEOUT", "TIMEOUT"),
    getattr(ErrorCode, "RESOURCE_EXHAUSTED", "RESOURCE_EXHAUSTED"),
    # Add numeric codes if MCP uses them alongside string codes
    # 429, 500, 502, 503, 504
}

class MCPProvider(BaseProvider):
    """
    LLM Gateway provider implementation using the MCP standard via the official SDK.
    Connects to an MCP server process using stdio.
    """

    def __init__(self, provider_config: ProviderConfig, gateway_config: GatewayConfig):
        """Initialize the MCP Provider."""
        # Call super init *first*
        super().__init__(provider_config, gateway_config) # Pass gateway_config here
        # self.gateway_config is now set by super()
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._is_healthy = False # Track health status

        # Extract connection params
        conn_params = provider_config.connection_params
        self._server_command = conn_params.get("command", "npx")
        self._server_args = conn_params.get("args", [])
        self._server_env = conn_params.get("env", {})

        # Extract operational params, falling back to gateway defaults
        self._enable_streaming = conn_params.get("enable_streaming", False) # Default to disabled until implemented
        self._max_retry_attempts = conn_params.get("max_retries", self.gateway_config.max_retries)
        self._retry_delay_seconds = conn_params.get("retry_delay_seconds", self.gateway_config.retry_delay_seconds)
        self._timeout_seconds = conn_params.get("timeout_seconds", self.gateway_config.default_timeout_seconds)

        logger.info(f"Initialized MCPProvider '{self.provider_id}' (Streaming: {self._enable_streaming}, Retries: {self._max_retry_attempts})")

    # _initialize_client is removed from BaseProvider, init happens here or in initialize_async

    async def initialize_async(self):
        """Perform async initialization, like establishing the initial session."""
        # Optionally pre-warm the connection
        if self.provider_config.connection_params.get("connect_on_init", False):
             logger.info(f"Attempting initial connection for MCP provider '{self.provider_id}'...")
             try:
                  async with self._ensure_session() as session:
                       # Perform a lightweight check, e.g., get capabilities if available
                       # capabilities = await session.get_capabilities() # Hypothetical
                       logger.info(f"MCP provider '{self.provider_id}' initial connection successful.")
                       self._is_healthy = True
             except Exception as e:
                  logger.error(f"Initial connection failed for MCP provider '{self.provider_id}': {e}")
                  self._is_healthy = False
                  # Decide whether to raise the error or just log it
                  # raise ConnectionError(f"Initial MCP session failed") from e

    @asynccontextmanager
    async def _ensure_session(self) -> AsyncGenerator[ClientSession, None]:
        """Ensures an active MCP session is available, establishing one if needed."""
        # Check if MCP SDK was imported successfully
        if 'mcp' not in globals():
             raise RuntimeError("MCP SDK is not installed or failed to import. MCPProvider cannot function.")

        async with self._session_lock:
            if self._session is None:
                logger.info(f"No active MCP session for {self.provider_id}. Initializing...")
                try:
                    command = shutil.which(self._server_command) # Simplified resolution
                    if command is None:
                        raise ValueError(f"MCP server command not found: {self._server_command}")

                    server_params = StdioServerParameters(
                        command=command,
                        args=self._server_args,
                        env={**os.environ, **self._server_env, **self._get_auth_env()},
                        # Pass timeout if SDK supports it
                        # timeout_seconds=self._timeout_seconds
                    )

                    # Ensure previous stack is closed before creating new contexts
                    await self._exit_stack.aclose()
                    self._exit_stack = AsyncExitStack() # Create a new stack

                    stdio_transport = await self._exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    read, write = stdio_transport
                    session = await self._exit_stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    # Initialize with timeout if SDK supports it
                    await asyncio.wait_for(session.initialize(), timeout=self._timeout_seconds)
                    self._session = session
                    self._is_healthy = True # Mark as healthy after successful init
                    logger.info(f"MCP session initialized successfully for {self.provider_id}.")
                except (Exception, asyncio.TimeoutError) as e:
                     logger.error(f"Failed to initialize MCP session for {self.provider_id}: {e}", exc_info=True)
                     await self._exit_stack.aclose() # Ensure cleanup on error
                     self._session = None
                     self._is_healthy = False
                     raise ConnectionError(f"MCP session initialization failed: {e}") from e

            # Check if the existing session is still valid (if SDK provides a way)
            # if hasattr(self._session, 'is_active') and not self._session.is_active():
            #     logger.warning(f"MCP session for {self.provider_id} is inactive. Re-initializing.")
            #     await self.cleanup() # Close existing stack/session
            #     # Let the logic re-run to create a new session in the next call
            #     # This requires careful state management or restructure the context manager
            #     # For now, assume session remains valid until cleanup or error

            if self._session is None:
                 # Should not happen if logic above is correct, but defensive
                 raise ConnectionError("MCP session is unexpectedly None after initialization attempt.")

            yield self._session

    async def cleanup(self):
         """Cleans up the MCP session and transport."""
         logger.info(f"Cleaning up MCP session for {self.provider_id}...")
         async with self._session_lock:
              if self._session is not None:
                   try:
                        # Add specific session closing logic if the SDK requires it
                        # await self._session.close() # Hypothetical
                        pass
                   except Exception as e:
                        logger.warning(f"Error during MCP session close for {self.provider_id}: {e}")
                   finally:
                        await self._exit_stack.aclose()
                        self._session = None
                        self._is_healthy = False
         logger.info(f"MCP session cleanup complete for {self.provider_id}.")

    def _get_auth_env(self) -> Dict[str, str]:
         """Get API keys/auth tokens for the MCP server environment."""
         auth_env = {}
         # Look for specific key in connection_params first
         auth_key_env_var = self.provider_config.connection_params.get("api_key_env_var")
         if auth_key_env_var:
             api_key = os.environ.get(auth_key_env_var)
             if api_key:
                  # Use a standard name expected by MCP servers, or make it configurable
                  auth_env["MCP_API_KEY"] = api_key
                  logger.debug(f"Using specific API key env var '{auth_key_env_var}' for MCP auth.")
                  return auth_env

         # Fallback to generic key
         generic_key = os.environ.get("MCP_SERVER_API_KEY")
         if generic_key:
             auth_env["MCP_API_KEY"] = generic_key
             logger.debug("Using generic MCP_SERVER_API_KEY for MCP auth.")
             return auth_env

         logger.warning(f"No API key environment variable configured or found for MCP provider '{self.provider_id}'.")
         return {}

    async def health_check(self) -> Dict[str, Any]:
        """Check MCP server health by attempting to initialize or ping session."""
        check_start_time = datetime.utcnow()
        if self._session is not None and self._is_healthy:
             # Basic check: if we have an active session, assume healthy for now
             # TODO: Add a lightweight ping/status call if MCP SDK/spec supports it
             # e.g., try: await self._session.ping(); status="available" except: status="unhealthy"
             status = "available"
             message = "Session appears active."
             self._is_healthy = True # Assume still healthy
        else:
            # Try to establish a session as a health check
            try:
                async with self._ensure_session() as session:
                     # If ensure_session succeeds, it's healthy
                     status = "available"
                     message = "Successfully initialized new session."
                     self._is_healthy = True
            except Exception as e:
                status = "unhealthy"
                message = f"Failed to initialize session: {str(e)}"
                self._is_healthy = False # Explicitly mark as unhealthy

        return {
            "provider_id": self.provider_id,
            "status": status,
            "provider_type": self.provider_config.provider_type,
            "checked_at": check_start_time.isoformat(),
            "message": message
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using MCP non-streaming `create_message`."""
        start_time = datetime.utcnow()
        llm_latency_ms = None
        mcp_result: Optional[mcp_types.CreateMessageResult] = None
        error_details: Optional[ErrorDetails] = None
        session_instance: Optional[ClientSession] = None

        for attempt in range(self._max_retry_attempts + 1):
            is_last_attempt = attempt == self._max_retry_attempts
            try:
                async with self._ensure_session() as session:
                    session_instance = session
                    mcp_messages = self._map_to_mcp_sampling_messages(request)
                    # Pass tools definition if present in request
                    mcp_params = self._prepare_mcp_sampling_params(request.config, request.tools)

                    logger.debug(f"Attempt {attempt+1}: Sending createMessage to MCP model {request.config.model_identifier}...")
                    llm_call_start = datetime.utcnow()

                    # Use timeout for the API call itself
                    mcp_result = await asyncio.wait_for(
                        session.create_message(
                            messages=mcp_messages,
                            max_tokens=mcp_params.pop("max_tokens", 1024), # Ensure max_tokens is removed if present
                             # Add tools if mapped
                            tools=mcp_params.pop("tools", None),
                            **mcp_params
                        ),
                        timeout=self._timeout_seconds
                    )

                    llm_latency_ms = (datetime.utcnow() - llm_call_start).total_seconds() * 1000
                    logger.debug(f"Attempt {attempt+1}: Received createMessage result successfully.")
                    error_details = None # Clear previous error if retry succeeded
                    break # Success

            except McpError as e:
                logger.warning(f"Attempt {attempt+1}: MCPError during generate: {getattr(e, 'error', e)}", exc_info=not self._is_retryable_error(e))
                error_details = self._map_mcp_error(e)
                if not error_details.retryable or is_last_attempt:
                     self._is_healthy = not self._is_fatal_error(e) # Mark unhealthy on fatal errors
                     break
                await asyncio.sleep(self._retry_delay_seconds * (2 ** attempt)) # Exponential backoff
            except asyncio.TimeoutError as e:
                 logger.warning(f"Attempt {attempt+1}: Timeout error during generate after {self._timeout_seconds}s.")
                 error_details = self._map_error(e, retryable=True, stage="provider_call") # Map generic timeout
                 if is_last_attempt:
                      self._is_healthy = False # Mark unhealthy after repeated timeouts
                      break
                 await asyncio.sleep(self._retry_delay_seconds * (2 ** attempt))
            except Exception as e:
                logger.error(f"Attempt {attempt+1}: Unexpected error during generate: {e}", exc_info=True)
                error_details = self._map_error(e, stage="provider_call") # Map generic error
                self._is_healthy = False # Mark unhealthy on unexpected errors
                break # Non-MCP errors are typically not retryable here

        # Map MCP result back to gateway LLMResponse
        total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        # Use original context as base, interventions might modify it later
        final_context = copy.deepcopy(request.initial_context)

        response = self._map_from_mcp_create_message_result(
            mcp_result=mcp_result,
            original_request=request,
            final_context_state=final_context, # Pass context to be included in response
            error_details=error_details,
            llm_latency_ms=llm_latency_ms,
            total_duration_ms=total_duration_ms,
            mcp_session=session_instance # Pass session for metadata if needed
        )
        # Update context with info from response (e.g., provider metadata) if needed
        # response.final_context will hold the context state *before* post-interventions

        return response

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Generate response using MCP streaming (or fallback)."""
        request_id = request.initial_context.request_id

        if not self._enable_streaming:
            logger.warning(f"Streaming requested for {request_id} but disabled for MCP provider '{self.provider_id}'. Using non-streaming fallback.")
            response = await self.generate(request) # Call non-streaming method
            chunk_id = 0
            if response.error_details:
                yield StreamChunk(
                    chunk_id=chunk_id, request_id=request_id, finish_reason=FinishReason.ERROR,
                    provider_specific_data={"error": response.error_details.model_dump()}
                )
            else:
                # Yield a single chunk representing the full response
                delta_text = None
                delta_content_items = None
                if isinstance(response.generated_content, str): delta_text = response.generated_content
                elif isinstance(response.generated_content, list): delta_content_items = response.generated_content

                yield StreamChunk(
                    chunk_id=chunk_id, request_id=request_id,
                    delta_text=delta_text, delta_content_items=delta_content_items,
                    delta_tool_calls=response.tool_use_requests,
                    finish_reason=response.finish_reason,
                    usage_update=response.usage,
                    provider_specific_data=response.mcp_metadata.model_dump() if response.mcp_metadata else None
                )
            return # End of fallback stream

        # --- Native MCP Streaming (Placeholder) ---
        logger.warning(f"Native MCP streaming requested for {request_id} but is not implemented in this provider.")
        # If MCP SDK adds streaming support (e.g., session.stream_message), implement here:
        # try:
        #     async with self._ensure_session() as session:
        #         mcp_messages = self._map_to_mcp_sampling_messages(request)
        #         mcp_params = self._prepare_mcp_sampling_params(request.config, request.tools)
        #         mcp_stream = session.stream_message(messages=mcp_messages, **mcp_params) # Hypothetical call
        #         chunk_index = 0
        #         async for mcp_chunk in mcp_stream:
        #             # Map mcp_chunk to gateway StreamChunk
        #             gateway_chunk = self._map_mcp_stream_chunk(mcp_chunk, request_id, chunk_index)
        #             if gateway_chunk:
        #                 yield gateway_chunk
        #                 chunk_index += 1
        # except McpError as e:
        #     yield self._create_mcp_error_chunk(request_id, chunk_index, e)
        # except Exception as e:
        #     yield self._create_generic_error_chunk(request_id, chunk_index, e)
        # --- End Placeholder ---

        # Yield error because it's not implemented
        yield StreamChunk(
             chunk_id=0, request_id=request_id, finish_reason=FinishReason.ERROR,
             provider_specific_data={
                 "error": ErrorDetails(
                     code="NOT_IMPLEMENTED", stage="provider_call",
                     message="Native MCP streaming is not yet implemented.",
                     level=ErrorLevel.ERROR
                 ).model_dump()
             }
        )
        # Required for async generator typing
        if False: # pragma: no cover
            yield # type: ignore


    def _is_retryable_error(self, error: Union[McpError, Exception]) -> bool:
        """Determine if an error is likely retryable."""
        if isinstance(error, McpError) and hasattr(error, 'error') and hasattr(error.error, 'code'):
            return error.error.code in MCP_RETRYABLE_ERROR_CODES
        elif isinstance(error, asyncio.TimeoutError):
             return True # Network timeouts often retryable
        # Add other potentially retryable exceptions wrapped by McpError if needed
        # elif isinstance(error.__cause__, SomeRetryableNetworkError):
        #     return True
        return False

    def _is_fatal_error(self, error: Union[McpError, Exception]) -> bool:
        """Determine if an error indicates a persistent problem (e.g., bad auth, invalid command)."""
        if isinstance(error, McpError) and hasattr(error, 'error') and hasattr(error.error, 'code'):
             # Define codes that indicate a non-recoverable state for this session
             fatal_codes = [
                  getattr(ErrorCode, "AUTHENTICATION_FAILED", "AUTHENTICATION_FAILED"),
                  getattr(ErrorCode, "INVALID_ARGUMENT", "INVALID_ARGUMENT"), # Usually indicates bad request mapping
                  getattr(ErrorCode, "PERMISSION_DENIED", "PERMISSION_DENIED"),
             ]
             return error.error.code in fatal_codes
        elif isinstance(error, (ValueError, TypeError)): # Often indicates bad config or mapping
            return True
        elif isinstance(error, ConnectionRefusedError): # Cannot connect at all
             return True
        return False


    # --- Mapping Functions ---

    def _map_to_mcp_sampling_messages(self, request: LLMRequest) -> List[mcp_types.SamplingMessage]:
        """Convert gateway request history and prompt to MCP SamplingMessage list."""
        mcp_messages: List[mcp_types.SamplingMessage] = []

        # Map history turns
        for turn in request.initial_context.conversation_history:
            mcp_role = self._map_gateway_role_to_mcp(turn.role)
            if mcp_role is None and turn.role != GatewayMCPRole.TOOL.value: # Skip system (handled elsewhere), allow tool for now
                 logger.debug(f"Skipping history turn {turn.turn_id} with unmappable role: {turn.role}")
                 continue

            try:
                content_blocks: List[Union[mcp_types.TextContent, mcp_types.ImageContent, mcp_types.ToolResultContent]] = [] # Allow tool result content type

                # Handle TOOL role specifically
                if turn.role == GatewayMCPRole.TOOL.value:
                     if isinstance(turn.content, GatewayToolResult):
                          # Map GatewayToolResult -> mcp_types.ToolResultContent (Hypothetical)
                          # This structure depends heavily on MCP spec
                          tool_content = mcp_types.ToolResultContent( # Assuming this type exists
                               type="tool_result",
                               toolUseId=turn.content.tool_call_id, # Link back to the call ID
                               # Content could be simple string or complex dict based on spec
                               content=str(turn.content.output) if not isinstance(turn.content.output, (dict, list)) else json.dumps(turn.content.output),
                               isError=turn.content.is_error,
                          )
                          content_blocks.append(tool_content)
                          # MCP might require tool results under USER role, adjust 'role' if needed
                          # For now, assume MCP supports a specific message structure for tool results
                          # Needs clarification based on MCP spec. If tool results MUST be in a USER message,
                          # this logic needs significant change. Let's assume a TOOL_RESULT content type exists for now.
                          mcp_role = "user" # Or maybe keep as 'tool' if MCP supports it? Assume user for now based on OpenAI/Anthropic
                          logger.debug(f"Mapping TOOL turn {turn.turn_id} as USER message with ToolResultContent.")

                     else:
                          logger.warning(f"History turn {turn.turn_id} has role TOOL but content is not GatewayToolResult ({type(turn.content)}). Skipping.")
                          continue

                # Handle USER/ASSISTANT/SYSTEM roles
                elif isinstance(turn.content, str):
                    content_blocks.append(mcp_types.TextContent(type="text", text=turn.content))
                elif isinstance(turn.content, list): # Multimodal
                    for item in turn.content:
                         if item.type == GatewayMCPContentType.TEXT and item.text_content:
                              content_blocks.append(mcp_types.TextContent(type="text", text=item.text_content))
                         elif item.type == GatewayMCPContentType.IMAGE:
                              try: content_blocks.append(self._map_gateway_image_to_mcp(item))
                              except ValueError as e: logger.warning(f"Skipping image in history turn {turn.turn_id}: {e}")
                         # Add other supported types (Audio, Video...)
                         else: logger.warning(f"Skipping unsupported content type '{item.type}' in history turn {turn.turn_id}")
                else:
                    logger.warning(f"Cannot map history content type {type(turn.content)} for turn {turn.turn_id}")
                    continue # Skip turn if content unmappable

                # Add message if content was successfully mapped
                if content_blocks and mcp_role: # Ensure role is valid too
                     # MCP might expect single content block per message, or allow list
                     # Assuming list is allowed based on modern APIs:
                     mcp_messages.append(mcp_types.SamplingMessage(role=mcp_role, content=content_blocks))
                     # If only single block allowed:
                     # for block in content_blocks:
                     #    mcp_messages.append(mcp_types.SamplingMessage(role=role, content=block))

            except Exception as e: # Catch errors during content mapping for a single turn
                logger.error(f"Skipping history turn {turn.turn_id} due to mapping error: {e}", exc_info=True)
                continue

        # Map current prompt content (always USER role for the *final* prompt message)
        current_content_blocks: List[Union[mcp_types.TextContent, mcp_types.ImageContent]] = []
        current_content_items: List[ContentItem] = [] # Ensure it's a list
        if isinstance(request.prompt_content, str):
            current_content_items = [ContentItem.from_text(request.prompt_content)]
        elif isinstance(request.prompt_content, list):
            current_content_items = request.prompt_content
        # else: prompt_content is invalid type - caught by Pydantic likely

        for item in current_content_items:
            try:
                if item.type == GatewayMCPContentType.TEXT and item.text_content:
                     current_content_blocks.append(mcp_types.TextContent(type="text", text=item.text_content))
                elif item.type == GatewayMCPContentType.IMAGE:
                     try: current_content_blocks.append(self._map_gateway_image_to_mcp(item))
                     except ValueError as e: logger.warning(f"Skipping image in current prompt: {e}")
                # Add other types here
                else: logger.warning(f"Skipping unsupported prompt content type '{item.type}'")
            except Exception as e:
                 logger.error(f"Skipping prompt content item due to mapping error: {e}", exc_info=True)

        # Add the final user prompt message
        if current_content_blocks:
             # Check if last message was already user (e.g. empty history, multi-part prompt)
             if mcp_messages and mcp_messages[-1].role == "user":
                  # Append content to last user message if MCP supports list content
                  # If MCP requires single content block, this merge isn't possible easily
                  logger.debug("Appending current prompt to last user message (assuming list content support in MCP).")
                  # This assumes mcp_messages[-1].content is mutable and a list
                  if isinstance(mcp_messages[-1].content, list):
                       mcp_messages[-1].content.extend(current_content_blocks)
                  else: # If single content block, create new message
                       mcp_messages.append(mcp_types.SamplingMessage(role="user", content=current_content_blocks))
             else:
                  mcp_messages.append(mcp_types.SamplingMessage(role="user", content=current_content_blocks))
        elif not mcp_messages: # No history and no valid current prompt
            logger.error("Cannot send request: No valid message content after mapping.")
            raise ValueError("Cannot send request to MCP with no valid user message content.")

        # Final validation: Ensure user/assistant alternation if required by MCP
        # This is complex to enforce perfectly without knowing MCP spec, add logging for now
        last_role = None
        for i, msg in enumerate(mcp_messages):
             if msg.role == last_role and msg.role in ["user", "assistant"]:
                  logger.warning(f"Consecutive '{msg.role}' messages detected at index {i}. MCP may reject.")
             if msg.role in ["user", "assistant"]:
                  last_role = msg.role

        return mcp_messages


    def _map_gateway_role_to_mcp(self, gateway_role: str) -> Optional[str]:
        """Map gateway role string to MCP role string."""
        role_lower = gateway_role.lower()
        # This mapping depends on the exact strings expected/supported by the mcp.types.Role
        # Assuming direct mapping for user/assistant, handle system separately, ignore tool for input context
        if role_lower == GatewayMCPRole.USER.value: return "user"
        if role_lower == GatewayMCPRole.ASSISTANT.value: return "assistant"
        # System prompt handled via _prepare_mcp_sampling_params if needed
        # Tool results handled separately
        return None

    def _map_gateway_image_to_mcp(self, item: ContentItem) -> mcp_types.ImageContent:
        """Maps a gateway ContentItem (Image) to mcp.types.ImageContent."""
        # Check if mcp_types.ImageContent actually exists
        if not hasattr(mcp_types, 'ImageContent'):
             raise NotImplementedError("MCP SDK type 'ImageContent' not found or imported.")

        b64_data: Optional[str] = None
        mime_type = item.mime_type or "image/png" # Default

        source = item.data.get("image", {}).get("source", {})
        source_type = source.get("type")

        if source_type == "base64":
             b64_data = source.get("data")
        elif source_type == "url":
              # Fetching URLs is generally bad practice here. Require base64.
              raise ValueError(f"Image source type 'url' not supported by MCPProvider mapping. Provide base64.")
        # Add file path reading if needed and secure
        # elif source_type == "file": ... read file ... base64 encode ...

        if not b64_data:
            raise ValueError(f"Could not extract/convert image data to base64 for ContentItem: {item.type}")
        if not mime_type:
             raise ValueError("MIME type is required for image content.")

        # Construct the MCP type, assuming 'type', 'data', 'mimeType' fields
        return mcp_types.ImageContent(type="image", data=b64_data, mimeType=mime_type)


    def _prepare_mcp_sampling_params(self, config: LLMConfig, tools: Optional[List[ToolDefinition]] = None) -> Dict[str, Any]:
        """Maps gateway LLMConfig and Tools to MCP create_message parameters."""
        params: Dict[str, Any] = {}

        # --- Required/Strongly Recommended ---
        # Max Tokens (ensure it's set, default if needed)
        params["max_tokens"] = config.max_tokens or self.provider_config.connection_params.get("default_max_tokens", 1024)
        if config.max_tokens is None:
             logger.warning(f"max_tokens not specified, using default: {params['max_tokens']}")

        # --- Optional Standard Parameters ---
        if config.temperature is not None: params["temperature"] = config.temperature
        if config.stop_sequences: params["stop_sequences"] = config.stop_sequences
        if config.top_p is not None: params["top_p"] = config.top_p
        # Add top_k if MCP supports it
        # if config.additional_config.get("top_k") is not None: params["top_k"] = config.additional_config["top_k"]

        # --- System Prompt ---
        # MCP spec might use top-level 'system' param or first message role.
        # Assume top-level for now if present in config.
        if config.system_prompt:
            params["system"] = config.system_prompt # Requires MCP SDK/server support

        # --- Tool Mapping ---
        if tools:
            # Verify mcp_types.Tool exists before proceeding
            if hasattr(mcp_types, 'Tool'):
                mcp_tools: List[mcp_types.Tool] = []
                try:
                    for tool_def in tools:
                        # Map ToolDefinition -> mcp_types.Tool
                        mcp_tools.append(mcp_types.Tool(
                            name=tool_def.function.name,
                            description=tool_def.function.description,
                            input_schema=tool_def.function.parameters or {"type": "object", "properties": {}}
                        ))
                    if mcp_tools:
                        params["tools"] = mcp_tools
                        # Add tool_choice if specified in request extensions/config
                        tool_choice = config.additional_config.get("tool_choice")
                        if tool_choice: params["tool_choice"] = tool_choice # e.g., "auto", {"type": "tool", "name": "..."}
                except AttributeError:
                     logger.warning("MCP SDK mcp_types.Tool structure mismatch. Skipping tool mapping.")
                except Exception as e:
                     logger.error(f"Error mapping tools to MCP format: {e}", exc_info=True)
            else:
                logger.warning("Attempted to map tools, but mcp_types.Tool is not available. Skipping.")

        # --- Metadata ---
        params["metadata"] = params.get("metadata", {})
        params["metadata"]["gateway_request_id"] = config.initial_context.request_id # Pass request ID
        if config.model_identifier:
            params["metadata"]["model_requested"] = config.model_identifier
        # Add other relevant context? (user_id, session_id) - be careful with PII

        # Log unmapped parameters
        unmapped = {
            k: v for k, v in config.model_dump(exclude={'model_identifier', 'temperature', 'max_tokens', 'top_p', 'stop_sequences', 'system_prompt'}).items()
            if v is not None and k not in ['presence_penalty', 'frequency_penalty'] # Explicitly ignore some common ones not typically in MCP
        }
        if unmapped:
            logger.debug(f"LLMConfig parameters not directly mapped to MCP params: {list(unmapped.keys())}")

        return params


    def _map_from_mcp_create_message_result(
        self,
        mcp_result: Optional[mcp_types.CreateMessageResult],
        original_request: LLMRequest,
        final_context_state: InterventionContext, # Passed in context
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

        # Set default finish reason to ERROR if error occurred
        if error_details:
             finish_reason = FinishReason.ERROR

        elif mcp_result:
            # Map Stop Reason
            finish_reason = self._map_mcp_stop_reason_to_gateway(getattr(mcp_result, 'stopReason', None))

            # Map Content and Tool Calls based on stop reason
            mcp_content = getattr(mcp_result, 'content', None) # Get content block/list

            if finish_reason == FinishReason.TOOL_CALLS:
                 # --- Tool Call Mapping ---
                 tool_use_reqs = []
                 # MCP spec needed: Assume content is a list of tool_use blocks
                 if isinstance(mcp_content, list):
                      for tool_call_block in mcp_content:
                           if hasattr(tool_call_block, 'type') and tool_call_block.type == "tool_use": # Hypothetical
                                try:
                                     tool_name = getattr(tool_call_block, 'name', None)
                                     tool_input = getattr(tool_call_block, 'input', {})
                                     tool_call_id = getattr(tool_call_block, 'tool_use_id', str(uuid.uuid4()))
                                     if tool_name:
                                          original_tool_def = next((t.function for t in original_request.tools or [] if t.function.name == tool_name), None)
                                          tool_use_reqs.append(ToolUseRequest(
                                               id=tool_call_id, type="function",
                                               function=ToolFunction(
                                                    name=tool_name,
                                                    description=original_tool_def.description if original_tool_def else None,
                                                    parameters=tool_input # Assume input is the parameters dict
                                               )))
                                except Exception as e: logger.error(f"Error mapping MCP tool_use block: {e}")
                           else: logger.warning(f"Non-tool_use block found when expecting tool calls: type={getattr(tool_call_block, 'type', 'unknown')}")
                 else: logger.warning(f"Expected list content for TOOL_CALLS, got {type(mcp_content)}")
                 generated_content = None # No text content when tool calls are made

            elif mcp_content:
                 # --- Regular Content Mapping ---
                 gateway_items = self._map_mcp_content_block_to_gateway(mcp_content)
                 if len(gateway_items) == 1 and gateway_items[0].type == GatewayMCPContentType.TEXT:
                      generated_content = gateway_items[0].text_content
                 elif gateway_items:
                      generated_content = gateway_items # Multimodal or multiple parts

            # --- Usage Mapping ---
            mcp_usage_data = getattr(mcp_result, 'usage', None) or getattr(mcp_result, 'metadata', {}).get('usage')
            if mcp_usage_data and isinstance(mcp_usage_data, dict):
                 prompt_tokens = mcp_usage_data.get('input_tokens', 0)
                 completion_tokens = mcp_usage_data.get('output_tokens', 0)
                 if prompt_tokens > 0 or completion_tokens > 0:
                      gateway_usage = UsageStats(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            elif mcp_usage_data: logger.warning(f"MCP usage data in unexpected format: {type(mcp_usage_data)}")

            # --- MCP Metadata ---
            provider_usage_mcp = MCPUsage(**mcp_usage_data) if gateway_usage else None
            mcp_meta = MCPMetadata(
                mcp_version=getattr(mcp_session, 'mcp_version', 'unknown') if mcp_session else 'unknown',
                model_version_reported=getattr(mcp_result, 'model', None) or getattr(mcp_result, 'metadata', {}).get('model_version'),
                context_id=getattr(mcp_result, 'context_id', None),
                provider_usage=provider_usage_mcp
            )

        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            total_duration_ms=total_duration_ms,
            llm_latency_ms=llm_latency_ms,
        )

        # Construct final response
        return LLMResponse(
            version=original_request.version,
            request_id=original_request.initial_context.request_id,
            generated_content=generated_content,
            # Ensure finish_reason is set, defaulting to UNKNOWN if needed
            finish_reason=finish_reason if finish_reason is not None else FinishReason.UNKNOWN,
            tool_use_requests=tool_use_reqs,
            usage=gateway_usage,
            compliance_result=None,  # Set by post-interventions
            final_context=final_context_state, # Return the context passed in
            error_details=error_details,
            performance_metrics=perf_metrics,
            mcp_metadata=mcp_meta,
            # Store raw MCP result if needed for debugging, ensure it's serializable
            # raw_provider_response=self._safe_dump_mcp_object(mcp_result) if mcp_result else None
        )


    def _map_mcp_stop_reason_to_gateway(self, mcp_reason: Optional[str]) -> FinishReason:
        """Maps MCP stop reason string to the gateway's FinishReason enum."""
        # ... (Keep previous robust implementation) ...
        if not mcp_reason or not isinstance(mcp_reason, str):
            return FinishReason.UNKNOWN
        reason_lower = mcp_reason.lower()
        mapping = { # Based on common conventions, adjust to MCP spec
            "end_turn": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "stop_sequence": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALLS,
            "content_filtered": FinishReason.CONTENT_FILTERED,
            "error": FinishReason.ERROR,
        }
        mapped = mapping.get(reason_lower)
        if mapped: return mapped
        # Handle potential camelCase versions
        if reason_lower == "endturn": return FinishReason.STOP
        if reason_lower == "maxtokens": return FinishReason.LENGTH
        if reason_lower == "stopsequence": return FinishReason.STOP
        if reason_lower == "tooluse": return FinishReason.TOOL_CALLS
        if reason_lower == "contentfiltered": return FinishReason.CONTENT_FILTERED

        logger.warning(f"Unmapped MCP stop reason '{mcp_reason}'. Mapping to UNKNOWN.")
        return FinishReason.UNKNOWN


    def _map_mcp_content_block_to_gateway(self, mcp_content_block: Any) -> List[ContentItem]:
        """Maps a single MCP content block back to a list of gateway ContentItems."""
        # ... (Keep previous robust implementation with type checking and raw storage fallback) ...
        gateway_items: List[ContentItem] = []
        if not mcp_content_block: return gateway_items

        # If content is already a list (e.g., multimodal response), process each block
        if isinstance(mcp_content_block, list):
             for block in mcp_content_block:
                  gateway_items.extend(self._map_single_mcp_block(block))
             return gateway_items
        else:
             # Process single block
             return self._map_single_mcp_block(mcp_content_block)


    def _map_single_mcp_block(self, block: Any) -> List[ContentItem]:
         """Maps a single MCP content block object."""
         gateway_items: List[ContentItem] = []
         try:
            content_type_str = getattr(block, 'type', 'unknown')
            content_type_lower = content_type_str.lower()

            if content_type_lower == "text":
                text = getattr(block, 'text', '')
                gateway_items.append(ContentItem.from_text(text))
            elif content_type_lower == "image":
                b64_data = getattr(block, 'data', None)
                mime = getattr(block, 'mimeType', 'image/png')
                if b64_data: gateway_items.append(ContentItem.from_image_base64(b64_data, mime))
                else: logger.warning("MCP image block missing 'data'")
            # Add mappings for tool_use, tool_result if they can appear here
            elif content_type_lower == "tool_use":
                 raw_data = self._safe_dump_mcp_object(block)
                 gateway_items.append(ContentItem(type=GatewayMCPContentType.TOOL_USE, data={"raw_mcp_tool_use": raw_data}))
            elif content_type_lower == "tool_result":
                 raw_data = self._safe_dump_mcp_object(block)
                 gateway_items.append(ContentItem(type=GatewayMCPContentType.TOOL_RESULT, data={"raw_mcp_tool_result": raw_data}))
            else: # Fallback
                 logger.warning(f"Unsupported MCP content block type '{content_type_str}'. Storing raw.")
                 raw_data = self._safe_dump_mcp_object(block)
                 try: gateway_type = GatewayMCPContentType(content_type_lower)
                 except ValueError: gateway_type = GatewayMCPContentType.FILE
                 gateway_items.append(ContentItem(type=gateway_type, data={"raw": raw_data, "original_type": content_type_str}))

         except Exception as e:
            logger.error(f"Failed to map MCP content block: {e}", exc_info=True)
            gateway_items.append(ContentItem(type=GatewayMCPContentType.FILE, data={"mapping_error": str(e), "raw": self._safe_dump_mcp_object(block)}))

         return gateway_items

    def _safe_dump_mcp_object(self, mcp_obj: Any) -> Any: # Return Any as it might be string or dict
         """Safely attempts to serialize an MCP object to a dict or string."""
         # ... (Keep previous implementation) ...
         if hasattr(mcp_obj, 'model_dump'): # Pydantic v2+
             try: return mcp_obj.model_dump()
             except Exception: pass
         if hasattr(mcp_obj, 'dict'): # Pydantic v1
              try: return mcp_obj.dict()
              except Exception: pass
         try:
             if hasattr(mcp_obj, '__dict__'):
                 # Basic attempt for generic objects
                 return {k: v for k, v in vars(mcp_obj).items() if not k.startswith('_')}
             else: return str(mcp_obj) # Fallback to string
         except Exception as e:
             logger.error(f"Failed to serialize MCP object {type(mcp_obj)}: {e}")
             return {"serialization_error": str(e)}


    def _map_mcp_error(self, error: McpError) -> ErrorDetails:
        """Maps an McpError to the gateway's ErrorDetails model."""
        # Extract code and message from the nested error object if possible
        mcp_error_obj = getattr(error, 'error', None)
        code = str(getattr(mcp_error_obj, 'code', "MCP_UNKNOWN_ERROR"))
        message = getattr(mcp_error_obj, 'message', str(error))
        provider_details = getattr(mcp_error_obj, 'data', {"raw_exception": str(error)})
        # Ensure provider_details is a dict
        if not isinstance(provider_details, dict):
             provider_details = {"details": str(provider_details)}

        return ErrorDetails(
            code=code,
            message=f"MCP Provider Error: {message}",
            level=ErrorLevel.ERROR,
            provider_error_details=provider_details,
            retryable=self._is_retryable_error(error),
            stage="provider_call" # Error occurred during provider interaction
        )

    def _map_error(self, error: Exception, retryable: Optional[bool] = None, stage: Optional[str] = "provider_call") -> ErrorDetails:
        """Maps a generic Exception to ErrorDetails."""
        # Basic mapping for non-MCP errors encountered within the provider
        return ErrorDetails(
            code="PROVIDER_UNEXPECTED_ERROR",
            message=f"MCP provider '{self.provider_id}' encountered an unexpected error: {str(error)}",
            level=ErrorLevel.ERROR,
            provider_error_details={"exception_type": type(error).__name__, "details": str(error)},
            retryable=retryable if retryable is not None else False,
            stage=stage
        )

    def _create_mcp_error_chunk(self, request_id: str, chunk_id: int, error: McpError) -> StreamChunk:
        """Creates an error chunk specifically from an McpError."""
        error_details = self._map_mcp_error(error)
        return StreamChunk(
            chunk_id=chunk_id,
            request_id=request_id,
            finish_reason=FinishReason.ERROR,
            provider_specific_data={"error": error_details.model_dump()}
        )

    def _create_generic_error_chunk(self, request_id: str, chunk_id: int, error: Exception) -> StreamChunk:
        """Creates an error chunk from a generic Exception."""
        error_details = self._map_error(error, stage="provider_stream") # Assume error during streaming
        return StreamChunk(
            chunk_id=chunk_id,
            request_id=request_id,
            finish_reason=FinishReason.ERROR,
            provider_specific_data={"error": error_details.model_dump()}
        )

# --- End of MCPProvider class ---
