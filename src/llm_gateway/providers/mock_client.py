# llm_gateway/providers/mock_client.py

"""
Mock LLM Provider implementation for testing and development purposes.

Simulates responses, delays, errors, tool calls, and streaming based on configuration.
"""

import asyncio
import logging
import random
import uuid
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

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
    ToolUseRequest,
    UsageStats,
)
from llm_gateway.providers.base import BaseProvider # Assuming this exists

logger = logging.getLogger(__name__)

class MockClient(BaseProvider):
    """
    Mock LLM provider for simulating LLM interactions.

    Reads simulation parameters from provider_config.connection_params:
    - simulate_delay_ms (int): Milliseconds to wait before responding. Defaults to 50.
    - simulate_error_code (str): If set, returns an error with this code.
    - simulate_error_message (str): Message for the simulated error.
    - simulate_finish_reason (str): Force a specific FinishReason ('stop', 'length', 'tool_calls', 'content_filtered', 'error').
    - mock_response_text (str): Specific text to return. Defaults to a standard mock message.
    - mock_tool_calls (list[dict]): A list of tool calls to simulate if finish_reason is 'tool_calls'.
      Each dict should represent a ToolUseRequest (e.g., {"id": "tool_...", "type": "function", "function": {"name": "...", "parameters": {...}}}).
    - stream_chunk_delay_ms (int): Delay between stream chunks. Defaults to 10.
    - stream_num_chunks (int): Number of text chunks for streaming. Defaults to 5.
    """

    DEFAULT_DELAY_MS = 50
    DEFAULT_STREAM_CHUNK_DELAY_MS = 10
    DEFAULT_STREAM_NUM_CHUNKS = 5

    def __init__(self, provider_config: ProviderConfig, gateway_config: GatewayConfig):
        """Initialize the Mock Client."""
        super().__init__(provider_config)
        self.gateway_config = gateway_config

        # --- Load Simulation Parameters ---
        conn_params = provider_config.connection_params
        self.delay_ms = conn_params.get("simulate_delay_ms", self.DEFAULT_DELAY_MS)
        self.error_code = conn_params.get("simulate_error_code")
        self.error_message = conn_params.get("simulate_error_message", "Simulated provider error.")
        self.force_finish_reason_str = conn_params.get("simulate_finish_reason")
        self.mock_response = conn_params.get(
            "mock_response_text", "This is a mock response from provider '{provider_id}' for model '{model_id}'."
        )
        self.mock_tool_calls_config = conn_params.get("mock_tool_calls") # Expects list of dicts
        self.stream_chunk_delay_ms = conn_params.get("stream_chunk_delay_ms", self.DEFAULT_STREAM_CHUNK_DELAY_MS)
        self.stream_num_chunks = conn_params.get("stream_num_chunks", self.DEFAULT_STREAM_NUM_CHUNKS)

        logger.info(f"Initialized MockClient provider '{self.provider_id}' with delay={self.delay_ms}ms")
        if self.error_code:
            logger.info(f"--> Will simulate error: {self.error_code}")
        if self.force_finish_reason_str:
            logger.info(f"--> Will force finish_reason: {self.force_finish_reason_str}")
        if self.mock_tool_calls_config:
            logger.info(f"--> Will simulate tool calls: {len(self.mock_tool_calls_config)} calls")


    async def cleanup(self):
        """Mock cleanup - does nothing."""
        logger.debug(f"MockClient cleanup called for '{self.provider_id}'. No action taken.")
        pass

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Simulates generating a response."""
        start_time = datetime.utcnow()
        request_id = request.initial_context.request_id
        model_id = request.config.model_identifier

        logger.debug(f"MockClient '{self.provider_id}' received generate request {request_id} for model {model_id}")

        # 1. Simulate Delay
        actual_delay_ms = self.delay_ms + random.uniform(-self.delay_ms * 0.1, self.delay_ms * 0.1) # Add jitter
        await asyncio.sleep(max(0, actual_delay_ms / 1000.0))
        llm_latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0

        # 2. Simulate Error if configured
        if self.error_code:
            logger.warning(f"MockClient simulating error {self.error_code} for request {request_id}")
            error_details = ErrorDetails(
                code=self.error_code,
                message=self.error_message,
                level=ErrorLevel.ERROR, # Default level
                provider_error_details={"simulation": True}
            )
            return self._create_final_response(
                request=request,
                finish_reason=FinishReason.ERROR,
                error_details=error_details,
                llm_latency_ms=llm_latency_ms,
                total_duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000.0
            )

        # 3. Determine Finish Reason and Content/ToolCalls
        generated_content: Optional[str] = None
        tool_use_requests: Optional[List[ToolUseRequest]] = None
        finish_reason: FinishReason = FinishReason.STOP # Default

        forced_reason = self._get_forced_finish_reason()

        if forced_reason == FinishReason.TOOL_CALLS:
             finish_reason = FinishReason.TOOL_CALLS
             tool_use_requests = self._generate_mock_tool_calls(request.tools)
        elif forced_reason == FinishReason.LENGTH:
             finish_reason = FinishReason.LENGTH
             # Generate partial content if needed, otherwise use standard mock response
             response_text = self.mock_response.format(provider_id=self.provider_id, model_id=model_id)
             generated_content = response_text[:max(10, len(response_text)//2)] # Truncated
        elif forced_reason == FinishReason.CONTENT_FILTERED:
             finish_reason = FinishReason.CONTENT_FILTERED
             generated_content = None # Content is filtered
        elif forced_reason == FinishReason.ERROR: # Should have been caught above, but handle defensively
             finish_reason = FinishReason.ERROR
             generated_content = None
        else: # Default STOP reason
             finish_reason = FinishReason.STOP
             generated_content = self.mock_response.format(provider_id=self.provider_id, model_id=model_id)


        # 4. Simulate Usage
        # Generate some plausible looking token counts
        prompt_str = self._extract_prompt_text(request.prompt_content)
        prompt_tokens = len(prompt_str.split()) + random.randint(5, 20) # Rough estimate
        completion_tokens = 0
        if generated_content:
             completion_tokens = len(generated_content.split()) + random.randint(5, 15)
        elif tool_use_requests:
             completion_tokens = random.randint(10, 30) * len(tool_use_requests) # Estimate for tool call overhead

        usage = UsageStats(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


        # 5. Construct and return LLMResponse
        return self._create_final_response(
            request=request,
            finish_reason=finish_reason,
            generated_content=generated_content,
            tool_use_requests=tool_use_requests,
            usage=usage,
            llm_latency_ms=llm_latency_ms,
            total_duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000.0
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Simulates generating a streaming response."""
        request_id = request.initial_context.request_id
        model_id = request.config.model_identifier
        chunk_index = 0

        logger.debug(f"MockClient '{self.provider_id}' received stream request {request_id} for model {model_id}")

        # 1. Simulate Initial Delay (optional, could delay first chunk instead)
        initial_delay_ms = self.delay_ms / 2 # Example: faster time-to-first-token
        await asyncio.sleep(max(0, initial_delay_ms / 1000.0))

        # 2. Simulate Error if configured
        if self.error_code:
            logger.warning(f"MockClient simulating stream error {self.error_code} for request {request_id}")
            error_details = ErrorDetails(
                code=self.error_code, message=self.error_message, level=ErrorLevel.ERROR, provider_error_details={"simulation": True}
            )
            yield StreamChunk(chunk_id=0, request_id=request_id, finish_reason=FinishReason.ERROR, provider_specific_data={"error": error_details.model_dump()})
            return

        # 3. Determine Finish Reason and simulate streaming
        forced_reason = self._get_forced_finish_reason()

        if forced_reason == FinishReason.TOOL_CALLS:
            # Simulate yielding tool call info (e.g., yield one chunk per tool call, then final chunk)
            tool_calls = self._generate_mock_tool_calls(request.tools)
            if tool_calls:
                 for i, tool_call in enumerate(tool_calls):
                      # Yield each tool call individually or grouped? Grouping is simpler for mock.
                      yield StreamChunk(chunk_id=chunk_index, request_id=request_id, delta_tool_calls=[tool_call])
                      chunk_index += 1
                      await asyncio.sleep(max(0, self.stream_chunk_delay_ms / 1000.0))

            # Yield final chunk for tool calls
            usage = self._generate_mock_usage(request.prompt_content, None, tool_calls)
            yield StreamChunk(chunk_id=chunk_index, request_id=request_id, finish_reason=FinishReason.TOOL_CALLS, usage_update=usage)

        elif forced_reason == FinishReason.LENGTH or forced_reason == FinishReason.CONTENT_FILTERED or forced_reason == FinishReason.ERROR:
             # Simulate abrupt end or empty response for these cases in stream
             finish_reason = forced_reason
             delta_text = None
             if finish_reason == FinishReason.LENGTH:
                   # Optionally yield some partial text first
                   response_text = self.mock_response.format(provider_id=self.provider_id, model_id=model_id)
                   partial_text = response_text[:max(10, len(response_text)//2)]
                   yield StreamChunk(chunk_id=chunk_index, request_id=request_id, delta_text=partial_text)
                   chunk_index += 1
                   await asyncio.sleep(max(0, self.stream_chunk_delay_ms / 1000.0))
             
             usage = self._generate_mock_usage(request.prompt_content, partial_text if finish_reason == FinishReason.LENGTH else None, None)
             yield StreamChunk(chunk_id=chunk_index, request_id=request_id, finish_reason=finish_reason, usage_update=usage)

        else: # Default STOP reason - stream text chunks
             response_text = self.mock_response.format(provider_id=self.provider_id, model_id=model_id)
             words = response_text.split()
             num_chunks = max(1, self.stream_num_chunks)
             words_per_chunk = (len(words) + num_chunks - 1) // num_chunks # Ceiling division

             accumulated_text_for_usage = ""
             for i in range(num_chunks):
                 start_index = i * words_per_chunk
                 end_index = min((i + 1) * words_per_chunk, len(words))
                 if start_index >= len(words): break # Avoid empty chunks if response is short

                 text_chunk = " ".join(words[start_index:end_index])
                 # Add back leading space if not first chunk
                 if i > 0: text_chunk = " " + text_chunk

                 accumulated_text_for_usage += text_chunk
                 yield StreamChunk(chunk_id=chunk_index, request_id=request_id, delta_text=text_chunk)
                 chunk_index += 1
                 if i < num_chunks - 1: # Don't delay after last text chunk
                      await asyncio.sleep(max(0, self.stream_chunk_delay_ms / 1000.0))

             # Yield final chunk with STOP reason and usage
             usage = self._generate_mock_usage(request.prompt_content, accumulated_text_for_usage, None)
             yield StreamChunk(chunk_id=chunk_index, request_id=request_id, finish_reason=FinishReason.STOP, usage_update=usage)

        logger.debug(f"MockClient stream finished for request {request_id}")


    # --- Helper Methods ---

    def _get_forced_finish_reason(self) -> Optional[FinishReason]:
         """Parse the forced finish reason string from config."""
         if self.force_finish_reason_str:
              try:
                   return FinishReason(self.force_finish_reason_str.lower())
              except ValueError:
                   logger.warning(f"Invalid simulate_finish_reason '{self.force_finish_reason_str}' configured. Ignoring.")
         return None

    def _extract_prompt_text(self, content: Union[str, List[ContentItem], Any]) -> str:
         """Extracts simple text from prompt content for mock usage calculation."""
         if isinstance(content, str):
              return content
         elif isinstance(content, list):
              text = ""
              for item in content:
                   if item.type == GatewayContentType.TEXT and item.text_content:
                        text += item.text_content + "\n"
              return text
         return ""

    def _generate_mock_usage(self, prompt_content: Any, generated_text: Optional[str], tool_calls: Optional[List[ToolUseRequest]]) -> UsageStats:
         """Generates plausible mock usage stats."""
         prompt_str = self._extract_prompt_text(prompt_content)
         prompt_tokens = len(prompt_str.split()) + random.randint(5, 20)
         completion_tokens = 0
         if generated_text:
              completion_tokens = len(generated_text.split()) + random.randint(5, 15)
         elif tool_calls:
              completion_tokens = random.randint(10, 30) * len(tool_calls)
         return UsageStats(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def _generate_mock_tool_calls(self, available_tools: Optional[List[ToolDefinition]]) -> List[ToolUseRequest]:
        """Generates mock ToolUseRequests based on config or available tools."""
        if self.mock_tool_calls_config:
             # Generate from config
             requests = []
             for i, tool_config in enumerate(self.mock_tool_calls_config):
                  try:
                       # Basic validation/parsing of config dict
                       func_config = tool_config.get("function", {})
                       requests.append(ToolUseRequest(
                            id=tool_config.get("id", f"mocktool_{uuid.uuid4()}"),
                            type=tool_config.get("type", "function"),
                            function=ToolFunction(
                                 name=func_config.get("name", f"mock_func_{i}"),
                                 parameters=func_config.get("parameters", {"arg": "mock_value"}),
                                 description=func_config.get("description")
                            )
                       ))
                  except Exception as e:
                       logger.error(f"Error parsing mock_tool_calls config item: {tool_config}. Error: {e}")
             return requests
        elif available_tools:
             # Pick one available tool randomly
             chosen_tool_def = random.choice(available_tools)
             func = chosen_tool_def.function
             # Generate plausible args based on schema (simplified)
             mock_args = {}
             if func.parameters and isinstance(func.parameters, dict) and "properties" in func.parameters:
                  for param_name, schema in func.parameters["properties"].items():
                       if schema.get("type") == "string": mock_args[param_name] = "mock string value"
                       elif schema.get("type") == "integer": mock_args[param_name] = random.randint(1, 100)
                       elif schema.get("type") == "boolean": mock_args[param_name] = random.choice([True, False])
                       else: mock_args[param_name] = "mock value"
             else:
                  mock_args = {"arg": "mock_value"} # Default if no schema

             return [ToolUseRequest(
                  id=f"mocktool_{uuid.uuid4()}",
                  type="function",
                  function=ToolFunction(
                       name=func.name,
                       description=func.description,
                       parameters=mock_args
                  )
             )]
        else:
             # No config and no available tools provided
             return []


    def _create_final_response(
        self,
        request: LLMRequest,
        finish_reason: FinishReason,
        generated_content: Optional[Union[str, List[ContentItem]]] = None,
        tool_use_requests: Optional[List[ToolUseRequest]] = None,
        usage: Optional[UsageStats] = None,
        error_details: Optional[ErrorDetails] = None,
        llm_latency_ms: Optional[float] = 0.0,
        total_duration_ms: Optional[float] = 0.0,
    ) -> LLMResponse:
        """Helper to construct the final LLMResponse."""

        perf_metrics = PerformanceMetrics(
            total_duration_ms=total_duration_ms,
            llm_latency_ms=llm_latency_ms,
            # Ideally, pre/post processing times would be added by manager/pipeline
        )

        return LLMResponse(
            version=request.version,
            request_id=request.initial_context.request_id,
            generated_content=generated_content,
            finish_reason=finish_reason,
            tool_use_requests=tool_use_requests,
            usage=usage,
            compliance_result=None,  # Not simulated here
            final_context=request.initial_context, # Pass context through
            error_details=error_details,
            performance_metrics=perf_metrics,
            # Optionally add mock provider-specific metadata
            # extensions={"provider_metadata": {"mock_provider_id": self.provider_id}}
        )

# --- End of MockClient class ---