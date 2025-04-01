# llm_gateway/core/client.py
import asyncio
import logging
import uuid # Import uuid
from datetime import datetime # Import datetime
from typing import AsyncGenerator, Dict, List, Optional, Union, Any, Tuple

from llm_gateway.core.models import (
    BatchLLMRequest, BatchLLMResponse, LLMRequest, LLMResponse, StreamChunk,
    GatewayConfig, InterventionContext, ErrorDetails, ErrorLevel, # Import needed models
    PerformanceMetrics, FinishReason
)
from llm_gateway.core.factory import ProviderFactory, ProviderFactoryError # Import FactoryError
from llm_gateway.core.manager import InterventionManager
# Import common provider errors if needed for specific handling, or rely on mapping
# from openai._exceptions import APITimeoutError as OpenAITimeout, RateLimitError as OpenAIRateLimit # Example
# from anthropic._exceptions import APITimeoutError as AnthropicTimeout, RateLimitError as AnthropicRateLimit # Example

logger = logging.getLogger(__name__)

class LLMGatewayClient:
    def __init__(self, config: GatewayConfig, provider_factory: Optional[ProviderFactory] = None):
         self.config = config
         # Use a central factory instance
         self.provider_factory = provider_factory or ProviderFactory()
         # Pass the factory to the manager
         self.intervention_manager = InterventionManager(self.provider_factory, self.config)
         # Configure semaphore from GatewayConfig
         batch_limit = config.additional_config.get("max_concurrent_batch_requests", 10)
         self._batch_semaphore = asyncio.Semaphore(batch_limit)
         logger.info(f"Initialized LLMGatewayClient with batch concurrency limit: {batch_limit}")


    async def generate(self, request: LLMRequest) -> LLMResponse:
         """Process a single LLM request."""
         # Add request ID to logger context if possible (using contextvars or similar)
         logger.info(f"Processing generate request: {request.initial_context.request_id}")
         try:
              # Delegate to intervention manager
              response = await self.intervention_manager.process_request(request)
              logger.info(f"Request successful: {request.initial_context.request_id}")
              return response
         except ProviderFactoryError as e: # Handle factory errors specifically
              logger.error(f"Provider creation failed for request {request.initial_context.request_id}: {e}", exc_info=True)
              return self._create_error_response(request, e, "PROVIDER_INIT_FAILED")
         except Exception as e:
              logger.error(f"Request processing failed: {request.initial_context.request_id}", exc_info=True)
              # Pass the original exception for more specific mapping if needed
              return self._create_error_response(request, e)

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Process a streaming request."""
        request_id = request.initial_context.request_id
        logger.info(f"Processing stream request: {request_id}")
        try:
            async for chunk in self.intervention_manager.process_stream(request):
                yield chunk
            logger.info(f"Stream finished successfully: {request_id}")
        except ProviderFactoryError as e:
              logger.error(f"Provider creation failed for stream {request_id}: {e}", exc_info=True)
              yield self._create_error_chunk(request_id, e, "PROVIDER_INIT_FAILED")
        except Exception as e:
            logger.error(f"Stream processing failed: {request_id}", exc_info=True)
            yield self._create_error_chunk(request_id, e)

    async def process_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """Process a batch of LLM requests."""
        batch_start_time = datetime.utcnow()
        logger.info(f"Processing batch request: {batch_request.batch_id} ({len(batch_request.requests)} requests)")
        async with self._batch_semaphore:
            # Create tasks that store the original request for error handling
            tasks = [
                self._process_single_request_in_batch(req)
                for req in batch_request.requests
            ]
            results_with_requests = await asyncio.gather(*tasks, return_exceptions=True)

            valid_responses: List[LLMResponse] = []
            for req, result in results_with_requests:
                if isinstance(result, Exception):
                     # Create error response using the stored request
                     logger.warning(f"Sub-request {req.initial_context.request_id} in batch {batch_request.batch_id} failed.", exc_info=result)
                     valid_responses.append(self._create_error_response(req, result))
                else:
                     valid_responses.append(result) # Already an LLMResponse

            total_duration_ms = (datetime.utcnow() - batch_start_time).total_seconds() * 1000
            logger.info(f"Batch processing complete: {batch_request.batch_id}, Duration: {total_duration_ms:.2f}ms")
            return BatchLLMResponse(
                batch_id=batch_request.batch_id,
                responses=valid_responses,
                total_duration_ms=total_duration_ms
                # Aggregated usage etc. calculated in model validator
            )

    # Store original request alongside the coroutine for batch error handling
    async def _process_single_request_in_batch(self, request: LLMRequest) -> Tuple[LLMRequest, LLMResponse]:
         """Wrapper for batch processing that returns request context on error."""
         try:
              # No need for try/except here, gather handles exceptions
              response = await self.intervention_manager.process_request(request)
              return request, response
         except Exception as e:
              # Let asyncio.gather catch the exception, just return request for context
              raise e # Re-raise the exception


    async def close(self):
        """Clean up provider resources."""
        logger.info("Shutting down gateway client and cleaning up providers...")
        await self.provider_factory.cleanup_all()
        logger.info("Gateway client shutdown complete.")

    def _create_error_response(self, request: Optional[LLMRequest], error: Exception, default_code: str = "CLIENT_PROCESSING_ERROR") -> LLMResponse:
         """Create a standardized error response."""
         request_id = request.initial_context.request_id if request else f"error_{uuid.uuid4()}"
         context = request.initial_context if request else InterventionContext(request_id=request_id)

         # Basic error mapping (can be expanded)
         # TODO: Add more sophisticated mapping based on specific exception types
         error_code = default_code
         message = str(error)
         level = ErrorLevel.ERROR
         retryable = False
         provider_details = {"exception_type": type(error).__name__, "details": str(error)}

         # Example: check for specific provider errors if needed (better done in provider mapping)
         # if isinstance(error, OpenAIRateLimit) or isinstance(error, AnthropicRateLimit):
         #     error_code = "PROVIDER_RATE_LIMIT"
         #     retryable = True
         # elif isinstance(error, ProviderFactoryError):
         #     error_code = "PROVIDER_INIT_FAILED"
         #     retryable = False # Usually config issue

         error_details = ErrorDetails(
             code=error_code,
             message=message,
             level=level,
             retryable=retryable,
             provider_error_details=provider_details
         )

         # Estimate duration if possible
         duration = (datetime.utcnow() - context.timestamp_start).total_seconds() * 1000 if context else 0

         return LLMResponse(
             request_id=request_id,
             generated_content=None,
             error_details=error_details,
             final_context=context, # Return the context state at time of error
             finish_reason=FinishReason.ERROR, # Set finish reason
             performance_metrics=PerformanceMetrics(
                  total_duration_ms=duration,
                  # Other metrics might be unknown
             )
         )

    def _create_error_chunk(self, request_id: str, error: Exception, error_code: str = "STREAM_PROCESSING_ERROR") -> StreamChunk:
        """Create an error chunk for streaming responses."""
        # Basic error mapping for stream errors
        error_details = ErrorDetails(
            code=error_code,
            message=str(error),
            level=ErrorLevel.ERROR,
            provider_error_details={"exception_type": type(error).__name__}
        )
        return StreamChunk(
            chunk_id=999, # Use a high index for error chunk? Or -1?
            request_id=request_id,
            finish_reason=FinishReason.ERROR,
            provider_specific_data={"error": error_details.model_dump()}
        )

    # ... (Utility Methods: get_active_providers, health_check, warmup_providers) ...
    # Modify health_check and warmup to use the new get_provider method
    async def health_check(self) -> Dict[str, Any]:
         """Check connectivity to all configured providers."""
         status = {}
         # Find provider configs from the main GatewayConfig (assuming it holds them)
         # This needs adjustment based on how provider configs are stored/accessed
         # Example: Assume config has a 'providers' dict: {provider_id: ProviderConfig}
         provider_configs = getattr(self.config, 'providers', {})
         if not provider_configs:
              logger.warning("No provider configurations found in GatewayConfig for health check.")
              # Look for allowed providers as fallback
              provider_ids_to_check = self.config.allowed_providers
              # Need dummy ProviderConfig if only IDs available
              if provider_ids_to_check and not provider_configs:
                  logger.warning("Checking allowed_providers without full configs.")
                  # This path is problematic - health check needs config
                  for provider_id in provider_ids_to_check:
                       status[provider_id] = {"status": "unknown", "error": "Missing provider configuration"}
                  return status
              elif not provider_ids_to_check:
                  return {"status": "no_providers_configured"}


         for provider_id, provider_conf in provider_configs.items():
              try:
                   # Get potentially existing or new provider instance
                   provider = await self.provider_factory.get_provider(provider_id, provider_conf, self.config)
                   status[provider_id] = await provider.health_check()
              except Exception as e:
                   status[provider_id] = {"status": "unhealthy", "provider_id": provider_id, "error": f"Failed to get/check provider: {str(e)}"}
         return status

    async def warmup_providers(self):
         """Pre-initialize providers based on configuration."""
         # Needs access to full ProviderConfig objects for preload_providers
         preload_ids = getattr(self.config, 'preload_providers', [])
         provider_configs = getattr(self.config, 'providers', {}) # Assume config holds provider details

         logger.info(f"Warming up providers: {preload_ids}")
         warmed_up = []
         for provider_id in preload_ids:
              provider_conf = provider_configs.get(provider_id)
              if provider_conf:
                   try:
                        await self.provider_factory.get_provider(provider_id, provider_conf, self.config)
                        warmed_up.append(provider_id)
                   except Exception as e:
                        logger.error(f"Failed to warm up provider {provider_id}: {e}")
              else:
                   logger.warning(f"Cannot warm up provider {provider_id}: Configuration not found.")
         logger.info(f"Successfully warmed up providers: {warmed_up}")
