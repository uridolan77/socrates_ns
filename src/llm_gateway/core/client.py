# llm_gateway/core/client.py

import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional, Union

from llm_gateway.core.models import (
    BatchLLMRequest,
    BatchLLMResponse,
    LLMRequest,
    LLMResponse,
    StreamChunk,
    GatewayConfig,
    InterventionContext,
    ErrorDetails
)
from llm_gateway.core.factory import ProviderFactory
from llm_gateway.core.manager import InterventionManager

logger = logging.getLogger(__name__)

class LLMGatewayClient:
    """
    Primary interface for interacting with the LLM Gateway.
    Handles request processing, streaming, batching, and error handling.
    """

    def __init__(self, config: GatewayConfig, provider_factory: Optional[ProviderFactory] = None):
        self.config = config
        self.provider_factory = provider_factory or ProviderFactory(config)
        self.intervention_manager = InterventionManager(self.provider_factory)
        self._batch_semaphore = asyncio.Semaphore(config.max_concurrent_batch_requests)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Process a single LLM request through the full intervention pipeline.
        
        Args:
            request: LLM request object with prompt and configuration
            
        Returns:
            LLMResponse with generated content and metadata
        """
        try:
            return await self.intervention_manager.process_request(request)
        except Exception as e:
            logger.error(f"Request failed: {request.initial_context.request_id}", exc_info=True)
            return self._create_error_response(request, e)

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a streaming request and yield response chunks.
        
        Args:
            request: LLM request object with streaming enabled
            
        Yields:
            StreamChunk objects with incremental response data
        """
        try:
            async for chunk in self.intervention_manager.process_stream(request):
                yield chunk
        except Exception as e:
            logger.error(f"Stream failed: {request.initial_context.request_id}", exc_info=True)
            yield self._create_error_chunk(request.initial_context.request_id, e)

    async def process_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """
        Process a batch of LLM requests with controlled concurrency.
        
        Args:
            batch_request: Batch request containing multiple LLM requests
            
        Returns:
            BatchLLMResponse with aggregated results and statistics
        """
        async with self._batch_semaphore:
            tasks = [self._process_single_request(req) for req in batch_request.requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_responses = []
            for result in results:
                if isinstance(result, Exception):
                    valid_responses.append(self._create_error_response(None, result))
                else:
                    valid_responses.append(result)
            
            return BatchLLMResponse(
                batch_id=batch_request.batch_id,
                responses=valid_responses,
                total_duration_ms=(datetime.utcnow() - batch_request.created_at).total_seconds() * 1000
            )

    async def _process_single_request(self, request: LLMRequest) -> LLMResponse:
        """Wrapper for individual batch request processing"""
        try:
            return await self.generate(request)
        except Exception as e:
            logger.error(f"Batch request failed: {request.initial_context.request_id}", exc_info=True)
            return self._create_error_response(request, e)

    async def close(self):
        """Clean up all provider resources"""
        await self.provider_factory.cleanup_all()
        logger.info("Gateway client shutdown complete")

    def _create_error_response(self, request: Optional[LLMRequest], error: Exception) -> LLMResponse:
        """Create error response with standardized formatting"""
        error_details = ErrorDetails(
            code="CLIENT_ERROR",
            message=str(error),
            level=ErrorLevel.ERROR,
            retryable=isinstance(error, (APITimeoutError, RateLimitError))
        )
        
        return LLMResponse(
            request_id=request.initial_context.request_id if request else str(uuid.uuid4()),
            generated_content=None,
            error_details=error_details,
            final_context=request.initial_context if request else InterventionContext(),
            performance_metrics=PerformanceMetrics(
                total_duration_ms=0,
                gateway_overhead_ms=0
            )
        )

    def _create_error_chunk(self, request_id: str, error: Exception) -> StreamChunk:
        """Create error chunk for streaming responses"""
        return StreamChunk(
            chunk_id=0,
            request_id=request_id,
            finish_reason=FinishReason.ERROR,
            provider_specific_data={
                "error": ErrorDetails(
                    code="STREAM_ERROR",
                    message=str(error),
                    level=ErrorLevel.ERROR
                ).model_dump()
            }
        )

    # ---------- Utility Methods ---------- 
    
    def get_active_providers(self) -> List[str]:
        """Get list of currently loaded provider IDs"""
        return list(self.provider_factory.provider_instances.keys())

    async def health_check(self) -> Dict[str, Any]:
        """Check connectivity to all configured providers"""
        status = {}
        for provider_id in self.config.allowed_providers:
            try:
                provider = await self.provider_factory.get_provider(provider_id)
                status[provider_id] = await provider.health_check()
            except Exception as e:
                status[provider_id] = {"status": "unhealthy", "error": str(e)}
        return status

    async def warmup_providers(self):
        """Pre-initialize providers based on configuration"""
        for provider_id in self.config.preload_providers:
            await self.provider_factory.get_provider(provider_id)
        logger.info(f"Pre-warmed providers: {self.get_active_providers()}")
