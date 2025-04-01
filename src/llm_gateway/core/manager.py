# llm_gateway/core/manager.py

import asyncio
import logging
import time
from typing import Dict, List, Optional, Type

from llm_gateway.core.models import (
    InterventionContext, LLMRequest, LLMResponse, StreamChunk,
    PerformanceMetrics, ComplianceResult, FinishReason
)
from llm_gateway.core.factory import ProviderFactory
from llm_gateway.interventions.base import BaseIntervention

logger = logging.getLogger(__name__)

class InterventionManager:
    """
    Manages the intervention pipeline for request preprocessing, 
    generation monitoring, and response postprocessing.
    """
    
    def __init__(self, provider_factory: ProviderFactory):
        """Initialize the intervention manager."""
        self.provider_factory = provider_factory
        self.pre_interventions: Dict[str, BaseIntervention] = {}  # Load interventions
        self.post_interventions: Dict[str, BaseIntervention] = {}
        self._load_interventions()
        
    def _load_interventions(self) -> None:
        """Load and initialize all registered interventions."""
        # Implementation would load interventions from config
        # For now, this is a placeholder
        
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """
        Process a request through the complete intervention pipeline:
        1. Run pre-generation interventions
        2. Call the provider
        3. Run post-generation interventions
        """
        start_time = time.time()
        
        # 1. Pre-processing interventions
        pre_start = time.time()
        modified_request = await self._run_pre_interventions(request)
        pre_duration = (time.time() - pre_start) * 1000
        
        # 2. Get provider and generate response
        provider = await self.provider_factory.get_provider(modified_request.config.model_identifier)
        generation_start = time.time()
        response = await provider.generate(modified_request)
        generation_duration = (time.time() - generation_start) * 1000
        
        # 3. Post-processing interventions
        post_start = time.time()
        final_response = await self._run_post_interventions(response)
        post_duration = (time.time() - post_start) * 1000
        
        # 4. Update metrics
        total_duration = (time.time() - start_time) * 1000
        final_response.performance_metrics = PerformanceMetrics(
            total_duration_ms=total_duration,
            llm_latency_ms=response.performance_metrics.llm_latency_ms if response.performance_metrics else None,
            pre_processing_duration_ms=pre_duration,
            post_processing_duration_ms=post_duration,
        )
        
        return final_response
    
    async def _run_pre_interventions(self, request: LLMRequest) -> LLMRequest:
        """Run all enabled pre-interventions in sequence."""
        current_request = request
        
        for intervention_name in request.initial_context.intervention_config.enabled_pre_interventions:
            if intervention_name in self.pre_interventions:
                intervention = self.pre_interventions[intervention_name]
                try:
                    current_request = await intervention.process_request(current_request)
                except Exception as e:
                    logger.error(f"Error in pre-intervention {intervention_name}: {e}")
                    if not request.initial_context.intervention_config.fail_open:
                        raise
        
        return current_request
    
    async def _run_post_interventions(self, response: LLMResponse) -> LLMResponse:
        """Run all enabled post-interventions in sequence."""
        current_response = response
        
        # Similar implementation to pre-interventions, but for post-processing
        
        return current_response
    
    # Additional methods for streaming and batch processing would be added here
