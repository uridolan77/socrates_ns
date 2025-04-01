# llm_gateway/core/manager.py

import asyncio
import logging
import time
from typing import Dict, List, Optional, Type, AsyncGenerator

from llm_gateway.core.models import (
    InterventionContext, LLMRequest, LLMResponse, StreamChunk,
    PerformanceMetrics, ComplianceResult, FinishReason
)
from llm_gateway.core.factory import ProviderFactory
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
from llm_gateway.interventions.base import BaseIntervention
from llm_gateway.interventions.factory import InterventionFactory, InterventionFactoryError
from llm_gateway.interventions.manager import InterventionManager
from llm_gateway.providers.base import BaseProvider
from llm_gateway.core.factory import In
logger = logging.getLogger(__name__)

class InterventionManager:
    def __init__(self, provider_factory: ProviderFactory, gateway_config: GatewayConfig):
        self.provider_factory = provider_factory
        self.gateway_config = gateway_config
        # Use an InterventionFactory to load/manage interventions
        self.intervention_factory = InterventionFactory(gateway_config)
        self.pre_interventions: Dict[str, BaseIntervention] = {}
        self.post_interventions: Dict[str, BaseIntervention] = {}
        self._load_interventions() # Load interventions on init

    def _load_interventions(self) -> None:
        """Load and initialize interventions using the factory."""
        logger.info("Loading interventions...")
        # Example: Load based on config or discovery
        # Replace with actual loading logic
        enabled_interventions = self.gateway_config.additional_config.get("enabled_interventions", [])
        for name in enabled_interventions:
             try:
                  intervention = self.intervention_factory.get_intervention(name)
                  if intervention.hook_type == "pre": # Assume hook_type attribute
                       self.pre_interventions[name] = intervention
                       logger.info(f"Loaded pre-intervention: {name}")
                  elif intervention.hook_type == "post":
                       self.post_interventions[name] = intervention
                       logger.info(f"Loaded post-intervention: {name}")
                  # Add support for stream interventions if needed
             except Exception as e:
                  logger.error(f"Failed to load intervention '{name}': {e}", exc_info=True)


    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process a request through the complete intervention pipeline."""
        start_time = time.time()
        context = request.initial_context
        response: Optional[LLMResponse] = None
        error_details: Optional[ErrorDetails] = None

        try:
            # 1. Pre-processing interventions
            pre_start = time.time()
            modified_request = await self._run_pre_interventions(request)
            pre_duration = (time.time() - pre_start) * 1000
            context = modified_request.initial_context # Update context if modified

            # --- Add check for pre-intervention termination ---
            # An intervention might decide to block the request entirely
            if context.intervention_data.get("block_request", False):
                logger.warning(f"Request {context.request_id} blocked by pre-intervention: {context.intervention_data.get('block_reason')}")
                # Create an appropriate blocked response
                return LLMResponse( # Simplified - use error response helper
                    request_id=context.request_id,
                    finish_reason=FinishReason.CONTENT_FILTERED, # Or a specific 'BLOCKED' reason
                    final_context=context,
                    error_details=ErrorDetails(code="INTERVENTION_BLOCK", message=context.intervention_data.get('block_reason', 'Blocked by intervention')),
                    performance_metrics=PerformanceMetrics(total_duration_ms=(time.time() - start_time) * 1000, pre_processing_duration_ms=pre_duration)
                )

            # 2. Get provider and generate response
            # Fetch provider config based on model_identifier
            # This lookup logic needs refinement based on how configs are stored/passed
            provider_configs = getattr(self.gateway_config, 'providers', {})
            provider_id = self._get_provider_id_for_model(modified_request.config.model_identifier) # Need this helper
            provider_conf = provider_configs.get(provider_id)
            if not provider_conf:
                raise ValueError(f"Configuration not found for provider handling model '{modified_request.config.model_identifier}'")

            provider = await self.provider_factory.get_provider(provider_id, provider_conf, self.gateway_config)

            generation_start = time.time()
            response = await provider.generate(modified_request)
            generation_duration = (time.time() - generation_start) * 1000

            # Carry over context from the original request unless provider modifies it
            # The provider's LLMResponse should ideally contain the *final* context state *after* the LLM call
            # but before post-processing. Let's assume the provider returns the context it used.
            # If provider doesn't return context, use the one from modified_request.
            context = response.final_context if response and response.final_context else context

            # 3. Post-processing interventions
            post_start = time.time()
            if response and not response.error_details: # Only run post if LLM call succeeded
                 response = await self._run_post_interventions(response, context)
            post_duration = (time.time() - post_start) * 1000


        except Exception as e:
             logger.error(f"Error during intervention manager processing for {context.request_id}: {e}", exc_info=True)
             # Create a generic error response if one wasn't generated by provider
             if response is None or response.error_details is None:
                # Need a way to create error response using context
                error_details = ErrorDetails(code="MANAGER_ERROR", message=str(e)) # Simplified
                response = LLMResponse(request_id=context.request_id, final_context=context, error_details=error_details, finish_reason=FinishReason.ERROR)

        # 4. Finalize Metrics and Response
        total_duration = (time.time() - start_time) * 1000

        # Ensure response exists even if errors occurred early
        if response is None:
             response = LLMResponse(request_id=context.request_id, final_context=context, error_details=error_details or ErrorDetails(code="UNKNOWN_MANAGER_ERROR"), finish_reason=FinishReason.ERROR)


        # Update performance metrics on the final response object
        llm_latency = response.performance_metrics.llm_latency_ms if response.performance_metrics else None
        # Preserve existing metrics if they exist, otherwise create new
        perf = response.performance_metrics or PerformanceMetrics()
        final_perf = PerformanceMetrics(
            total_duration_ms=total_duration,
            llm_latency_ms=llm_latency, # Comes from provider response map
            pre_processing_duration_ms=pre_duration if 'pre_duration' in locals() else None,
            post_processing_duration_ms=post_duration if 'post_duration' in locals() else None,
            # Add other potential timings like compliance check duration
            # gateway_overhead_ms calculated by model validator
        )

        # Use model_copy to update immutable fields if necessary
        final_response_dict = response.model_dump()
        final_response_dict['performance_metrics'] = final_perf
        final_response_dict['final_context'] = context # Ensure latest context is set

        return LLMResponse(**final_response_dict)


    async def process_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Process streaming requests with interventions (conceptual)."""
        context = request.initial_context
        logger.info(f"Manager processing stream: {context.request_id}")

        try:
            # 1. Pre-processing (on the initial request)
            pre_start = time.time()
            modified_request = await self._run_pre_interventions(request)
            pre_duration = (time.time() - pre_start) * 1000
            context = modified_request.initial_context
            logger.debug(f"Stream pre-processing done for {context.request_id} in {pre_duration:.2f}ms")

            # Check for pre-intervention block
            if context.intervention_data.get("block_request", False):
                 logger.warning(f"Stream request {context.request_id} blocked by pre-intervention.")
                 yield StreamChunk(chunk_id=0, request_id=context.request_id, finish_reason=FinishReason.CONTENT_FILTERED, provider_specific_data={"error": {"code": "INTERVENTION_BLOCK", "message": context.intervention_data.get('block_reason')}})
                 return

            # 2. Get Provider and Start Streaming
            provider_id = self._get_provider_id_for_model(modified_request.config.model_identifier)
            provider_configs = getattr(self.gateway_config, 'providers', {})
            provider_conf = provider_configs.get(provider_id)
            if not provider_conf: raise ValueError(f"Config not found for provider handling model '{modified_request.config.model_identifier}'")
            provider = await self.provider_factory.get_provider(provider_id, provider_conf, self.gateway_config)

            # 3. Stream Interventions (Chunk-level processing - Complex!)
            # This requires interventions designed for streaming.
            # Simplest approach: buffer chunks and apply post-interventions at the end.
            # More complex: apply filtering/modification interventions per chunk.

            # --- Example: Simple Pass-through Streaming with Final Post-Intervention ---
            buffer = []
            final_stream_response_metadata = {} # To store usage, finish_reason etc. from final chunk

            async for chunk in provider.generate_stream(modified_request):
                buffer.append(chunk)
                # Store metadata from the last chunk
                if chunk.finish_reason: final_stream_response_metadata['finish_reason'] = chunk.finish_reason
                if chunk.usage_update: final_stream_response_metadata['usage'] = chunk.usage_update
                if chunk.provider_specific_data: final_stream_response_metadata['provider_specific_data'] = chunk.provider_specific_data

                # Yield chunk immediately (no chunk-level intervention in this simple example)
                yield chunk

            # 4. Post-processing (on the *aggregated* result - potentially memory intensive)
            post_start = time.time()
            # Reconstruct the full response (or relevant parts) for post-interventions
            # This is a simplification; real post-interventions might need more context or operate differently on streams.
            reconstructed_content_items = []
            final_tool_calls = None
            for chunk in buffer:
                 if chunk.delta_text: reconstructed_content_items.append(ContentItem.from_text(chunk.delta_text))
                 if chunk.delta_content_items: reconstructed_content_items.extend(chunk.delta_content_items)
                 if chunk.delta_tool_calls: final_tool_calls = chunk.delta_tool_calls # Assume last chunk has final list

            # Create a pseudo-response object for post-intervention processing
            pseudo_response = LLMResponse(
                 request_id=context.request_id,
                 generated_content=reconstructed_content_items, # This might need flattening/joining text
                 finish_reason=final_stream_response_metadata.get('finish_reason', FinishReason.UNKNOWN),
                 usage=final_stream_response_metadata.get('usage'),
                 tool_use_requests=final_tool_calls,
                 final_context=context, # Pass context
                 # Include other metadata if available
            )

            final_pseudo_response = await self._run_post_interventions(pseudo_response, context)
            post_duration = (time.time() - post_start) * 1000
            logger.debug(f"Stream post-processing done for {context.request_id} in {post_duration:.2f}ms")

            # Note: This simple approach doesn't modify the already yielded chunks.
            # A true streaming intervention system would need to yield modified chunks
            # or potentially a final chunk with modifications/compliance results.
            # We could yield a final "summary" chunk here if needed.
            if final_pseudo_response.compliance_result or final_pseudo_response.error_details:
                 yield StreamChunk(
                      chunk_id=len(buffer), # Final chunk index
                      request_id=context.request_id,
                      # No delta content in summary chunk
                      finish_reason=final_pseudo_response.finish_reason,
                      usage_update=final_pseudo_response.usage,
                      provider_specific_data={ # Add post-processing results here
                           "compliance_result": final_pseudo_response.compliance_result.model_dump() if final_pseudo_response.compliance_result else None,
                           "post_processing_error": final_pseudo_response.error_details.model_dump() if final_pseudo_response.error_details else None,
                           "post_processing_duration_ms": post_duration
                      }
                 )

        except Exception as e:
             logger.error(f"Error during manager stream processing for {context.request_id}: {e}", exc_info=True)
             # Yield a final error chunk
             yield StreamChunk(chunk_id=999, request_id=context.request_id, finish_reason=FinishReason.ERROR, provider_specific_data={"error": {"code": "MANAGER_STREAM_ERROR", "message": str(e)}})


    async def _run_pre_interventions(self, request: LLMRequest) -> LLMRequest:
        """Run enabled pre-interventions sequentially."""
        current_request = request
        context = current_request.initial_context
        enabled = context.intervention_config.enabled_pre_interventions

        logger.debug(f"Running pre-interventions for {context.request_id}: {enabled}")
        for name in enabled:
            intervention = self.pre_interventions.get(name)
            if intervention:
                intervention_start = time.time()
                try:
                    logger.debug(f"Executing pre-intervention: {name}")
                    # Pass context for modification
                    current_request = await intervention.process_request(current_request, context)
                    # Update context from potentially modified request
                    context = current_request.initial_context
                    logger.debug(f"Finished pre-intervention: {name} in {(time.time() - intervention_start)*1000:.2f}ms")
                    # Check if intervention blocked the request
                    if context.intervention_data.get("block_request", False):
                         logger.info(f"Request {context.request_id} blocked by intervention: {name}")
                         break # Stop processing further pre-interventions
                except Exception as e:
                    logger.error(f"Error in pre-intervention '{name}' for {context.request_id}: {e}", exc_info=True)
                    context.intervention_data.set(f"{name}_error", str(e))
                    if not context.intervention_config.fail_open:
                        raise # Re-raise if fail_closed
            else:
                 logger.warning(f"Configured pre-intervention '{name}' not found/loaded.")

        return current_request # Return the potentially modified request

    async def _run_post_interventions(self, response: LLMResponse, context: InterventionContext) -> LLMResponse:
        """Run enabled post-interventions sequentially."""
        current_response = response
        # Use context passed in, as response.final_context might not be fully updated yet
        enabled = context.intervention_config.enabled_post_interventions

        logger.debug(f"Running post-interventions for {context.request_id}: {enabled}")
        for name in enabled:
            intervention = self.post_interventions.get(name)
            if intervention:
                intervention_start = time.time()
                try:
                    logger.debug(f"Executing post-intervention: {name}")
                    # Pass both response and context
                    current_response = await intervention.process_response(current_response, context)
                    logger.debug(f"Finished post-intervention: {name} in {(time.time() - intervention_start)*1000:.2f}ms")
                    # Check if intervention modified response significantly (e.g., blocked content)
                    if current_response.finish_reason == FinishReason.CONTENT_FILTERED and response.finish_reason != FinishReason.CONTENT_FILTERED:
                        logger.info(f"Response {context.request_id} content modified/blocked by post-intervention: {name}")
                        # Optionally break if intervention took terminal action
                        # break
                except Exception as e:
                    logger.error(f"Error in post-intervention '{name}' for {context.request_id}: {e}", exc_info=True)
                    # Add error details to the response without failing if possible
                    intervention_error = ErrorDetails(code=f"INTERVENTION_ERROR_{name.upper()}", message=str(e), level=ErrorLevel.WARNING)
                    # How to merge errors? Add to a list? Replace existing?
                    # Simplest: Store in context, or create a list in response extensions
                    context.intervention_data.set(f"{name}_error", intervention_error.model_dump())
                    if not context.intervention_config.fail_open:
                        # If fail_closed, modify response to indicate error
                        error_response_dict = current_response.model_dump()
                        error_response_dict['error_details'] = intervention_error
                        error_response_dict['finish_reason'] = FinishReason.ERROR
                        current_response = LLMResponse(**error_response_dict)
                        break # Stop further post-processing on error if fail_closed
            else:
                 logger.warning(f"Configured post-intervention '{name}' not found/loaded.")

        # Ensure the final context from interventions is attached
        final_response_dict = current_response.model_dump()
        final_response_dict['final_context'] = context
        return LLMResponse(**final_response_dict)

    def _get_provider_id_for_model(self, model_identifier: str) -> str:
        """Helper to determine which provider_id handles a given model."""
        # This needs a robust way to map model IDs to provider IDs.
        # Simplistic approach: Iterate through provider configs in GatewayConfig.
        provider_configs = getattr(self.gateway_config, 'providers', {})
        for provider_id, config in provider_configs.items():
             # Check if model_identifier is listed in this provider's config.models
             # The structure of config.models needs to be defined (e.g., just a list or dict)
             if isinstance(config.models, list) and model_identifier in config.models:
                  return provider_id
             elif isinstance(config.models, dict) and model_identifier in config.models:
                  return provider_id
        # Fallback to default provider if model not explicitly listed? Or raise error?
        default_provider = self.gateway_config.default_provider
        logger.warning(f"Model '{model_identifier}' not explicitly mapped to a provider. Falling back to default provider '{default_provider}'.")
        if not default_provider:
             raise ValueError(f"Cannot determine provider for model '{model_identifier}' and no default provider is set.")
        return default_provider

