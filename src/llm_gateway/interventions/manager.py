# llm_gateway/core/manager.py

import asyncio
import logging
import time
import copy
from typing import Dict, List, Optional, Type, AsyncGenerator, Any, Union, cast, Tuple

from llm_gateway.core.models import (
    InterventionContext, LLMRequest, LLMResponse, StreamChunk, ContentItem,
    PerformanceMetrics, ComplianceResult, FinishReason, ErrorDetails, ErrorLevel,
    GatewayConfig, ProviderConfig, RoutingError, UsageStats, ToolUseRequest # Import ProviderConfig
)
from llm_gateway.core.factory import ProviderFactory, ProviderFactoryError
from llm_gateway.interventions.factory import InterventionFactory, InterventionFactoryError
from llm_gateway.interventions.base import BaseIntervention, InterventionHookType # Import HookType
from llm_gateway.providers.base import BaseProvider # Import BaseProvider

logger = logging.getLogger(__name__)

# Custom Exception for Intervention Errors
class InterventionExecutionError(Exception):
    """Raised when an intervention fails in fail_closed mode."""
    pass

class InterventionManager:
    """
    Manages the intervention pipeline and provider routing.
    Handles requests, streaming, interventions, provider selection, and failover.
    """

    def __init__(self, provider_factory: ProviderFactory, gateway_config: GatewayConfig):
        """Initialize the intervention manager."""
        self.provider_factory = provider_factory
        self.gateway_config = gateway_config
        self.intervention_factory = InterventionFactory(gateway_config)
        # Cache intervention instances managed by the factory
        # self._intervention_instances: Dict[str, BaseIntervention] = {} # Managed by factory now
        # self._instance_lock = asyncio.Lock() # Managed by factory now
        logger.info("InterventionManager initialized.")

    # Removed _get_intervention_instance, rely on factory's get_intervention

    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process a request through interventions, routing, and provider call."""
        start_time = time.time()
        context = copy.deepcopy(request.initial_context)
        response: Optional[LLMResponse] = None
        pre_duration: Optional[float] = None
        routing_duration: Optional[float] = None
        generation_duration: Optional[float] = None
        post_duration: Optional[float] = None
        final_error: Optional[ErrorDetails] = None
        selected_provider_id: Optional[str] = None
        failover_attempted = False

        try:
            # --- 1. Pre-processing Interventions ---
            pre_start = time.time()
            modified_request = await self._run_interventions_at_hook(request, context, "pre")
            pre_duration = (time.time() - pre_start) * 1000
            logger.debug(f"Pre-processing complete for {context.request_id} in {pre_duration:.2f}ms")

            # Check for block
            if context.intervention_data.get("block_request", False):
                # ... (keep block handling code as before) ...
                block_reason = context.intervention_data.get('block_reason', 'Blocked by pre-processing intervention')
                logger.warning(f"Request {context.request_id} blocked by pre-intervention: {block_reason}")
                final_error = ErrorDetails(code="INTERVENTION_BLOCK", message=block_reason, level=ErrorLevel.WARNING, stage="pre_intervention")
                return LLMResponse()

            # --- 2. Provider Routing & Selection ---
            routing_start = time.time()
            try:
                 provider, selected_provider_id = await self._select_provider_with_failover(modified_request, context)
            except RoutingError as e:
                 logger.error(f"Routing failed for {context.request_id}: {e}", exc_info=True)
                 raise # Let outer handler catch routing failure
            finally:
                 routing_duration = (time.time() - routing_start) * 1000
            logger.debug(f"Routing complete for {context.request_id}. Selected provider: {selected_provider_id} in {routing_duration:.2f}ms")


            # --- 3. Provider Call ---
            generation_start = time.time()
            logger.debug(f"Sending request {context.request_id} to provider {selected_provider_id}")
            response = await provider.generate(modified_request)
            generation_duration = (time.time() - generation_start) * 1000
            logger.debug(f"Received response from {selected_provider_id} for {context.request_id} in {generation_duration:.2f}ms")

            # Handle provider errors before post-processing
            if response and response.error_details:
                 final_error = response.error_details
                 # Check if failover should have happened but didn't (e.g., retryable error)
                 # This logic might be complex depending on retry implementation within provider vs manager
                 logger.warning(f"Provider {selected_provider_id} returned error for {context.request_id}: {final_error.code}")
                 # Potentially trigger failover *here* if not handled by _select_provider_with_failover
                 # For simplicity, assume _select_provider handles initial failover attempt

            # --- 4. Post-processing Interventions ---
            post_start = time.time()
            if response and not final_error: # Only run post if no provider or manager error yet
                 response = await self._run_interventions_at_hook(response, context, "post")
                 post_duration = (time.time() - post_start) * 1000
                 logger.debug(f"Post-processing complete for {context.request_id} in {post_duration:.2f}ms")
                 # Check if post-intervention caused an error or blocked
                 if response.error_details and not final_error:
                      final_error = response.error_details
                      logger.warning(f"Post-intervention error for {context.request_id}: {final_error.code}")
                 elif response.finish_reason == FinishReason.CONTENT_FILTERED and \
                      (not final_error or final_error.code != "INTERVENTION_BLOCK"): # Check if *newly* filtered
                      logger.warning(f"Response {context.request_id} filtered by post-intervention.")
                      # Optionally set an error detail for post-intervention filtering
                      final_error = ErrorDetails(code="INTERVENTION_FILTER", message="Content filtered by post-processing intervention", level=ErrorLevel.WARNING, stage="post_intervention")

            elif not response and not final_error: # Provider returned None, no prior error
                 logger.error(f"Provider {selected_provider_id} returned None response without error for {context.request_id}")
                 final_error = ErrorDetails(code="PROVIDER_EMPTY_RESPONSE", message="Provider returned None without error.", level=ErrorLevel.ERROR, stage="provider_call")
                 post_duration = 0.0
            else: # Provider error occurred, skip post-processing
                 post_duration = 0.0


        except InterventionExecutionError as e: # Catch fail_closed errors from interventions
             logger.error(f"Fail-closed intervention error for {context.request_id}: {e}", exc_info=True)
             final_error = ErrorDetails(code="INTERVENTION_FAILURE", message=f"Intervention failed (fail_closed=True): {str(e)}", level=ErrorLevel.ERROR, stage=e.args[1] if len(e.args)>1 else "unknown_intervention") # Pass stage if possible
        except RoutingError as e: # Catch routing errors
             logger.error(f"Routing/Failover failed for {context.request_id}: {e}", exc_info=True)
             final_error = ErrorDetails(code="ROUTING_FAILURE", message=str(e), level=ErrorLevel.ERROR, stage="manager")
        except Exception as e: # Catch other unexpected manager errors
             logger.error(f"Critical error during manager processing for {context.request_id}: {e}", exc_info=True)
             final_error = ErrorDetails(code="MANAGER_PROCESSING_ERROR", message=f"Unhandled exception: {str(e)}", level=ErrorLevel.CRITICAL, stage="manager")

        # --- 5. Finalize Metrics and Response ---
        total_duration = (time.time() - start_time) * 1000

        # Ensure response exists, creating error response if needed
        if response is None:
            response = LLMResponse(request_id=context.request_id, final_context=context, error_details=final_error or ErrorDetails(code="UNKNOWN_FAILURE"), finish_reason=FinishReason.ERROR)

        # Consolidate performance metrics
        llm_latency = response.performance_metrics.llm_latency_ms if response.performance_metrics else None
        compliance_duration = context.intervention_data.get("compliance_check_duration_ms") # Get from context if set by intervention
        final_perf = PerformanceMetrics(
            total_duration_ms=total_duration,
            llm_latency_ms=llm_latency,
            pre_processing_duration_ms=pre_duration,
            routing_duration_ms=routing_duration, # Added routing duration
            post_processing_duration_ms=post_duration,
            compliance_check_duration_ms=compliance_duration,
            # gateway_overhead_ms calculated by model validator
        )

        # Rebuild final response ensuring context, metrics, and errors are correct
        final_response_dict = response.model_dump(exclude={'performance_metrics', 'final_context', 'error_details'})
        final_response_dict['performance_metrics'] = final_perf
        final_response_dict['final_context'] = context # Attach the final context state
        final_response_dict['error_details'] = final_error if final_error else response.error_details # Prioritize manager/routing/intervention errors

        # Ensure finish reason reflects error state
        if final_error and final_error.level in [ErrorLevel.ERROR, ErrorLevel.CRITICAL]:
            final_response_dict['finish_reason'] = FinishReason.ERROR
        elif final_response_dict.get('finish_reason') is None: # Ensure finish_reason is always set
             final_response_dict['finish_reason'] = FinishReason.UNKNOWN

        return LLMResponse(**final_response_dict)


    async def process_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """Process streaming requests with interventions, routing, and failover (conceptual)."""
        start_time = time.time()
        context = copy.deepcopy(request.initial_context)
        request_id = context.request_id
        logger.info(f"Manager processing stream: {request_id}")
        pre_duration: Optional[float] = None
        post_duration: Optional[float] = None
        final_response_summary: Optional[LLMResponse] = None
        selected_provider_id: Optional[str] = None
        provider: Optional[BaseProvider] = None
        stream_interventions: List[BaseIntervention] = []

        try:
            # --- 1. Pre-processing ---
            pre_start = time.time()
            modified_request = await self._run_interventions_at_hook(request, context, "pre")
            pre_duration = (time.time() - pre_start) * 1000
            logger.debug(f"Stream pre-processing done for {request_id} in {pre_duration:.2f}ms")

            # Check for block
            if context.intervention_data.get("block_request", False):
                # ... (yield block error chunk as before) ...
                block_reason = context.intervention_data.get('block_reason', 'Blocked by pre-processing intervention')
                logger.warning(f"Stream request {request_id} blocked by pre-intervention: {block_reason}")
                yield self._create_manager_error_chunk(request_id, 0, "INTERVENTION_BLOCK", block_reason, ErrorLevel.WARNING, "pre_intervention")
                return

            # --- 2. Provider Routing & Selection ---
            routing_start = time.time()
            try:
                 # For streams, failover is trickier. Select primary initially.
                 provider, selected_provider_id = await self._select_provider_with_failover(modified_request, context, attempt_failover=False) # Don't failover yet
            except RoutingError as e:
                 logger.error(f"Initial routing failed for stream {request_id}: {e}")
                 yield self._create_manager_error_chunk(request_id, 0, "ROUTING_FAILURE", str(e), ErrorLevel.ERROR, "manager")
                 return
            finally:
                 routing_duration = (time.time() - routing_start) * 1000
            logger.debug(f"Stream routing complete for {request_id}. Selected provider: {selected_provider_id}")

            # --- 3. Prepare Stream Interventions ---
            enabled_stream_interventions = self._get_enabled_interventions(context, "stream")
            if enabled_stream_interventions:
                 stream_interventions = [
                     inst for name in enabled_stream_interventions
                     if (inst := await self.intervention_factory.get_intervention(name)) # Use factory cache
                 ]
                 logger.debug(f"Enabled stream interventions for {request_id}: {[i.name for i in stream_interventions]}")


            # --- 4. Stream Processing Loop ---
            aggregated_chunks: List[StreamChunk] = []
            final_finish_reason: Optional[FinishReason] = None
            final_usage: Optional[UsageStats] = None
            final_tool_calls: Optional[List[ToolUseRequest]] = None
            chunk_index = 0
            provider_stream_failed = False

            try:
                async for chunk in provider.generate_stream(modified_request):
                    current_chunk: Optional[StreamChunk] = chunk

                    # Apply chunk-level interventions
                    for intervention in stream_interventions:
                        if current_chunk is None: break # Stop if chunk was filtered by previous intervention
                        try:
                             current_chunk = await intervention.process_stream_chunk(current_chunk, context)
                        except Exception as e:
                             logger.error(f"Error in stream intervention '{intervention.name}' for chunk {chunk.chunk_id} of {request_id}: {e}", exc_info=True)
                             context.intervention_data.set(f"{intervention.name}_stream_error", str(e))
                             if not context.intervention_config.fail_open:
                                  # Yield specific intervention error chunk and stop
                                  yield self._create_manager_error_chunk(request_id, chunk.chunk_id, f"INTERVENTION_STREAM_ERROR_{intervention.name.upper()}", str(e), ErrorLevel.ERROR, "stream_intervention")
                                  return # Stop stream processing

                    # Yield the processed (or original, or None) chunk
                    if current_chunk is not None:
                        # Add chunk index if missing (should be set by provider ideally)
                        if current_chunk.chunk_id is None:
                             current_chunk = current_chunk.model_copy(update={"chunk_id": chunk_index})

                        aggregated_chunks.append(current_chunk) # Store chunk for final processing
                        yield current_chunk
                        chunk_index += 1

                        # Update final state from the last valid chunk
                        if current_chunk.finish_reason: final_finish_reason = current_chunk.finish_reason
                        if current_chunk.usage_update: final_usage = current_chunk.usage_update
                        if current_chunk.delta_tool_calls: final_tool_calls = current_chunk.delta_tool_calls # Overwrite with latest

            except Exception as provider_stream_error:
                 # Error occurred during provider streaming
                 logger.error(f"Error during provider stream for {request_id} from {selected_provider_id}: {provider_stream_error}", exc_info=True)
                 provider_stream_failed = True
                 # Attempt failover if enabled
                 if self.gateway_config.failover_enabled:
                      logger.warning(f"Attempting stream failover for {request_id} due to provider error.")
                      try:
                           fallback_provider, fallback_provider_id = await self._select_provider_with_failover(
                               modified_request, context, failed_provider_id=selected_provider_id
                           )
                           logger.info(f"Failing over stream {request_id} to {fallback_provider_id}")
                           # Re-run the stream loop with the fallback provider
                           # Reset aggregation variables
                           aggregated_chunks = []
                           final_finish_reason = None
                           final_usage = None
                           final_tool_calls = None
                           chunk_index = 0 # Reset chunk index for fallback stream
                           provider_stream_failed = False # Reset failure flag

                           async for chunk in fallback_provider.generate_stream(modified_request):
                                # Re-apply chunk interventions for fallback stream
                                current_chunk = chunk
                                for intervention in stream_interventions:
                                     if current_chunk is None: break
                                     try: current_chunk = await intervention.process_stream_chunk(current_chunk, context)
                                     except Exception as e_int_fb:
                                          logger.error(f"Error in intervention '{intervention.name}' during fallback stream for {request_id}: {e_int_fb}")
                                          if not context.intervention_config.fail_open:
                                               yield self._create_manager_error_chunk(request_id, chunk.chunk_id, f"INTERVENTION_STREAM_ERROR_{intervention.name.upper()}", str(e_int_fb), ErrorLevel.ERROR, "stream_intervention")
                                               return

                                if current_chunk is not None:
                                     if current_chunk.chunk_id is None: current_chunk = current_chunk.model_copy(update={"chunk_id": chunk_index})
                                     aggregated_chunks.append(current_chunk)
                                     yield current_chunk
                                     chunk_index += 1
                                     if current_chunk.finish_reason: final_finish_reason = current_chunk.finish_reason
                                     if current_chunk.usage_update: final_usage = current_chunk.usage_update
                                     if current_chunk.delta_tool_calls: final_tool_calls = current_chunk.delta_tool_calls

                      except RoutingError as failover_route_error:
                           logger.error(f"Stream failover failed for {request_id}: No suitable fallback provider found. {failover_route_error}")
                           yield self._create_manager_error_chunk(request_id, chunk_index, "ROUTING_FAILOVER_FAILURE", str(failover_route_error), ErrorLevel.ERROR, "manager")
                           return
                      except Exception as failover_provider_error:
                           logger.error(f"Error during fallback provider stream for {request_id}: {failover_provider_error}", exc_info=True)
                           provider_stream_failed = True # Mark as failed again
                 else:
                      logger.warning(f"Failover disabled, stream {request_id} failed.")

                 # If still failed after failover attempt or if failover disabled
                 if provider_stream_failed:
                      yield self._create_manager_error_chunk(request_id, chunk_index, "PROVIDER_STREAM_ERROR", str(provider_stream_error), ErrorLevel.ERROR, "provider_call")
                      return # Stop processing


            # --- After Stream Ends (Successfully or after failover) ---
            logger.debug(f"Stream loop finished for {request_id}. Aggregating for post-processing.")
            # ... (Reconstruct pseudo-response logic as before) ...
            full_text = "".join([c.delta_text for c in aggregated_chunks if c.delta_text])
            full_content_items = [] # ... aggregate delta_content_items ...
            final_aggregated_content = full_content_items or full_text or None

            final_response_summary = LLMResponse(
                 request_id=request_id,
                 generated_content=final_aggregated_content,
                 finish_reason=final_finish_reason or FinishReason.UNKNOWN,
                 usage=final_usage,
                 tool_use_requests=final_tool_calls,
                 final_context=context,
                 performance_metrics=PerformanceMetrics(pre_processing_duration_ms=pre_duration)
            )

            # 5. Post-processing Interventions (on the aggregated summary)
            post_start = time.time()
            final_response_summary = await self._run_interventions_at_hook(final_response_summary, context, "post")
            post_duration = (time.time() - post_start) * 1000
            logger.debug(f"Stream post-processing done for {request_id} in {post_duration:.2f}ms")

            # Yield a final summary chunk IF post-processing added value (e.g., compliance)
            # OR if an error occurred during post-processing
            yield_summary_chunk = False
            summary_chunk_data = {
                 "post_processing_duration_ms": post_duration
            }
            final_summary_finish_reason = final_response_summary.finish_reason

            if final_response_summary.compliance_result:
                 summary_chunk_data["compliance_result"] = final_response_summary.compliance_result.model_dump()
                 yield_summary_chunk = True
            if final_response_summary.error_details:
                 summary_chunk_data["error"] = final_response_summary.error_details.model_dump()
                 final_summary_finish_reason = FinishReason.ERROR # Ensure error finish reason
                 yield_summary_chunk = True

            if yield_summary_chunk:
                 yield StreamChunk(
                      chunk_id=chunk_index, # Next index
                      request_id=request_id,
                      finish_reason=final_summary_finish_reason,
                      usage_update=final_response_summary.usage, # Include final usage if available
                      provider_specific_data=summary_chunk_data
                 )

        except InterventionExecutionError as e: # Catch fail-closed from pre or stream interventions
             logger.error(f"Fail-closed intervention error during stream {request_id}: {e}", exc_info=True)
             yield self._create_manager_error_chunk(request_id, 998, "INTERVENTION_FAILURE", str(e), ErrorLevel.ERROR, e.args[1] if len(e.args)>1 else "unknown_intervention")
        except Exception as e: # Catch other unexpected manager errors during stream setup/post-processing
             logger.error(f"Critical error during manager stream processing for {request_id}: {e}", exc_info=True)
             yield self._create_manager_error_chunk(request_id, 999, "MANAGER_STREAM_ERROR", f"Unhandled exception: {str(e)}", ErrorLevel.CRITICAL, "manager")
        finally:
             total_duration = (time.time() - start_time) * 1000
             logger.info(f"Stream processing finished for {request_id}. Total duration: {total_duration:.2f}ms")


    # Consolidated Intervention Runner
    async def _run_interventions_at_hook(
        self,
        data: Union[LLMRequest, LLMResponse],
        context: InterventionContext,
        hook: InterventionHookType
    ) -> Union[LLMRequest, LLMResponse]:
        """Runs all enabled interventions for a specific hook point."""
        current_data = data
        enabled_names = self._get_enabled_interventions(context, hook)
        if not enabled_names:
            return current_data # No interventions for this hook

        logger.debug(f"Running {hook} interventions for {context.request_id}: {enabled_names}")
        for name in enabled_names:
            intervention = await self.intervention_factory.get_intervention(name) # Use factory cache
            if not intervention:
                 logger.warning(f"Intervention '{name}' configured for hook '{hook}' but could not be loaded.")
                 continue

            # Check if intervention supports the current hook
            if hook not in intervention.hook_type: continue

            intervention_start = time.time()
            try:
                logger.debug(f"Executing {hook} intervention: {name}")
                if hook == "pre":
                     current_data = await intervention.process_request(cast(LLMRequest, current_data), context)
                elif hook == "post":
                     current_data = await intervention.process_response(cast(LLMResponse, current_data), context)
                # Note: 'stream' hook is handled directly in process_stream loop

                duration_ms = (time.time() - intervention_start) * 1000
                logger.debug(f"Finished {hook} intervention: {name} in {duration_ms:.2f}ms")
                context.intervention_data.set(f"{name}_{hook}_duration_ms", duration_ms) # Store duration

                # Check for blocking flags set by interventions (relevant for 'pre')
                if hook == "pre" and context.intervention_data.get("block_request", False):
                     logger.info(f"Request {context.request_id} blocked by intervention: {name}")
                     break # Stop further processing for this hook

                # Check for errors set by interventions (relevant for 'post')
                if hook == "post" and isinstance(current_data, LLMResponse) and current_data.error_details:
                     if not context.intervention_config.fail_open:
                          logger.warning(f"Post-intervention '{name}' resulted in error (fail_closed=True). Stopping post-processing.")
                          break # Stop further post-interventions

            except Exception as e:
                logger.error(f"Error in {hook} intervention '{name}' for {context.request_id}: {e}", exc_info=True)
                context.intervention_data.set(f"{name}_{hook}_error", str(e))
                if not context.intervention_config.fail_open:
                     # Raise specific error indicating stage
                     raise InterventionExecutionError(f"{hook.capitalize()}-intervention '{name}' failed (fail_closed=True)", hook) from e

        return current_data


    # --- Provider Selection & Routing ---

    async def _select_provider_with_failover(
        self,
        request: LLMRequest,
        context: InterventionContext,
        attempt_failover: bool = True,
        failed_provider_id: Optional[str] = None
    ) -> Tuple[BaseProvider, str]:
        """
        Selects a provider based on routing strategy and attempts failover if enabled.

        Args:
            request: The potentially modified LLMRequest.
            context: The intervention context.
            attempt_failover: Whether failover logic should be considered in this call.
            failed_provider_id: The ID of the provider that just failed (if applicable).

        Returns:
            A tuple containing the selected BaseProvider instance and its provider_id.

        Raises:
            RoutingError: If no suitable provider can be found or configured.
        """
        provider_id = self._get_provider_id_for_model(request.config.model_identifier, failed_provider_id)
        provider_config = self._get_provider_config(provider_id)

        try:
            provider = await self.provider_factory.get_provider(provider_id, provider_config, self.gateway_config)
            # Optional: Perform a quick health check before returning? Could add latency.
            # health = await provider.health_check()
            # if health.get("status") != "available":
            #     raise ConnectionError(f"Selected provider {provider_id} is unhealthy: {health.get('message')}")
            return provider, provider_id
        except (ProviderFactoryError, ValueError, ConnectionError) as initial_error:
             logger.warning(f"Failed to get initial provider '{provider_id}': {initial_error}")
             if attempt_failover and self.gateway_config.failover_enabled:
                  logger.info(f"Attempting failover for request {context.request_id}...")
                  try:
                       # Try selecting again, explicitly excluding the failed provider
                       fallback_provider_id = self._get_provider_id_for_model(
                           request.config.model_identifier, exclude_provider_id=provider_id
                       )
                       fallback_config = self._get_provider_config(fallback_provider_id)
                       fallback_provider = await self.provider_factory.get_provider(
                           fallback_provider_id, fallback_config, self.gateway_config
                       )
                       logger.info(f"Failover successful. Selected fallback provider: {fallback_provider_id}")
                       return fallback_provider, fallback_provider_id
                  except (RoutingError, ProviderFactoryError, ValueError, ConnectionError) as failover_error:
                       logger.error(f"Failover failed for request {context.request_id}: {failover_error}")
                       raise RoutingError(f"Initial provider '{provider_id}' failed and no suitable fallback found.") from failover_error
             else:
                  # Failover disabled or initial attempt failed without trying failover
                  raise RoutingError(f"Failed to get required provider '{provider_id}'.") from initial_error


    def _get_provider_config(self, provider_id: str) -> ProviderConfig:
         """Safely retrieves the ProviderConfig for a given provider_id."""
         provider_configs = self.gateway_config.providers # Use structured field
         config_data = provider_configs.get(provider_id)
         if not config_data:
              raise ValueError(f"Configuration not found for provider_id: '{provider_id}'")
         # Assuming config_data is already a ProviderConfig object due to pydantic validation
         # If it's still a dict, parse it: return ProviderConfig(**config_data)
         return config_data


    def _get_provider_id_for_model(self, model_identifier: str, exclude_provider_id: Optional[str] = None) -> str:
        """
        Selects a provider ID based on model, strategy, and potential exclusions.

        Args:
            model_identifier: The requested model.
            exclude_provider_id: A provider ID to exclude from selection (used for failover).

        Returns:
            The selected provider_id.

        Raises:
            RoutingError: If no suitable provider is found.
        """
        # 1. Check explicit mapping
        model_mapping = self.gateway_config.model_provider_mapping
        if model_identifier in model_mapping:
             explicit_provider_id = model_mapping[model_identifier]
             if explicit_provider_id != exclude_provider_id:
                  logger.debug(f"Routing model '{model_identifier}' to provider '{explicit_provider_id}' via explicit mapping.")
                  # Verify this provider is configured
                  if explicit_provider_id not in self.gateway_config.providers:
                      raise RoutingError(f"Provider '{explicit_provider_id}' mapped for model '{model_identifier}' but not defined in config.")
                  return explicit_provider_id
             else:
                  logger.warning(f"Explicit provider '{exclude_provider_id}' for model '{model_identifier}' is excluded. Looking for alternatives.")


        # 2. Find all configured providers supporting the model
        candidate_providers: List[str] = []
        for provider_id, config in self.gateway_config.providers.items():
            if provider_id == exclude_provider_id:
                continue # Skip excluded provider
            # Check models list/dict within provider config
            provider_models = config.models
            if isinstance(provider_models, list) and model_identifier in provider_models:
                 candidate_providers.append(provider_id)
            elif isinstance(provider_models, dict) and model_identifier in provider_models:
                 candidate_providers.append(provider_id)

        if not candidate_providers:
             # 3. Fallback to Default Provider (if not excluded)
             default_provider = self.gateway_config.default_provider
             if default_provider and default_provider != exclude_provider_id:
                  logger.warning(f"Model '{model_identifier}' not explicitly supported by any provider (excluding '{exclude_provider_id}'). Falling back to default '{default_provider}'.")
                  # Verify default provider is configured
                  if default_provider not in self.gateway_config.providers:
                      raise RoutingError(f"Default provider '{default_provider}' not defined in config.")
                  return default_provider
             else:
                  raise RoutingError(f"No configured provider found for model '{model_identifier}' (excluding '{exclude_provider_id}').")


        # 4. Apply Routing Strategy (if multiple candidates)
        if len(candidate_providers) == 1:
             selected_id = candidate_providers[0]
             logger.debug(f"Routing model '{model_identifier}' to only available candidate: '{selected_id}'.")
             return selected_id
        else:
             # --- Implement different strategies ---
             strategy = self.gateway_config.routing_strategy
             selected_id: Optional[str] = None

             if strategy == "round_robin":
                 # Simple round robin - needs state management (e.g., a counter per model)
                 # For simplicity, just pick the first candidate for now
                 selected_id = candidate_providers[0]
                 logger.debug(f"Routing model '{model_identifier}' via round_robin (simple): Selecting '{selected_id}'.")

             # Add other strategies: "least_connections", "latency_based", "random"
             # elif strategy == "random":
             #     import random
             #     selected_id = random.choice(candidate_providers)

             else: # Default or unknown strategy
                 selected_id = candidate_providers[0]
                 logger.debug(f"Routing model '{model_identifier}' via default strategy: Selecting '{selected_id}'.")

             if selected_id is None: # Should not happen with current logic, but defensive
                 raise RoutingError(f"Routing strategy '{strategy}' failed to select a provider for '{model_identifier}'.")

             return selected_id


    def _create_manager_error_chunk(
        self, request_id: str, chunk_id: int, code: str, message: str,
        level: ErrorLevel = ErrorLevel.ERROR, stage: Optional[str] = None
    ) -> StreamChunk:
         """Creates a stream chunk representing an error originating from the manager."""
         error_details = ErrorDetails(code=code, message=message, level=level, stage=stage)
         return StreamChunk(
              chunk_id=chunk_id,
              request_id=request_id,
              finish_reason=FinishReason.ERROR,
              provider_specific_data={"error": error_details.model_dump()}
         )
