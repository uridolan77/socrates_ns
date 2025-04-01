# llm_gateway/interventions/base.py

import abc
import logging
from typing import AsyncGenerator, Optional, Any, Dict, Literal

from llm_gateway.core.models import (
    LLMRequest,
    LLMResponse,
    StreamChunk,
    InterventionContext
)

logger = logging.getLogger(__name__)

# Define possible hook points
InterventionHookType = Literal["pre", "post", "stream", "pre_post", "pre_stream", "post_stream", "pre_post_stream"]

class BaseIntervention(abc.ABC):
    """
    Abstract base class for all interventions in the LLM Gateway pipeline.

    Interventions can inspect and modify requests/responses, interact with
    external systems, enforce policies, and add context.
    """

    # Class attributes to be overridden by subclasses
    # A unique identifier for the intervention type
    name: str = "base_intervention"
    # Specifies when the intervention runs: "pre", "post", "stream", or combinations
    hook_type: InterventionHookType = "pre" # Default, override as needed

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the intervention.

        Args:
            config: Dictionary containing configuration specific to this intervention,
                    loaded from the gateway configuration.
        """
        self.config = config or {}
        logger.info(f"Initialized intervention: {self.name} (Type: {self.hook_type})")

    async def initialize_async(self):
        """
        Optional asynchronous initialization for interventions that need
        to perform async setup (e.g., load models, connect to services).
        Called by the factory after instantiation.
        """
        pass # Override if needed

    async def process_request(self, request: LLMRequest, context: InterventionContext) -> LLMRequest:
        """
        Process an incoming LLMRequest before it's sent to the provider.
        Only called if hook_type includes "pre".

        Args:
            request: The incoming LLM request.
            context: The mutable intervention context associated with the request.
                     Interventions can read/write data here.

        Returns:
            The potentially modified LLMRequest. Interventions can modify the
            request prompt, configuration, or add data to the context.
            Returning the original request object means no modification.
            To block the request, set `context.intervention_data["block_request"] = True`
            and optionally add a reason.
        """
        # Default implementation: No change
        return request

    async def process_response(self, response: LLMResponse, context: InterventionContext) -> LLMResponse:
        """
        Process an outgoing LLMResponse after it's received from the provider.
        Only called if hook_type includes "post".

        Args:
            response: The LLM response received from the provider.
            context: The mutable intervention context.

        Returns:
            The potentially modified LLMResponse. Interventions can modify the
            generated content, add compliance results, trigger alerts, or filter
            the response. Returning the original response object means no modification.
        """
        # Default implementation: No change
        return response

    async def process_stream_chunk(self, chunk: StreamChunk, context: InterventionContext) -> Optional[StreamChunk]:
        """
        Process a single chunk from a streaming response.
        Only called if hook_type includes "stream".

        Args:
            chunk: The stream chunk received from the provider.
            context: The mutable intervention context.

        Returns:
            The potentially modified StreamChunk, or None to filter/drop the chunk.
            Returning the original chunk object means no modification.
            Note: Modifying finish_reason or usage_update in intermediate chunks
                  might have unintended consequences. Primarily focus on delta_text,
                  delta_content_items, or provider_specific_data.
        """
        # Default implementation: No change, pass chunk through
        return chunk

    async def cleanup(self) -> None:
        """
        Clean up any resources used by the intervention.
        Called when the gateway shuts down or the intervention is unloaded.
        """
        # Default implementation: No cleanup needed
        pass
