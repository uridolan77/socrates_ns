# llm_gateway/interventions/examples.py
import time
import logging
import re
from typing import Optional, List

from llm_gateway.core.models import (
    LLMRequest, LLMResponse, StreamChunk, InterventionContext, FinishReason, ErrorDetails, ErrorLevel
)
from llm_gateway.interventions.base import BaseIntervention, InterventionHookType

logger = logging.getLogger(__name__)

class LoggingIntervention(BaseIntervention):
    name: str = "request_logger"
    hook_type: InterventionHookType = "pre_post" # Runs before and after

    def __init__(self, config=None):
        super().__init__(config)
        try:
            self.log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        except AttributeError:
            self.log_level = logging.INFO

    async def process_request(self, request: LLMRequest, context: InterventionContext) -> LLMRequest:
        logger.log(self.log_level, f"[{self.name} PRE] Request ID: {context.request_id}, Model: {request.config.model_identifier}")
        # You could log more details about the request here
        context.intervention_data.set(f"{self.name}_pre_timestamp", time.time()) # Example: store data
        return request

    async def process_response(self, response: LLMResponse, context: InterventionContext) -> LLMResponse:
        pre_time = context.intervention_data.get(f"{self.name}_pre_timestamp")
        duration = (time.time() - pre_time) * 1000 if pre_time else None
        logger.log(self.log_level, f"[{self.name} POST] Request ID: {context.request_id}, Finish: {response.finish_reason}, Duration: {duration:.2f}ms (approx)")
        # You could log details about the response here
        return response

class KeywordBlocker(BaseIntervention):
    name: str = "keyword_blocker"
    hook_type: InterventionHookType = "pre" # Only checks the request prompt

    def __init__(self, config=None):
        super().__init__(config)
        self.blocked_keywords: List[str] = self.config.get("blocked_keywords", [])
        self.case_sensitive: bool = self.config.get("case_sensitive", False)
        if not self.case_sensitive:
            self.blocked_keywords = [kw.lower() for kw in self.blocked_keywords]
        logger.info(f"[{self.name}] Initialized with {len(self.blocked_keywords)} keywords.")

    async def process_request(self, request: LLMRequest, context: InterventionContext) -> LLMRequest:
        prompt_text = ""
        if isinstance(request.prompt_content, str):
            prompt_text = request.prompt_content
        elif isinstance(request.prompt_content, list):
            # Extract text from ContentItems
            prompt_text = "\n".join([item.text_content for item in request.prompt_content if item.type == "text" and item.text_content])

        text_to_check = prompt_text if self.case_sensitive else prompt_text.lower()

        for keyword in self.blocked_keywords:
            # Use regex for word boundary matching
            pattern = f"\\b{re.escape(keyword)}\\b"
            if re.search(pattern, text_to_check):
                block_reason = f"Blocked due to keyword: '{keyword}'"
                logger.warning(f"[{self.name}] {block_reason} in request {context.request_id}")
                # Signal the manager to block the request
                context.intervention_data.set("block_request", True)
                context.intervention_data.set("block_reason", block_reason)
                # Add compliance violation details if needed
                # context.intervention_data.add_violation(...)
                break # Stop checking after first match

        return request # Return original request, block happens via context flag

# --- Example Stream Intervention (Conceptual) ---
class StreamWordCounter(BaseIntervention):
    name: str = "stream_word_counter"
    # Indicates it processes chunks and potentially a final response summary
    hook_type: InterventionHookType = "post_stream" # "stream" or "post_stream"

    async def process_stream_chunk(self, chunk: StreamChunk, context: InterventionContext) -> Optional[StreamChunk]:
        # Executed for each chunk if hook_type includes "stream"
        count = context.intervention_data.get(f"{self.name}_count", 0)
        if chunk.delta_text:
             count += len(chunk.delta_text.split())
             context.intervention_data.set(f"{self.name}_count", count)
        return chunk # Pass chunk through

    async def process_response(self, response: LLMResponse, context: InterventionContext) -> LLMResponse:
        # Executed after the stream ends if hook_type includes "post"
        final_count = context.intervention_data.get(f"{self.name}_count", 0)
        logger.info(f"[{self.name}] Final word count for {context.request_id}: {final_count}")
        # Add info to response extensions or context data for logging/auditing
        response_dict = response.model_dump()
        if 'extensions' not in response_dict: response_dict['extensions'] = {}
        if 'intervention_results' not in response_dict['extensions']: response_dict['extensions']['intervention_results'] = {}
        response_dict['extensions']['intervention_results'][self.name] = {"word_count": final_count}
        return LLMResponse(**response_dict)

