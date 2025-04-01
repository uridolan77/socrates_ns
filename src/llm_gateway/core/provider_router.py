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
import logging
from typing import Dict, List, Optional, Type, Any, Union
from typing import Dict, List, Optional, Type, AsyncGenerator
logger = logging.getLogger(__name__)

class ProviderRouter:
    """Routes requests to appropriate providers with load balancing and failover."""
    
    def __init__(self, provider_configs, gateway_config):
        self.providers = {}  # Provider pools by type
        self.routing_strategy = gateway_config.routing_strategy  # "round-robin", "least-used", etc
        self.failover_enabled = gateway_config.failover_enabled
        self.provider_factory = ProviderFactory()
        self._setup_provider_pools(provider_configs)
        self._usage_stats = {}  # Track usage for load balancing
        
    async def route_request(self, request: LLMRequest) -> Union[LLMResponse, AsyncGenerator[StreamChunk, None]]:
        """Route request to appropriate provider with failover support"""
        primary_provider = self._select_provider(request)
        try:
            if request.stream:
                return await primary_provider.generate_stream(request)
            else:
                return await primary_provider.generate(request)
        except Exception as e:
            if self.failover_enabled:
                fallback_provider = self._select_fallback_provider(request, primary_provider, e)
                if fallback_provider:
                    logger.warning(f"Failing over from {primary_provider.provider_id} to {fallback_provider.provider_id}")
                    if request.stream:
                        return await fallback_provider.generate_stream(request)
                    else:
                        return await fallback_provider.generate(request)
            raise  # Re-raise if no fallback or fallback failed
