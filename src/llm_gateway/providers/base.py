# llm_gateway/providers/base.py

"""
Base provider interface for the LLM Gateway.
Defines the contract that all provider implementations must follow.
"""

import abc
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any

from llm_gateway.core.models import (
    LLMRequest,
    LLMResponse,
    ProviderConfig,
    StreamChunk,
)

logger = logging.getLogger(__name__)

class BaseProvider(abc.ABC):
    """
    Abstract base class for all LLM Gateway provider implementations.
    
    A provider is responsible for connecting to a specific LLM service,
    sending requests, and processing responses according to the gateway's
    standardized formats.
    """

    def __init__(self, provider_config: ProviderConfig):
        """
        Initialize the provider with its configuration.
        
        Args:
            provider_config: Configuration specific to this provider instance
        """
        self.provider_config = provider_config
        self.provider_id = provider_config.provider_id
        self.created_at = datetime.utcnow()
        
        logger.info(f"Initializing {self.__class__.__name__} provider with ID: {self.provider_id}")
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """
        Initialize the client for this provider.
        
        This method should handle any setup required before making API calls,
        like creating API clients, loading credentials, etc.
        
        Each provider implementation should override this method with its
        specific initialization logic.
        """
        pass
    
    @abc.abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for the given request.
        
        This is the main method that processes a request and returns a complete response.
        
        Args:
            request: The request containing prompt, conversation history, and config
            
        Returns:
            A complete response with generated content or error details
        """
        pass
    
    @abc.abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response for the given request.
        
        This method returns an async generator that yields chunks of the response
        as they become available.
        
        Args:
            request: The request containing prompt, conversation history, and config
            
        Yields:
            Chunks of the generated response
        """
        pass
    
    @abc.abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up resources used by this provider.
        
        This method should be called when the provider is no longer needed
        to ensure proper resource cleanup.
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the provider is healthy and available.
        
        Returns:
            A dictionary with health check information
        """
        # Default implementation that can be overridden by providers
        # that support health checks
        return {
            "provider_id": self.provider_id,
            "status": "available",
            "created_at": self.created_at.isoformat(),
            "message": "Provider health check not implemented"
        }
    
    def get_model_info(self, model_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model supported by this provider.
        
        Args:
            model_identifier: The identifier of the model
            
        Returns:
            Model information or None if the model is not supported
        """
        # Look up the model in the provider config
        if model_identifier in self.provider_config.models:
            return self.provider_config.models[model_identifier]
        return None
    
    @property
    def supported_models(self) -> Dict[str, Any]:
        """Get all models supported by this provider."""
        return self.provider_config.models