# llm_gateway/core/factory.py

import logging
from typing import Dict, Type, Optional, List
import asyncio

# Gateway core models
from llm_gateway.core.models import ProviderConfig, GatewayConfig

# Base provider interface (assuming it exists)
from llm_gateway.providers.base import BaseProvider

# Import specific provider implementations
# It's important that these classes are imported so the factory knows about them.
from llm_gateway.providers.openai_client import OpenAIClient
from llm_gateway.providers.anthropic_client import AnthropicClient
from llm_gateway.providers.mcp_provider import MCPProvider
from llm_gateway.providers.mock_client import MockClient # Assuming a mock provider exists

logger = logging.getLogger(__name__)

# Define a custom exception for factory-specific errors
class ProviderFactoryError(Exception):
    """Custom exception for errors during provider factory operations."""
    pass

class ProviderFactory:
    """
    Factory class responsible for creating instances of LLM providers.

    This factory uses a registry to map provider type strings (from configuration)
    to the actual provider implementation classes.
    """

    def _register_known_providers(self):
        """Registers the provider classes known to this factory."""
        self.register_provider("openai", OpenAIClient)
        self.register_provider("anthropic", AnthropicClient)
        self.register_provider("mcp", MCPProvider)
        self.register_provider("mock", MockClient)
        # Add other providers here

    def register_provider(self, provider_type: str, provider_class: Type[BaseProvider]):
        """Registers a provider type string and its corresponding class."""
        type_lower = provider_type.lower()
        if type_lower in self._provider_registry and not self._allow_overwrite:
            raise ValueError(f"Provider type '{type_lower}' already registered and overwrite is disallowed.")
        elif type_lower in self._provider_registry and self._allow_overwrite:
            logger.warning(f"Provider type '{type_lower}' is already registered. Overwriting.")

        if not issubclass(provider_class, BaseProvider):
             raise TypeError(f"Provider class {provider_class.__name__} must inherit from BaseProvider.")

        self._provider_registry[type_lower] = provider_class
        logger.debug(f"Registered provider type '{type_lower}' with class {provider_class.__name__}")

    def get_registered_types(self) -> List[str]:
         """Returns a list of registered provider type strings."""
         return list(self._provider_registry.keys())

    async def get_provider(self, provider_id: str, provider_config: ProviderConfig, gateway_config: GatewayConfig) -> BaseProvider:
        """
        Gets or creates a provider instance. Ensures only one instance per provider_id.
        Includes asynchronous initialization pattern.

        Args:
            provider_id: Unique ID for the specific provider instance.
            provider_config: Configuration for this provider instance.
            gateway_config: Global gateway configuration.

        Returns:
            An initialized provider instance.

        Raises:
            ProviderFactoryError: If creation or initialization fails.
        """
        # Ensure lock exists for the provider_id
        if provider_id not in self._instance_locks:
            # This needs its own lock to prevent race conditions when creating instance locks
            # A simpler approach might be to pre-populate locks if provider_ids are known upfront
            # For dynamic providers, this lock needs careful handling.
            # Using a single lock for lock creation for simplicity here:
            async with asyncio.Lock(): # Temporary lock for creating the specific lock
                if provider_id not in self._instance_locks:
                    self._instance_locks[provider_id] = asyncio.Lock()

        async with self._instance_locks[provider_id]:
            if provider_id not in self._provider_instances:
                logger.info(f"Creating new provider instance for ID: {provider_id}")
                try:
                    instance = self._create_provider_sync( # Keep instantiation sync
                        provider_id=provider_id,
                        provider_config=provider_config,
                        gateway_config=gateway_config
                    )
                    # Perform async initialization if the method exists
                    if hasattr(instance, "initialize_async") and callable(instance.initialize_async):
                         logger.info(f"Running async initialization for provider {provider_id}...")
                         await instance.initialize_async()
                         logger.info(f"Async initialization complete for {provider_id}.")

                    self._provider_instances[provider_id] = instance
                except Exception as e:
                     # Catch errors during creation or async init
                     logger.error(f"Failed to create or initialize provider '{provider_id}': {e}", exc_info=True)
                     # Remove potentially partial instance if error occurred
                     if provider_id in self._provider_instances:
                          del self._provider_instances[provider_id]
                     raise ProviderFactoryError(f"Failed to get provider '{provider_id}': {e}") from e

            return self._provider_instances[provider_id]

    def _create_provider_sync(self, provider_id: str, provider_config: ProviderConfig, gateway_config: GatewayConfig) -> BaseProvider:
        """Synchronously instantiates a provider class."""
        if not provider_config or not hasattr(provider_config, 'provider_type'):
            raise TypeError("Invalid provider_config. Must be ProviderConfig with 'provider_type'.")

        provider_type = provider_config.provider_type
        if not provider_type or not isinstance(provider_type, str):
            raise TypeError(f"provider_config for '{provider_id}' must have a non-empty string 'provider_type'.")

        provider_type_lower = provider_type.lower()
        provider_class = self._provider_registry.get(provider_type_lower)

        if not provider_class:
            raise ProviderFactoryError(
                f"Unknown provider type '{provider_type}' for ID '{provider_id}'. "
                f"Registered: {self.get_registered_types()}"
            )

        logger.info(f"Instantiating provider ID '{provider_id}' (Type: '{provider_type}')")
        try:
            # Instantiate, passing necessary configs
            provider_instance = provider_class(
                provider_config=provider_config,
                gateway_config=gateway_config
            )
            logger.info(f"Successfully instantiated provider ID '{provider_id}'.")
            return provider_instance
        except Exception as e:
            # Log details but re-raise as factory error
            logger.error(f"Instantiation error for provider '{provider_id}': {e}", exc_info=True)
            raise ProviderFactoryError(f"Instantiation failed for '{provider_id}': {e}") from e

    def get_cached_instance(self, provider_id: str) -> Optional[BaseProvider]:
         """Gets a cached provider instance if it exists, without creating."""
         return self._provider_instances.get(provider_id)

    def get_all_cached_instances(self) -> List[BaseProvider]:
         """Gets all currently cached provider instances."""
         return list(self._provider_instances.values())

    async def cleanup_all(self):
         """Cleans up all cached provider instances."""
         logger.info(f"Cleaning up {len(self._provider_instances)} provider instance(s)...")
         tasks = []
         provider_ids = list(self._provider_instances.keys()) # Avoid dict size change during iteration
         for provider_id in provider_ids:
              instance = self._provider_instances.pop(provider_id, None)
              if instance and hasattr(instance, "cleanup") and callable(instance.cleanup):
                   logger.debug(f"Scheduling cleanup for provider {provider_id}")
                   tasks.append(instance.cleanup())
              # Remove lock too
              if provider_id in self._instance_locks:
                   del self._instance_locks[provider_id]

         if tasks:
              results = await asyncio.gather(*tasks, return_exceptions=True)
              for i, result in enumerate(results):
                   if isinstance(result, Exception):
                        # Find corresponding provider ID - relies on order
                        # A more robust way would be to wrap tasks with metadata
                        logger.error(f"Error during cleanup for a provider: {result}") # ID unknown here easily
         logger.info("Provider instance cleanup complete.")


# Example Usage (typically called by the GatewayManager)
if __name__ == "__main__":
    # Configure logging for demonstration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # --- Dummy Config Objects ---
    # Load actual configs in real usage
    gw_config = GatewayConfig(gateway_id="test-gateway", default_provider="mock")

    # Valid OpenAI config (assuming API key is in env var OPENAI_API_KEY)
    openai_provider_conf = ProviderConfig(
        provider_id="openai-gpt4",
        provider_type="openai",
        # connection_params might specify API key env var if non-default needed
    )

    # Valid Anthropic config (assuming API key is in env var ANTHROPIC_API_KEY)
    anthropic_provider_conf = ProviderConfig(
        provider_id="anthropic-claude3",
        provider_type="anthropic",
    )

    # Invalid type config
    unknown_provider_conf = ProviderConfig(
        provider_id="unknown-provider",
        provider_type="some-new-llm",
    )

    # Config causing potential init error (e.g., missing Azure details if MockClient checks)
    mock_conf_maybe_bad = ProviderConfig(
         provider_id="mock-test",
         provider_type="mock",
         connection_params={"simulate_error_on_init": True} # Assuming MockClient supports this
    )


    # --- Factory Instantiation ---
    factory = ProviderFactory()

    # --- Test Cases ---
    print("\n--- Factory Tests ---")
    created_providers: Dict[str, BaseProvider] = {}

    # 1. Create OpenAI
    try:
        print(f"\nAttempting to create provider: {openai_provider_conf.provider_id}")
        provider_openai = factory.create_provider(
            openai_provider_conf.provider_id, openai_provider_conf, gw_config
        )
        created_providers[openai_provider_conf.provider_id] = provider_openai
        print(f"Success: Created {type(provider_openai).__name__}")
    except ProviderFactoryError as e:
        print(f"Error creating OpenAI: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # 2. Create Anthropic
    try:
        print(f"\nAttempting to create provider: {anthropic_provider_conf.provider_id}")
        provider_anthropic = factory.create_provider(
            anthropic_provider_conf.provider_id, anthropic_provider_conf, gw_config
        )
        created_providers[anthropic_provider_conf.provider_id] = provider_anthropic
        print(f"Success: Created {type(provider_anthropic).__name__}")
    except ProviderFactoryError as e:
        print(f"Error creating Anthropic: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


    # 3. Create Unknown Type
    try:
        print(f"\nAttempting to create provider: {unknown_provider_conf.provider_id}")
        provider_unknown = factory.create_provider(
            unknown_provider_conf.provider_id, unknown_provider_conf, gw_config
        )
        # Should not reach here
    except ProviderFactoryError as e:
        print(f"Successfully caught expected error for unknown type: {e}")
    except Exception as e:
        print(f"Caught unexpected error for unknown type: {e}")


    # 4. Create Mock that might fail init
    try:
        print(f"\nAttempting to create provider: {mock_conf_maybe_bad.provider_id} (may fail)")
        provider_mock_fail = factory.create_provider(
            mock_conf_maybe_bad.provider_id, mock_conf_maybe_bad, gw_config
        )
        # May or may not reach here depending on MockClient implementation and config
        print(f"Success: Created {type(provider_mock_fail).__name__} (or MockClient handled simulated error internally)")
    except ProviderFactoryError as e:
        print(f"Successfully caught expected error during mock init: {e}")
    except Exception as e:
        print(f"Caught unexpected error during mock init: {e}")

    # --- Cleanup (if needed, usually done by manager) ---
    print("\n--- Cleanup ---")
    for provider_id, provider_instance in created_providers.items():
        try:
             # In a real app, manager would call cleanup
             if hasattr(provider_instance, 'cleanup') and callable(provider_instance.cleanup):
                  # await provider_instance.cleanup() # If async
                  pass
             print(f"Placeholder cleanup for {provider_id}")
        except Exception as e:
             print(f"Error during cleanup for {provider_id}: {e}")