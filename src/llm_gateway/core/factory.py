# llm_gateway/core/factory.py

import logging
from typing import Dict, Type, Optional, List

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

    def __init__(self):
        """Initializes the ProviderFactory and registers known provider types."""
        self._provider_registry: Dict[str, Type[BaseProvider]] = {}
        self._register_known_providers()
        logger.info(f"ProviderFactory initialized with registered types: {list(self._provider_registry.keys())}")

    def _register_known_providers(self):
        """Registers the provider classes known to this factory."""
        # Register providers using a unique type string (e.g., matching config)
        self.register_provider("openai", OpenAIClient)
        self.register_provider("anthropic", AnthropicClient)
        self.register_provider("mcp", MCPProvider)
        self.register_provider("mock", MockClient)
        # Add other providers here as they are implemented

    def register_provider(self, provider_type: str, provider_class: Type[BaseProvider]):
        """
        Registers a provider type string and its corresponding class.

        Args:
            provider_type: The unique string identifier for the provider type (e.g., 'openai').
            provider_class: The class implementing the BaseProvider interface.

        Raises:
            ValueError: If the provider_type is already registered.
        """
        type_lower = provider_type.lower()
        if type_lower in self._provider_registry:
            logger.warning(f"Provider type '{type_lower}' is already registered. Overwriting is not allowed by default.")
            # Or raise ValueError(f"Provider type '{type_lower}' already registered.")
            return # Avoid overwriting silently unless intended

        if not issubclass(provider_class, BaseProvider):
             raise TypeError(f"Provider class {provider_class.__name__} must inherit from BaseProvider.")

        self._provider_registry[type_lower] = provider_class
        logger.debug(f"Registered provider type '{type_lower}' with class {provider_class.__name__}")

    def get_registered_types(self) -> List[str]:
         """Returns a list of registered provider type strings."""
         return list(self._provider_registry.keys())

    def create_provider(
        self,
        provider_id: str, # Added for context in logging/errors
        provider_config: ProviderConfig,
        gateway_config: GatewayConfig
    ) -> BaseProvider:
        """
        Creates and returns an instance of the specified LLM provider.

        Args:
            provider_id: A unique identifier for this specific provider instance (e.g., 'azure-gpt4-eastus').
            provider_config: The configuration object for the provider instance.
                             Must contain a valid 'provider_type'.
            gateway_config: The global gateway configuration object.

        Returns:
            An initialized instance of the requested provider class inheriting from BaseProvider.

        Raises:
            ProviderFactoryError: If the provider type is unknown, or if instantiation fails.
            TypeError: If configuration arguments are invalid.
        """
        if not provider_config or not hasattr(provider_config, 'provider_type'):
            raise TypeError("Invalid provider_config provided. Must be a ProviderConfig object with 'provider_type'.")

        provider_type = provider_config.provider_type
        if not provider_type or not isinstance(provider_type, str):
            raise TypeError(f"provider_config for '{provider_id}' must have a non-empty string 'provider_type'.")

        provider_type_lower = provider_type.lower()
        provider_class = self._provider_registry.get(provider_type_lower)

        # 1. Handle Unknown Provider Type
        if not provider_class:
            error_message = (f"Unknown provider type '{provider_type}' requested for provider ID '{provider_id}'. "
                             f"Registered types are: {self.get_registered_types()}")
            logger.error(error_message)
            raise ProviderFactoryError(error_message)

        logger.info(f"Creating provider instance for ID '{provider_id}' of type '{provider_type}' using class {provider_class.__name__}...")

        # 2. Handle Instantiation Errors
        try:
            # Instantiate the provider class, passing necessary configs
            provider_instance = provider_class(
                provider_config=provider_config,
                gateway_config=gateway_config
            )
            logger.info(f"Successfully created provider instance for ID '{provider_id}'.")
            return provider_instance

        except (ValueError, TypeError, KeyError) as config_err:
            # Errors often related to missing/invalid config values or API keys within provider __init__
            error_message = (f"Configuration error during instantiation of provider ID '{provider_id}' "
                             f"(type: '{provider_type}'): {config_err}")
            logger.error(error_message, exc_info=True) # Include traceback for config errors
            raise ProviderFactoryError(error_message) from config_err

        except ConnectionError as conn_err:
             # Errors related to initial connectivity checks if performed in __init__
            error_message = (f"Connection error during instantiation of provider ID '{provider_id}' "
                             f"(type: '{provider_type}'): {conn_err}")
            logger.error(error_message, exc_info=False) # Don't need full traceback for simple connection fail
            raise ProviderFactoryError(error_message) from conn_err

        except ImportError as import_err:
            # Handles cases where provider dependencies might be missing, though ideally caught earlier
             error_message = (f"Import error likely due to missing dependencies for provider type '{provider_type}' "
                              f"(provider ID: '{provider_id}'): {import_err}")
             logger.error(error_message, exc_info=False)
             raise ProviderFactoryError(error_message) from import_err

        except Exception as e:
            # Catch any other unexpected errors during provider initialization
            error_message = (f"Unexpected error during instantiation of provider ID '{provider_id}' "
                             f"(type: '{provider_type}'): {e}")
            logger.error(error_message, exc_info=True) # Include traceback for unexpected errors
            raise ProviderFactoryError(error_message) from e


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