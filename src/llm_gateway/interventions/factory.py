# llm_gateway/interventions/factory.py
import asyncio
import importlib
import logging
from typing import Dict, List, Optional, Type, Any

# Ensure correct import paths based on your project structure
from llm_gateway.core.models import GatewayConfig
from llm_gateway.interventions.base import BaseIntervention

logger = logging.getLogger(__name__)

class InterventionFactoryError(Exception):
    """Custom exception for intervention factory errors."""
    pass

class InterventionFactory:
    """
    Factory responsible for discovering, loading, and instantiating interventions.
    Manages intervention registration, instantiation, caching, and cleanup.
    """
    def __init__(self, gateway_config: GatewayConfig, allow_overwrite: bool = False):
        """Initialize the InterventionFactory."""
        self.gateway_config = gateway_config
        self._allow_overwrite = allow_overwrite
        self._intervention_classes: Dict[str, Type[BaseIntervention]] = {}
        self._intervention_instances: Dict[str, BaseIntervention] = {} # Cache instances
        self._instance_locks: Dict[str, asyncio.Lock] = {}
        self._load_intervention_definitions() # Load classes on init

    def _load_intervention_definitions(self):
        """Load intervention class definitions based on gateway configuration."""
        # Look for intervention definitions in the structured 'interventions' field
        intervention_configs = self.gateway_config.interventions
        if not isinstance(intervention_configs, dict):
            logger.warning("GatewayConfig.interventions is not a dict or is missing. No interventions loaded.")
            return

        logger.info(f"Loading intervention definitions ({len(intervention_configs)} found)...")
        for name, config in intervention_configs.items():
            if not isinstance(config, dict) or "class" not in config:
                logger.error(f"Invalid config for intervention '{name}'. Missing 'class'. Skipping.")
                continue

            class_path = config["class"]
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                intervention_class = getattr(module, class_name)

                if not issubclass(intervention_class, BaseIntervention):
                    raise TypeError(f"Class '{class_path}' doesn't inherit from BaseIntervention.")

                # Use intervention's declared name; config key 'name' is just for lookup
                intervention_name = getattr(intervention_class, 'name', None)
                if not intervention_name:
                     logger.error(f"Intervention class {class_path} must define a 'name' attribute. Skipping.")
                     continue

                if intervention_name in self._intervention_classes and not self._allow_overwrite:
                    raise ValueError(f"Intervention name '{intervention_name}' (from {class_path}) already registered and overwrite disallowed.")
                elif intervention_name in self._intervention_classes and self._allow_overwrite:
                    logger.warning(f"Intervention name '{intervention_name}' (from {class_path}) conflicts. Overwriting.")

                self._intervention_classes[intervention_name] = intervention_class
                logger.debug(f"Registered intervention '{intervention_name}' -> {class_path}")

            except (ImportError, AttributeError, TypeError, ValueError) as e:
                logger.error(f"Failed to load intervention '{name}' (class: '{class_path}'): {e}", exc_info=True)

        logger.info(f"Registered {len(self._intervention_classes)} intervention types.")

    def get_registered_intervention_names(self) -> List[str]:
        """Return names of all successfully registered interventions."""
        return list(self._intervention_classes.keys())

    async def get_intervention(self, name: str) -> BaseIntervention:
        """Gets or creates an intervention instance by its registered name."""
        # Ensure lock exists for the intervention name
        async with asyncio.Lock(): # Lock for creating locks
             if name not in self._instance_locks:
                  self._instance_locks[name] = asyncio.Lock()

        async with self._instance_locks[name]:
            if name not in self._intervention_instances:
                logger.info(f"Creating new instance for intervention: {name}")
                intervention_class = self._intervention_classes.get(name)
                if not intervention_class:
                    raise InterventionFactoryError(f"Unknown intervention name: '{name}'. Available: {self.get_registered_intervention_names()}")

                # Get specific config from the *original* config structure using the key 'name' might map to
                # This requires finding the original config key if intervention_class.name differs from the key 'name'
                # For simplicity, assume the key used in config ('name' in the loop above) IS the unique identifier
                # Or better: Use intervention_class.name as the primary key everywhere
                intervention_config_entry = self.gateway_config.interventions.get(name, {})
                intervention_specific_config = intervention_config_entry.get("config", {})

                try:
                    instance = intervention_class(config=intervention_specific_config)
                    if hasattr(instance, "initialize_async") and callable(instance.initialize_async):
                        logger.info(f"Running async initialization for intervention {name}...")
                        await instance.initialize_async()
                        logger.info(f"Async initialization complete for {name}.")
                    self._intervention_instances[name] = instance
                    logger.info(f"Successfully created instance for intervention: {name}")

                except Exception as e:
                    logger.error(f"Failed to instantiate/initialize intervention '{name}': {e}", exc_info=True)
                    # Clean up potential partial instance
                    if name in self._intervention_instances: del self._intervention_instances[name]
                    raise InterventionFactoryError(f"Failed to create intervention '{name}': {e}") from e

            return self._intervention_instances[name]

    def get_cached_instance(self, name: str) -> Optional[BaseIntervention]:
         """Gets a cached intervention instance if it exists, without creating."""
         return self._intervention_instances.get(name)

    def get_all_cached_instances(self) -> List[BaseIntervention]:
         """Gets all currently cached intervention instances."""
         return list(self._intervention_instances.values())

    async def cleanup_all(self):
        """Cleans up all cached intervention instances."""
        logger.info(f"Cleaning up {len(self._intervention_instances)} intervention instance(s)...")
        tasks = []
        intervention_names = list(self._intervention_instances.keys())
        for name in intervention_names:
            instance = self._intervention_instances.pop(name, None)
            if instance and hasattr(instance, "cleanup") and callable(instance.cleanup):
                 logger.debug(f"Scheduling cleanup for intervention {name}")
                 tasks.append(instance.cleanup())
            if name in self._instance_locks:
                 del self._instance_locks[name]

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Log errors during cleanup
            for i, result in enumerate(results):
                 if isinstance(result, Exception):
                      # Need a way to map result back to intervention name if logging specific failures
                      logger.error(f"Error during intervention cleanup: {result}")
        logger.info("Intervention instance cleanup complete.")

