import os
import json
import yaml
import base64
import logging
from typing import Dict, Any, Optional, Union, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("secure.config")

class SecureConfigManager:
    """
    Secure configuration manager for compliance systems.
    Handles loading, validation, and encryption of sensitive configuration.
    """
    
    def __init__(self, 
                config_path: str = None,
                encryption_key: str = None,
                env_prefix: str = "COMPLIANCE_"):
        """
        Initialize the secure configuration manager
        
        Args:
            config_path: Path to configuration file
            encryption_key: Key for encrypting/decrypting sensitive values (or env var)
            env_prefix: Prefix for environment variables
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.config = {}
        self.encrypted_fields = set()
        
        # Setup encryption
        self._setup_encryption(encryption_key)
        
        # Load configuration if path provided
        if config_path:
            self.load_config(config_path)
            
    def _setup_encryption(self, encryption_key: str = None):
        """
        Setup encryption with the provided key or from environment
        
        Args:
            encryption_key: Encryption key or name of environment variable
        """
        # Try to get key from provided value, environment, or generate new one
        key = None
        
        if encryption_key:
            if encryption_key.startswith("env:"):
                # Get from environment variable
                env_var = encryption_key[4:]
                key_str = os.environ.get(env_var)
                if key_str:
                    try:
                        key = base64.urlsafe_b64decode(key_str)
                    except:
                        logger.warning(f"Invalid encryption key in environment variable {env_var}")
            else:
                # Direct key value
                try:
                    key = base64.urlsafe_b64decode(encryption_key)
                except:
                    logger.warning("Invalid encryption key provided")
        
        # Look for key in environment with default name
        if not key:
            key_str = os.environ.get(f"{self.env_prefix}ENCRYPTION_KEY")
            if key_str:
                try:
                    key = base64.urlsafe_b64decode(key_str)
                except:
                    logger.warning(f"Invalid encryption key in environment variable {self.env_prefix}ENCRYPTION_KEY")
                    
        # Generate new key if none found
        if not key:
            key = Fernet.generate_key()
            logger.warning(
                f"No encryption key provided. Generated temporary key: "
                f"{base64.urlsafe_b64encode(key).decode()}. "
                f"This key will not persist across restarts."
            )
            
        # Create cipher
        self.cipher = Fernet(key)
        
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file (override instance value)
            
        Returns:
            Loaded configuration
        """
        path = config_path or self.config_path
        if not path:
            raise ValueError("No configuration path provided")
            
        # Check file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        # Determine file type from extension
        _, ext = os.path.splitext(path.lower())
        
        try:
            if ext == '.json':
                with open(path, 'r') as f:
                    self.config = json.load(f)
            elif ext in ('.yaml', '.yml'):
                with open(path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file type: {ext}")
                
            # Identify encrypted fields
            self._identify_encrypted_fields(self.config)
            
            # Decrypt encrypted values
            self._decrypt_config(self.config)
            
            # Override with environment variables
            self._apply_environment_overrides()
            
            logger.info(f"Loaded configuration from {path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def save_config(self, config_path: str = None, encrypt_sensitive: bool = True) -> None:
        """
        Save configuration to file
        
        Args:
            config_path: Path to save configuration (override instance value)
            encrypt_sensitive: Whether to encrypt sensitive fields
        """
        path = config_path or self.config_path
        if not path:
            raise ValueError("No configuration path provided")
            
        # Create a copy of the config to avoid modifying the original
        config_copy = self._deep_copy(self.config)
        
        # Encrypt sensitive fields if requested
        if encrypt_sensitive:
            self._encrypt_config(config_copy)
        
        # Determine file type from extension and save
        _, ext = os.path.splitext(path.lower())
        
        try:
            if ext == '.json':
                with open(path, 'w') as f:
                    json.dump(config_copy, f, indent=2)
            elif ext in ('.yaml', '.yml'):
                with open(path, 'w') as f:
                    yaml.dump(config_copy, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported configuration file type: {ext}")
                
            logger.info(f"Saved configuration to {path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key
        
        Args:
            key: Configuration key (dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Split key by dots for nested access
        parts = key.split('.')
        
        # Traverse config to find value
        current = self.config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
    
    def set(self, key: str, value: Any, sensitive: bool = False) -> None:
        """
        Set a configuration value
        
        Args:
            key: Configuration key (dot notation for nested keys)
            value: Value to set
            sensitive: Whether this value should be encrypted when saved
        """
        # Split key by dots for nested access
        parts = key.split('.')
        
        # Traverse config to set value
        current = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value
        
        # Mark as encrypted if sensitive
        if sensitive:
            self.encrypted_fields.add(key)
    
    def validate_required_fields(self, required_fields: List[str]) -> List[str]:
        """
        Validate that required fields are present
        
        Args:
            required_fields: List of required field keys
            
        Returns:
            List of missing fields
        """
        missing = []
        
        for field in required_fields:
            value = self.get(field)
            if value is None:
                missing.append(field)
                
        return missing
    
    def generate_config_hash(self) -> str:
        """
        Generate a hash of the configuration for integrity verification
        
        Returns:
            Configuration hash
        """
        # Convert config to a stable string representation
        config_str = json.dumps(self.config, sort_keys=True)
        
        # Generate hash
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def mask_sensitive_fields(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a copy of the configuration with sensitive fields masked
        
        Args:
            config: Configuration to mask (defaults to instance config)
            
        Returns:
            Masked configuration
        """
        if config is None:
            config = self.config
            
        # Create a deep copy of the config
        masked_config = self._deep_copy(config)
        
        # Mask encrypted fields
        for field in self.encrypted_fields:
            parts = field.split('.')
            current = masked_config
            
            # Traverse to the field
            found = True
            for i, part in enumerate(parts[:-1]):
                if part in current:
                    current = current[part]
                else:
                    found = False
                    break
                    
            # Mask the field if found
            if found and parts[-1] in current:
                current[parts[-1]] = "********"
                
        return masked_config
    
    def _identify_encrypted_fields(self, config: Dict[str, Any], prefix: str = "") -> None:
        """
        Identify encrypted fields in configuration
        
        Args:
            config: Configuration to scan
            prefix: Current key prefix for recursive calls
        """
        for key, value in config.items():
            current_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively scan dictionaries
                self._identify_encrypted_fields(value, current_key)
            elif isinstance(value, str) and value.startswith("encrypted:"):
                # Mark as encrypted field
                self.encrypted_fields.add(current_key)
    
    def _encrypt_config(self, config: Dict[str, Any], prefix: str = "") -> None:
        """
        Encrypt sensitive fields in configuration
        
        Args:
            config: Configuration to encrypt
            prefix: Current key prefix for recursive calls
        """
        for key, value in config.items():
            current_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process dictionaries
                self._encrypt_config(value, current_key)
            elif current_key in self.encrypted_fields and isinstance(value, str):
                # Don't re-encrypt already encrypted values
                if not value.startswith("encrypted:"):
                    # Encrypt the value
                    encrypted = self.cipher.encrypt(value.encode()).decode()
                    config[key] = f"encrypted:{encrypted}"
    
    def _decrypt_config(self, config: Dict[str, Any], prefix: str = "") -> None:
        """
        Decrypt encrypted fields in configuration
        
        Args:
            config: Configuration to decrypt
            prefix: Current key prefix for recursive calls
        """
        for key, value in config.items():
            current_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively process dictionaries
                self._decrypt_config(value, current_key)
            elif isinstance(value, str) and value.startswith("encrypted:"):
                # Extract and decrypt the value
                try:
                    encrypted = value[len("encrypted:"):]
                    decrypted = self.cipher.decrypt(encrypted.encode()).decode()
                    config[key] = decrypted
                    
                    # Remember this field is encrypted
                    self.encrypted_fields.add(current_key)
                except Exception as e:
                    logger.error(f"Error decrypting field {current_key}: {str(e)}")
                    # Leave as is if decryption fails
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration"""
        for env_key, env_value in os.environ.items():
            # Check if this is a configuration override
            if env_key.startswith(self.env_prefix):
                # Convert environment variable name to config key
                config_key = env_key[len(self.env_prefix):].lower().replace('__', '.')
                
                # Apply override
                self.set(config_key, env_value)
                logger.debug(f"Applied environment override for {config_key}")
    
    def _deep_copy(self, obj: Any) -> Any:
        """
        Create a deep copy of an object
        
        Args:
            obj: Object to copy
            
        Returns:
            Deep copy of object
        """
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj


class ComplianceConfigValidator:
    """
    Validator for compliance system configuration.
    Ensures that configuration meets requirements and follows best practices.
    """
    
    def __init__(self, schema_path: str = None):
        """
        Initialize the validator
        
        Args:
            schema_path: Path to validation schema file
        """
        self.schema = None
        if schema_path:
            self.load_schema(schema_path)
    
    def load_schema(self, schema_path: str) -> Dict[str, Any]:
        """
        Load validation schema from file
        
        Args:
            schema_path: Path to validation schema file
            
        Returns:
            Loaded schema
        """
        # Check file exists
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
        # Determine file type from extension
        _, ext = os.path.splitext(schema_path.lower())
        
        try:
            if ext == '.json':
                with open(schema_path, 'r') as f:
                    self.schema = json.load(f)
            elif ext in ('.yaml', '.yml'):
                with open(schema_path, 'r') as f:
                    self.schema = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported schema file type: {ext}")
                
            logger.info(f"Loaded validation schema from {schema_path}")
            return self.schema
            
        except Exception as e:
            logger.error(f"Error loading schema: {str(e)}")
            raise
    
    def validate(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate configuration against schema
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        if not self.schema:
            raise ValueError("No validation schema loaded")
            
        errors = []
        
        # Validate required fields
        for field_path, field_schema in self._get_field_schemas():
            if field_schema.get('required', False):
                value = self._get_nested_value(config, field_path)
                if value is None:
                    errors.append({
                        'path': field_path,
                        'error': 'required_field_missing',
                        'message': f"Required field '{field_path}' is missing"
                    })
                    continue
                    
            # If field exists, validate type and constraints
            value = self._get_nested_value(config, field_path)
            if value is not None:
                type_errors = self._validate_type(field_path, value, field_schema)
                if type_errors:
                    errors.extend(type_errors)
                    continue
                    
                # Validate constraints
                constraint_errors = self._validate_constraints(field_path, value, field_schema)
                if constraint_errors:
                    errors.extend(constraint_errors)
        
        # Validate conditional fields
        for condition in self.schema.get('conditions', []):
            condition_errors = self._validate_condition(config, condition)
            if condition_errors:
                errors.extend(condition_errors)
                
        return errors
    
    def _get_field_schemas(self) -> List[tuple]:
        """
        Get all field schemas from the validation schema
        
        Returns:
            List of tuples of (field_path, field_schema)
        """
        result = []
        
        def _extract_fields(schema, prefix=''):
            if 'properties' in schema:
                for field, field_schema in schema['properties'].items():
                    field_path = f"{prefix}.{field}" if prefix else field
                    result.append((field_path, field_schema))
                    
                    # Recurse into nested objects
                    if field_schema.get('type') == 'object':
                        _extract_fields(field_schema, field_path)
                        
        _extract_fields(self.schema)
        return result
    
    def _get_nested_value(self, obj: Dict[str, Any], path: str) -> Any:
        """
        Get a nested value from a dictionary
        
        Args:
            obj: Dictionary to get value from
            path: Dot-notation path to value
            
        Returns:
            Nested value or None if not found
        """
        parts = path.split('.')
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
                
        return current
    
    def _validate_type(self, path: str, value: Any, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate value type against schema
        
        Args:
            path: Path to the value
            value: Value to validate
            schema: Schema for the value
            
        Returns:
            List of validation errors
        """
        errors = []
        expected_type = schema.get('type')
        
        if not expected_type:
            return errors
            
        # Check type
        if expected_type == 'string':
            if not isinstance(value, str):
                errors.append({
                    'path': path,
                    'error': 'invalid_type',
                    'message': f"Field '{path}' should be a string",
                    'expected': 'string',
                    'actual': type(value).__name__
                })
        elif expected_type == 'number':
            if not isinstance(value, (int, float)):
                errors.append({
                    'path': path,
                    'error': 'invalid_type',
                    'message': f"Field '{path}' should be a number",
                    'expected': 'number',
                    'actual': type(value).__name__
                })
        elif expected_type == 'integer':
            if not isinstance(value, int):
                errors.append({
                    'path': path,
                    'error': 'invalid_type',
                    'message': f"Field '{path}' should be an integer",
                    'expected': 'integer',
                    'actual': type(value).__name__
                })
        elif expected_type == 'boolean':
            if not isinstance(value, bool):
                errors.append({
                    'path': path,
                    'error': 'invalid_type',
                    'message': f"Field '{path}' should be a boolean",
                    'expected': 'boolean',
                    'actual': type(value).__name__
                })
        elif expected_type == 'array':
            if not isinstance(value, list):
                errors.append({
                    'path': path,
                    'error': 'invalid_type',
                    'message': f"Field '{path}' should be an array",
                    'expected': 'array',
                    'actual': type(value).__name__
                })
            elif 'items' in schema and value:
                # Validate array items
                item_schema = schema['items']
                for i, item in enumerate(value):
                    item_path = f"{path}[{i}]"
                    item_errors = self._validate_type(item_path, item, item_schema)
                    errors.extend(item_errors)
        elif expected_type == 'object':
            if not isinstance(value, dict):
                errors.append({
                    'path': path,
                    'error': 'invalid_type',
                    'message': f"Field '{path}' should be an object",
                    'expected': 'object',
                    'actual': type(value).__name__
                })
                
        return errors
    
    def _validate_constraints(self, path: str, value: Any, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate value constraints against schema
        
        Args:
            path: Path to the value
            value: Value to validate
            schema: Schema for the value
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # String constraints
        if isinstance(value, str):
            if 'minLength' in schema and len(value) < schema['minLength']:
                errors.append({
                    'path': path,
                    'error': 'string_too_short',
                    'message': f"Field '{path}' should be at least {schema['minLength']} characters",
                    'min_length': schema['minLength'],
                    'actual_length': len(value)
                })
                
            if 'maxLength' in schema and len(value) > schema['maxLength']:
                errors.append({
                    'path': path,
                    'error': 'string_too_long',
                    'message': f"Field '{path}' should be at most {schema['maxLength']} characters",
                    'max_length': schema['maxLength'],
                    'actual_length': len(value)
                })
                
            if 'pattern' in schema:
                import re
                if not re.match(schema['pattern'], value):
                    errors.append({
                        'path': path,
                        'error': 'pattern_mismatch',
                        'message': f"Field '{path}' should match pattern: {schema['pattern']}",
                        'pattern': schema['pattern']
                    })
                    
            if 'enum' in schema and value not in schema['enum']:
                errors.append({
                    'path': path,
                    'error': 'invalid_enum_value',
                    'message': f"Field '{path}' should be one of: {', '.join(schema['enum'])}",
                    'allowed_values': schema['enum'],
                    'actual_value': value
                })
                
        # Number constraints
        if isinstance(value, (int, float)):
            if 'minimum' in schema and value < schema['minimum']:
                errors.append({
                    'path': path,
                    'error': 'number_too_small',
                    'message': f"Field '{path}' should be at least {schema['minimum']}",
                    'minimum': schema['minimum'],
                    'actual_value': value
                })
                
            if 'maximum' in schema and value > schema['maximum']:
                errors.append({
                    'path': path,
                    'error': 'number_too_large',
                    'message': f"Field '{path}' should be at most {schema['maximum']}",
                    'maximum': schema['maximum'],
                    'actual_value': value
                })
                
        # Array constraints
        if isinstance(value, list):
            if 'minItems' in schema and len(value) < schema['minItems']:
                errors.append({
                    'path': path,
                    'error': 'array_too_short',
                    'message': f"Field '{path}' should have at least {schema['minItems']} items",
                    'min_items': schema['minItems'],
                    'actual_items': len(value)
                })
                
            if 'maxItems' in schema and len(value) > schema['maxItems']:
                errors.append({
                    'path': path,
                    'error': 'array_too_long',
                    'message': f"Field '{path}' should have at most {schema['maxItems']} items",
                    'max_items': schema['maxItems'],
                    'actual_items': len(value)
                })
                
            if 'uniqueItems' in schema and schema['uniqueItems']:
                # Check for duplicates
                if len(value) != len(set(str(item) for item in value)):
                    errors.append({
                        'path': path,
                        'error': 'duplicate_items',
                        'message': f"Field '{path}' should have unique items"
                    })
                    
        # Object constraints
        if isinstance(value, dict):
            if 'minProperties' in schema and len(value) < schema['minProperties']:
                errors.append({
                    'path': path,
                    'error': 'object_too_small',
                    'message': f"Field '{path}' should have at least {schema['minProperties']} properties",
                    'min_properties': schema['minProperties'],
                    'actual_properties': len(value)
                })
                
            if 'maxProperties' in schema and len(value) > schema['maxProperties']:
                errors.append({
                    'path': path,
                    'error': 'object_too_large',
                    'message': f"Field '{path}' should have at most {schema['maxProperties']} properties",
                    'max_properties': schema['maxProperties'],
                    'actual_properties': len(value)
                })
                
            if 'required' in schema:
                for required_prop in schema['required']:
                    if required_prop not in value:
                        errors.append({
                            'path': f"{path}.{required_prop}",
                            'error': 'required_property_missing',
                            'message': f"Required property '{required_prop}' is missing from '{path}'"
                        })
                        
        return errors
    
    def _validate_condition(self, config: Dict[str, Any], condition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate a conditional constraint
        
        Args:
            config: Configuration to validate
            condition: Condition to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Get condition value
        if_path = condition.get('if', {}).get('path')
        if_value = condition.get('if', {}).get('value')
        
        if not if_path or if_value is None:
            return errors
            
        # Check if condition applies
        actual_value = self._get_nested_value(config, if_path)
        if actual_value == if_value:
            # Condition applies, check required fields
            for required_field in condition.get('then', {}).get('required', []):
                required_value = self._get_nested_value(config, required_field)
                if required_value is None:
                    errors.append({
                        'path': required_field,
                        'error': 'conditional_required_field_missing',
                        'message': f"Field '{required_field}' is required when '{if_path}' is '{if_value}'",
                        'condition': {'path': if_path, 'value': if_value}
                    })
                    
        return errors


# Example usage:
"""
# Create secure config manager
config_manager = SecureConfigManager(
    config_path="config.yaml",
    encryption_key="env:COMPLIANCE_ENCRYPTION_KEY",
    env_prefix="COMPLIANCE_"
)

# Get configuration values
api_key = config_manager.get("api.key")
db_password = config_manager.get("database.password")

# Set sensitive values
config_manager.set("api.key", "your-api-key", sensitive=True)
config_manager.set("database.password", "your-password", sensitive=True)

# Save configuration with encrypted sensitive values
config_manager.save_config()

# Create masked version for logging
masked_config = config_manager.mask_sensitive_fields()
print(json.dumps(masked_config, indent=2))

# Validate configuration
validator = ComplianceConfigValidator(schema_path="config-schema.yaml")
errors = validator.validate(config_manager.config)

if errors:
    print(f"Configuration validation errors:")
    for error in errors:
        print(f"- {error['message']}")
else:
    print("Configuration is valid")
"""