import numpy as np
import logging
import re
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from src.utils.text.semantic_analyzer import TextSe
from src.utils.text.token_tracker import TokenTracker

class TokenComplianceGate:
    """
    Gate that filters token predictions to ensure compliance
    by intervening at generation time
    """
    
    def __init__(self, tokenizer, regulatory_framework, rules):
        """
        Initialize token compliance gate
        
        Args:
            tokenizer: Tokenizer for decoding/encoding tokens
            regulatory_framework: Regulatory framework to enforce
            rules: Compliance rules to enforce
        """
        self.tokenizer = tokenizer
        self.framework = regulatory_framework
        self.rules = rules
        
        # Initialize semantic state tracking
        self.semantic_state = {
            "entities": {},
            "contexts": {},
            "violations": [],
            "sensitive_tokens": set(),
            "safe_tokens": set()
        }
    
    def filter_logits(self, logits, generated_tokens=None, context=None):
        """
        Filter logits to enforce compliance by masking prohibited tokens
        
        Args:
            logits: Token logits from the model
            generated_tokens: Previously generated tokens
            context: Additional context
            
        Returns:
            Filtered logits
        """
        # Convert to numpy for manipulation
        logits_np = logits.cpu().numpy() if hasattr(logits, 'cpu') else np.array(logits)
        
        # Update semantic state based on generated tokens
        if generated_tokens is not None:
            self._update_semantic_state(generated_tokens)
        
        # Get list of token IDs that would violate constraints
        prohibited_token_ids = self._get_prohibited_token_ids(context)
        
        # Create a mask to filter logits
        if prohibited_token_ids:
            # Set prohibited tokens to large negative values
            for token_id in prohibited_token_ids:
                logits_np[token_id] = -1e9
        
        # Convert back to original format
        if hasattr(logits, 'cpu'):
            filtered_logits = torch.tensor(logits_np, device=logits.device)
        else:
            filtered_logits = logits_np
        
        return filtered_logits
    
    def _update_semantic_state(self, tokens):
        """
        Update semantic state based on new tokens
        
        Args:
            tokens: New tokens to analyze
        """
        # Decode tokens to text
        if isinstance(tokens, list):
            text = self.tokenizer.decode(tokens)
        else:
            text = tokens
        
        # Extract entities and update state
        entities = self._extract_entities(text)
        for entity in entities:
            entity_type = entity["type"]
            entity_value = entity["text"]
            
            # Track entity in state
            if entity_type not in self.semantic_state["entities"]:
                self.semantic_state["entities"][entity_type] = []
            
            if entity_value not in self.semantic_state["entities"][entity_type]:
                self.semantic_state["entities"][entity_type].append(entity_value)
        
        # Extract context indicators
        contexts = self._extract_contexts(text)
        for context_type, context_value in contexts.items():
            self.semantic_state["contexts"][context_type] = context_value
        
        # Update sensitive tokens
        sensitive_tokens = self._identify_sensitive_tokens(text, entities)
        self.semantic_state["sensitive_tokens"].update(sensitive_tokens)
    
    def _extract_entities(self, text):
        """
        Extract entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities
        """
        # Simple entity extraction
        entity_patterns = {
            "PersonalData": {
                "Email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "Phone": r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b',
                "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
                "CreditCard": r'\b(?:\d{4}[- ]?){3}\d{4}\b'
            },
            "MedicalData": {
                "Diagnosis": r'\bdiagnosed with\s+([A-Za-z\s]+)\b',
                "Medication": r'\bprescribed\s+([A-Za-z\s]+)\b',
                "Treatment": r'\btreatment for\s+([A-Za-z\s]+)\b'
            },
            "FinancialData": {
                "BankAccount": r'\baccount\s+(?:number|#)\s*:?\s*\d{8,12}\b',
                "Income": r'\bincome of \$[0-9,.]+\b|\b\$[0-9,.]+\s+income\b'
            }
        }
        
        entities = []
        
        for category, patterns in entity_patterns.items():
            for entity_type, pattern in patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        "type": entity_type,
                        "category": category,
                        "text": match.group(),
                        "position": (match.start(), match.end())
                    })
        
        return entities
    
    def _extract_contexts(self, text):
        """
        Extract context indicators from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of context types and values
        """
        contexts = {}
        
        # Context patterns
        context_patterns = {
            "consent": {
                "pattern": r'\b(?:consent|permission|authorize|approval)\b',
                "default": False
            },
            "purpose": {
                "pattern": r'purpose is to\s+([^.]+)',
                "default": None
            },
            "security": {
                "pattern": r'\b(?:encrypted|secure|protected)\b',
                "default": False
            }
        }
        
        for context_type, config in context_patterns.items():
            pattern = config["pattern"]
            default = config["default"]
            
            match = re.search(pattern, text.lower())
            if match:
                if context_type == "purpose" and match.groups():
                    contexts[context_type] = match.group(1)
                else:
                    contexts[context_type] = True
            else:
                contexts[context_type] = default
        
        return contexts
    
    def _identify_sensitive_tokens(self, text, entities):
        """
        Identify sensitive tokens in text
        
        Args:
            text: Text to analyze
            entities: Extracted entities
            
        Returns:
            Set of sensitive token IDs
        """
        sensitive_tokens = set()
        
        # Add tokens from sensitive entities
        for entity in entities:
            # Tokenize the entity text
            entity_tokens = self.tokenizer.encode(entity["text"])
            sensitive_tokens.update(entity_tokens)
        
        # Add common sensitive terms
        sensitive_terms = [
            "password", "secret", "confidential", "private", 
            "ssn", "social security", "credit card", "cvv"
        ]
        
        for term in sensitive_terms:
            if term in text.lower():
                term_tokens = self.tokenizer.encode(term)
                sensitive_tokens.update(term_tokens)
        
        return sensitive_tokens
    
    def _get_prohibited_token_ids(self, context=None):
        """
        Get token IDs that would violate compliance constraints
        
        Args:
            context: Additional context for evaluation
            
        Returns:
            List of prohibited token IDs
        """
        prohibited_ids = set()
        
        # Check each rule
        for rule in self.rules:
            rule_type = rule.get("type", "unknown")
            
            if rule_type == "token_blacklist":
                # Direct token blacklist
                blacklisted_tokens = rule.get("tokens", [])
                for token in blacklisted_tokens:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    prohibited_ids.add(token_id)
            
            elif rule_type == "entity_disclosure":
                # Check if entity disclosure is allowed
                entity_category = rule.get("entity_category", "")
                if entity_category in self.semantic_state["entities"]:
                    # Entity exists in state
                    
                    # Check if disclosure is prohibited
                    disclosure_prohibited = self._violates_constraint(
                        entity_category, 
                        rule.get("conditions", {}),
                        context
                    )
                    
                    if disclosure_prohibited:
                        # Add entity tokens to prohibited list
                        for entity_value in self.semantic_state["entities"][entity_category]:
                            entity_tokens = self.tokenizer.encode(entity_value)
                            prohibited_ids.update(entity_tokens)
            
            elif rule_type == "sensitive_continuation":
                # Check if continuing with sensitive information
                if self.semantic_state["sensitive_tokens"]:
                    # We have sensitive tokens in state
                    
                    # Check if sensitive continuation is allowed
                    continuation_allowed = self._check_continuation_allowed(
                        rule.get("conditions", {}),
                        context
                    )
                    
                    if not continuation_allowed:
                        # Get tokens that would continue sensitive information
                        sensitive_continuations = self._get_sensitive_continuations()
                        prohibited_ids.update(sensitive_continuations)
        
        return list(prohibited_ids)
    
    def _violates_constraint(self, entity_category, conditions, context=None):
        """
        Check if using an entity category would violate constraints
        
        Args:
            entity_category: Category of entity to check
            conditions: Constraint conditions
            context: Additional context
            
        Returns:
            True if constraint would be violated, False otherwise
        """
        # Default to safe approach - assume violation
        if not conditions:
            return True
        
        # Check context requirements
        required_contexts = conditions.get("required_contexts", {})
        for context_type, required_value in required_contexts.items():
            # Get actual context value
            actual_value = self.semantic_state["contexts"].get(context_type)
            
            # Check if context requirement is satisfied
            if isinstance(required_value, bool):
                # Boolean context (e.g., consent)
                if required_value and not actual_value:
                    return True  # Violation - required context missing
            else:
                # String context (e.g., purpose)
                if required_value and not actual_value:
                    return True  # Violation - required context missing
                elif required_value and required_value not in str(actual_value):
                    return True  # Violation - wrong context value
        
        # Check prohibited combinations
        prohibited_combinations = conditions.get("prohibited_combinations", [])
        for combo in prohibited_combinations:
            # Check if all categories in combo are present
            if all(category in self.semantic_state["entities"] for category in combo):
                return True  # Violation - prohibited combination
        
        # No violations found
        return False
    
    def _check_continuation_allowed(self, conditions, context=None):
        """
        Check if continuation with sensitive information is allowed
        
        Args:
            conditions: Conditions for allowing continuation
            context: Additional context
            
        Returns:
            True if continuation is allowed, False otherwise
        """
        # Check context requirements
        required_contexts = conditions.get("required_contexts", {})
        for context_type, required_value in required_contexts.items():
            # Get actual context value
            actual_value = self.semantic_state["contexts"].get(context_type)
            
            # Check if context requirement is satisfied
            if isinstance(required_value, bool):
                # Boolean context (e.g., consent)
                if required_value and not actual_value:
                    return False  # Not allowed - required context missing
            else:
                # String context (e.g., purpose)
                if required_value and not actual_value:
                    return False  # Not allowed - required context missing
                elif required_value and required_value not in str(actual_value):
                    return False  # Not allowed - wrong context value
        
        # Check custom continuation conditions
        # ...
        
        # Default to allowing continuation if all checks pass
        return True
    
    def _get_sensitive_continuations(self):
        """
        Get token IDs that would continue sensitive information
        
        Returns:
            Set of token IDs for sensitive continuations
        """
        # This would be more sophisticated in a real implementation
        # For now, return some common continuation tokens
        continuations = [
            "is", "for", ":", "=", "-", "/", "\\", "_",
            "password", "number", "id", "secret", "key"
        ]
        
        continuation_ids = set()
        for word in continuations:
            token_ids = self.tokenizer.encode(word)
            continuation_ids.update(token_ids)
        
        return continuation_ids
    
    def _decode_token(self, token_id):
        """
        Decode token ID to string
        
        Args:
            token_id: Token ID to decode
            
        Returns:
            Decoded token string
        """
        if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
            return self.tokenizer.convert_ids_to_tokens(token_id)
        else:
            # Decode full sequence and take first token
            return self.tokenizer.decode([token_id]).strip()
    
    def get_safe_token_ids(self, context=None):
        """
        Get token IDs that are safe to use in the current context
        
        Args:
            context: Additional context
            
        Returns:
            List of safe token IDs
        """
        # Get prohibited tokens
        prohibited_ids = set(self._get_prohibited_token_ids(context))
        
        # Get all possible token IDs
        all_token_ids = set(range(self.tokenizer.vocab_size))
        
        # Safe tokens are those not prohibited
        safe_ids = all_token_ids - prohibited_ids
        
        return list(safe_ids)