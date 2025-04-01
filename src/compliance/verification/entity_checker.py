import re
from typing import List, Dict, Any, Optional
import uuid

class EntityComplianceChecker:
    """Checks entities for compliance violations (PII, PHI, etc.)."""
    def __init__(self, compliance_config):
        self.config = compliance_config

    def check_compliance(self, token_text, hypothetical_text, current_entities, constraints):
        """
        Check if adding token would create or complete protected entities.
        
        Args:
            token_text: The token text being considered
            hypothetical_text: Text if token is added
            current_entities: Already detected entities
            constraints: Compliance constraints to check
            
        Returns:
            Dict with is_compliant flag and compliance score
        """
        # Step 1: Extract entities from hypothetical text
        entity_constraints = [c for c in constraints if c.get("type") == "entity"]
        if not entity_constraints:
            return {'is_compliant': True, 'compliance_score': 1.0}
        
        # Check if we need to perform entity extraction
        # Only do this if token could complete an entity
        should_extract = self._could_complete_entity(token_text, hypothetical_text)
        if not should_extract:
            return {'is_compliant': True, 'compliance_score': 1.0}
        
        # Perform entity extraction (would call an NER model in a real implementation)
        hypothetical_entities = self._extract_entities(hypothetical_text)
        
        # Step 2: Check for new entities that weren't in current_entities
        new_entities = []
        for entity in hypothetical_entities:
            # Check if this entity is new or modified
            if not self._entity_exists(entity, current_entities):
                new_entities.append(entity)
        
        if not new_entities:
            return {'is_compliant': True, 'compliance_score': 1.0}
        
        # Step 3: Check new entities against constraints
        violations = []
        for entity in new_entities:
            for constraint in entity_constraints:
                if self._violates_constraint(entity, constraint):
                    violations.append({
                        "entity": entity,
                        "constraint": constraint,
                        "severity": constraint.get("severity", "medium")
                    })
        
        # Calculate compliance score
        if not violations:
            return {'is_compliant': True, 'compliance_score': 1.0}
        
        # Calculate score based on violation severity
        severity_sum = sum(self._severity_to_score(v["severity"]) for v in violations)
        compliance_score = max(0.0, 1.0 - (severity_sum / 10.0))  # Scale appropriately
        
        # Determine compliance based on threshold
        threshold = self.config.get("entity_compliance_threshold", 0.7)
        is_compliant = compliance_score >= threshold
        
        return {
            'is_compliant': is_compliant,
            'compliance_score': compliance_score,
            'violations': violations
        }

    def _could_complete_entity(self, token_text, hypothetical_text):
        """Check if token could potentially complete an entity"""
        # Simple heuristics to check if token_text could complete an entity
        # 1. Check if token is part of a word boundary
        if token_text.strip() and not token_text.isspace():
            return True
            
        # 2. Check token against common entity patterns
        entity_markers = ["Inc", "Corp", "LLC", "Ltd", "@", ".com", "Dr.", "Mr.", "Ms."]
        if any(marker in token_text for marker in entity_markers):
            return True
            
        # 3. Check if token completes a potential entity pattern
        entity_patterns = [
            r'\b[A-Z][a-z]+ ' + re.escape(token_text.strip()),  # First name + token
            r'\d{3}-\d{2}-' + re.escape(token_text.strip()),  # Partial SSN + token
            r'\w+@' + re.escape(token_text.strip())  # Partial email + token
        ]
        
        for pattern in entity_patterns:
            if re.search(pattern, hypothetical_text[-50:]):
                return True
                
        return False

    def _extract_entities(self, text):
        """Extract entities from text (simplified placeholder)"""
        entities = []
        
        # This is a simplified placeholder - in a real implementation, 
        # this would call a proper NER model or use regex patterns for entity detection
        
        # Check for email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "type": "EMAIL",
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # Check for SSN pattern
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            entities.append({
                "type": "SSN",
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95
            })
        
        # Check for credit card pattern
        cc_pattern = r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'
        for match in re.finditer(cc_pattern, text):
            entities.append({
                "type": "CREDIT_CARD",
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # Check for person name pattern (very simplified)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        for match in re.finditer(name_pattern, text):
            entities.append({
                "type": "PERSON",
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.7
            })
        
        return entities

    def _entity_exists(self, entity, entity_list):
        """Check if entity already exists in the entity list"""
        for existing in entity_list:
            # Check if the spans overlap
            if (entity["start"] <= existing["end"] and 
                entity["end"] >= existing["start"] and
                entity["type"] == existing["type"]):
                return True
        return False

    def _violates_constraint(self, entity, constraint):
        """Check if entity violates a constraint"""
        # Check if entity type is prohibited
        prohibited_types = constraint.get("prohibited_entity_types", [])
        if entity["type"] in prohibited_types:
            return True
            
        # Check if entity type requires authorization
        auth_required_types = constraint.get("authorization_required_types", [])
        if entity["type"] in auth_required_types:
            # In a real implementation, would check for authorization in context
            return True
            
        return False

    def _severity_to_score(self, severity):
        """Convert severity to numerical score for calculations"""
        severity_scores = {
            "low": 1.0,
            "medium": 3.0,
            "high": 6.0,
            "critical": 10.0
        }
        return severity_scores.get(severity, 3.0)