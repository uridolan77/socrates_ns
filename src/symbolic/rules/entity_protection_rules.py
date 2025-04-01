
class EntityProtectionRules:
    """
    Provides entity-based rules for regulatory compliance.
    """
    def __init__(self):
        # Initialize with standard entity rule templates
        self.rule_templates = self._initialize_rule_templates()
        
    def get_rules_for_framework(self, framework_id):
        """
        Get entity protection rules for a specific regulatory framework.
        
        Args:
            framework_id: ID of the regulatory framework
            
        Returns:
            List of entity protection rules for the framework
        """
        # Generate framework-specific rules
        if framework_id == "GDPR":
            return self._get_gdpr_entity_rules()
        elif framework_id == "HIPAA":
            return self._get_hipaa_entity_rules()
        elif framework_id == "EU_AI_Act":
            return self._get_eu_ai_act_entity_rules()
        elif framework_id == "FDA":
            return self._get_fda_entity_rules()
        elif framework_id == "ISO_21434":
            return self._get_iso_21434_entity_rules()
        else:
            return []  # No specific rules for this framework
    
    def get_conflict_resolution_strategy(self):
        """
        Get conflict resolution strategy for entity protection rules.
        
        Returns:
            Conflict resolution strategy
        """
        return {
            'priority': 'strictest',
            'resolution_method': 'combine_entities',
            'entity_handling': 'take_most_restrictive'
        }
    
    def _initialize_rule_templates(self):
        """Initialize rule templates for entity protection rules."""
        return {
            'entity_restriction': {
                'type': 'entity_protection',
                'subtype': 'entity_restriction',
                'action': 'block_entity',
                'entity_type_template': "{entity_type}",
                'severity': 'high'
            },
            'entity_anonymization': {
                'type': 'entity_protection',
                'subtype': 'entity_anonymization',
                'action': 'anonymize_entity',
                'entity_type_template': "{entity_type}",
                'severity': 'medium'
            },
            'entity_pseudonymization': {
                'type': 'entity_protection',
                'subtype': 'entity_pseudonymization',
                'action': 'pseudonymize_entity',
                'entity_type_template': "{entity_type}",
                'severity': 'low'
            },
            'entity_relationship_restriction': {
                'type': 'entity_protection',
                'subtype': 'entity_relationship_restriction',
                'action': 'block_relationship',
                'entity_types_template': "{entity_type1}, {entity_type2}",
                'relationship_template': "{relationship}",
                'severity': 'medium'
            }
        }
    
    def _get_gdpr_entity_rules(self):
        """Get GDPR-specific entity protection rules."""
        rules = []
        
        # PII entity protection
        rules.append({
            'id': 'gdpr_entity_001',
            'name': 'Personal Identifier Protection',
            'description': 'Protects personal identifiers such as names, emails, and addresses',
            'type': 'entity_protection',
            'subtype': 'entity_restriction',
            'action': 'anonymize_entity',
            'entity_types': ['PERSON', 'EMAIL', 'ADDRESS', 'PHONE_NUMBER', 'ID_NUMBER'],
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        # Special category data protection
        rules.append({
            'id': 'gdpr_entity_002',
            'name': 'Special Category Data Protection',
            'description': 'Protects special category data such as health, biometric, and political opinions',
            'type': 'entity_protection',
            'subtype': 'entity_restriction',
            'action': 'block_entity',
            'entity_types': ['HEALTH_DATA', 'BIOMETRIC_DATA', 'POLITICAL_OPINION', 'RELIGIOUS_BELIEF',
                            'ETHNIC_ORIGIN', 'SEXUAL_ORIENTATION'],
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        return rules
    
    def _get_hipaa_entity_rules(self):
        """Get HIPAA-specific entity protection rules."""
        rules = []
        
        # PHI entity protection
        rules.append({
            'id': 'hipaa_entity_001',
            'name': 'Protected Health Information Protection',
            'description': 'Protects PHI including patient names, medical record numbers, and health conditions',
            'type': 'entity_protection',
            'subtype': 'entity_restriction',
            'action': 'block_entity',
            'entity_types': ['PATIENT_NAME', 'MEDICAL_RECORD_NUMBER', 'HEALTH_CONDITION',
                            'TREATMENT_PROCEDURE', 'MEDICATION_NAME', 'PROVIDER_NAME'],
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        # Provider-patient relationship protection
        rules.append({
            'id': 'hipaa_entity_002',
            'name': 'Provider-Patient Relationship Protection',
            'description': 'Protects the relationship between providers and patients',
            'type': 'entity_protection',
            'subtype': 'entity_relationship_restriction',
            'action': 'block_relationship',
            'entity_types': ['PROVIDER_NAME', 'PATIENT_NAME'],
            'relationship': 'treats',
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        return rules
    
    def _get_eu_ai_act_entity_rules(self):
        """Get EU AI Act-specific entity protection rules."""
        rules = []
        
        # Biometric categorization protection
        rules.append({
            'id': 'eu_ai_act_entity_001',
            'name': 'Biometric Categorization Protection',
            'description': 'Prevents biometric categorization based on protected characteristics',
            'type': 'entity_protection',
            'subtype': 'entity_relationship_restriction',
            'action': 'block_relationship',
            'entity_types': ['BIOMETRIC_DATA', 'PROTECTED_CHARACTERISTIC'],
            'relationship': 'categorizes',
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        # Social scoring protection
        rules.append({
            'id': 'eu_ai_act_entity_002',
            'name': 'Social Scoring Protection',
            'description': 'Prevents social scoring of individuals',
            'type': 'entity_protection',
            'subtype': 'entity_relationship_restriction',
            'action': 'block_relationship',
            'entity_types': ['PERSON', 'SOCIAL_SCORE'],
            'relationship': 'has_score',
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        return rules
    
    def _get_fda_entity_rules(self):
        """Get FDA-specific entity protection rules."""
        rules = []
        
        # Unapproved drug-indication relationship protection
        rules.append({
            'id': 'fda_entity_001',
            'name': 'Unapproved Drug-Indication Relationship Protection',
            'description': 'Prevents associating drugs with unapproved indications',
            'type': 'entity_protection',
            'subtype': 'entity_relationship_restriction',
            'action': 'block_relationship',
            'entity_types': ['DRUG_NAME', 'UNAPPROVED_INDICATION'],
            'relationship': 'treats',
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        # Clinical trial result entity protection
        rules.append({
            'id': 'fda_entity_002',
            'name': 'Clinical Trial Result Protection',
            'description': 'Ensures accurate representation of clinical trial results',
            'type': 'entity_protection',
            'subtype': 'entity_accuracy',
            'action': 'flag_if_inaccurate',
            'entity_types': ['CLINICAL_TRIAL_RESULT'],
            'accuracy_threshold': 0.9,
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        return rules
    
    def _get_iso_21434_entity_rules(self):
        """Get ISO 21434-specific entity protection rules."""
        rules = []
        
        # Security vulnerability entity protection
        rules.append({
            'id': 'iso_21434_entity_001',
            'name': 'Security Vulnerability Protection',
            'description': 'Protects detailed information about security vulnerabilities',
            'type': 'entity_protection',
            'subtype': 'entity_restriction',
            'action': 'flag_entity',
            'entity_types': ['SECURITY_VULNERABILITY', 'EXPLOIT_CODE', 'ATTACK_VECTOR'],
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        # Vehicle security system entity protection
        rules.append({
            'id': 'iso_21434_entity_002',
            'name': 'Vehicle Security System Protection',
            'description': 'Protects detailed information about vehicle security systems',
            'type': 'entity_protection',
            'subtype': 'entity_restriction',
            'action': 'anonymize_entity',
            'entity_types': ['VEHICLE_SECURITY_SYSTEM', 'SECURITY_PROTOCOL', 'CRYPTOGRAPHIC_KEY'],
            'severity': 'high',
            'enforcement_level': 'mandatory'
        })
        
        return rules