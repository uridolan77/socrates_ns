

class DomainSpecificRules:
    """
    Provides domain-specific rules for regulatory compliance.
    """
    def __init__(self):
        # Initialize domain registry
        self.domain_registry = self._initialize_domain_registry()
        
    def get_rules_for_framework(self, framework_id):
        """
        Get domain-specific rules for a specific regulatory framework.
        
        Args:
            framework_id: ID of the regulatory framework
            
        Returns:
            List of domain-specific rules for the framework
        """
        # Generate framework-specific domain rules
        if framework_id == "GDPR":
            return self._get_gdpr_domain_rules()
        elif framework_id == "HIPAA":
            return self._get_hipaa_domain_rules()
        elif framework_id == "EU_AI_Act":
            return self._get_eu_ai_act_domain_rules()
        elif framework_id == "FDA":
            return self._get_fda_domain_rules()
        elif framework_id == "ISO_21434":
            return self._get_iso_21434_domain_rules()
        else:
            return []  # No specific rules for this framework
    
    def _initialize_domain_registry(self):
        """Initialize registry of specific domains with regulatory relevance."""
        return {
            'healthcare': {
                'keywords': ['healthcare', 'medical', 'clinical', 'patient', 'hospital', 'doctor', 'treatment'],
                'relevant_frameworks': ['HIPAA', 'GDPR', 'FDA']
            },
            'finance': {
                'keywords': ['finance', 'banking', 'investment', 'credit', 'loan', 'insurance', 'trading'],
                'relevant_frameworks': ['GDPR']
            },
            'automotive': {
                'keywords': ['automotive', 'vehicle', 'car', 'driving', 'driver', 'road', 'traffic'],
                'relevant_frameworks': ['ISO_21434', 'EU_AI_Act']
            },
            'education': {
                'keywords': ['education', 'school', 'student', 'teacher', 'learning', 'academic', 'classroom'],
                'relevant_frameworks': ['GDPR']
            }
        }
    
    def _get_gdpr_domain_rules(self):
        """Get GDPR-specific domain rules."""
        rules = []
        
        # Healthcare domain rules for GDPR
        rules.append({
            'id': 'gdpr_domain_001',
            'name': 'Healthcare Data Processing Requirements',
            'description': 'Special requirements for healthcare data processing under GDPR',
            'type': 'domain_specific',
            'subtype': 'domain_requirements',
            'action': 'enforce_requirements',
            'domain': 'healthcare',
            'requirements': [
                'explicit_consent_for_health_data',
                'data_minimization_for_health_data',
                'special_category_data_protection'
            ],
            'severity': 'high',
            'enforcement_level': 'mandatory',
            'conditions': [
                {'type': 'domain_match', 'domain': 'healthcare'}
            ]
        })
        
        # Finance domain rules for GDPR
        rules.append({
            'id': 'gdpr_domain_002',
            'name': 'Financial Data Processing Requirements',
            'description': 'Special requirements for financial data processing under GDPR',
            'type': 'domain_specific',
            'subtype': 'domain_requirements',
            'action': 'enforce_requirements',
            'domain': 'finance',
            'requirements': [
                'legitimate_interest_assessment',
                'data_protection_impact_assessment',
                'automated_decision_making_limitations'
            ],
            'severity': 'high',
            'enforcement_level': 'mandatory',
            'conditions': [
                {'type': 'domain_match', 'domain': 'finance'}
            ]
        })
        
        return rules
    
    def _get_hipaa_domain_rules(self):
        """Get HIPAA-specific domain rules."""
        rules = []
        
        # Healthcare domain rules for HIPAA
        rules.append({
            'id': 'hipaa_domain_001',
            'name': 'Clinical Advice Restrictions',
            'description': 'Restrictions on providing clinical advice under HIPAA',
            'type': 'domain_specific',
            'subtype': 'content_restrictions',
            'action': 'enforce_restrictions',
            'domain': 'healthcare',
            'restrictions': [
                'no_specific_treatment_recommendations',
                'no_diagnostic_conclusions',
                'require_medical_disclaimer'
            ],
            'severity': 'high',
            'enforcement_level': 'mandatory',
            'conditions': [
                {'type': 'domain_match', 'domain': 'healthcare'}
            ]
        })
        
        return rules
    
    def _get_eu_ai_act_domain_rules(self):
        """Get EU AI Act-specific domain rules."""
        rules = []
        
        # Healthcare domain rules for EU AI Act
        rules.append({
            'id': 'eu_ai_act_domain_001',
            'name': 'Medical AI Classification Requirements',
            'description': 'Requirements for AI systems in healthcare under EU AI Act',
            'type': 'domain_specific',
            'subtype': 'domain_requirements',
            'action': 'enforce_requirements',
            'domain': 'healthcare',
            'requirements': [
                'high_risk_classification_disclosure',
                'human_oversight_requirement',
                'technical_documentation_requirement'
            ],
            'severity': 'high',
            'enforcement_level': 'mandatory',
            'conditions': [
                {'type': 'domain_match', 'domain': 'healthcare'},
                {'type': 'concept_present', 'concept': 'ai_system'}
            ]
        })
        
        # Automotive domain rules for EU AI Act
        rules.append({
            'id': 'eu_ai_act_domain_002',
            'name': 'Automotive AI Classification Requirements',
            'description': 'Requirements for AI systems in automotive under EU AI Act',
            'type': 'domain_specific',
            'subtype': 'domain_requirements',
            'action': 'enforce_requirements',
            'domain': 'automotive',
            'requirements': [
                'safety_critical_systems_disclosure',
                'risk_management_requirement',
                'human_oversight_requirement'
            ],
            'severity': 'high',
            'enforcement_level': 'mandatory',
            'conditions': [
                {'type': 'domain_match', 'domain': 'automotive'},
                {'type': 'concept_present', 'concept': 'ai_system'}
            ]
        })
        
        return rules
    
    def _get_fda_domain_rules(self):
        """Get FDA-specific domain rules."""
        rules = []
        
        # Healthcare domain rules for FDA
        rules.append({
            'id': 'fda_domain_001',
            'name': 'Medical Product Claim Restrictions',
            'description': 'Restrictions on medical product claims under FDA regulations',
            'type': 'domain_specific',
            'subtype': 'content_restrictions',
            'action': 'enforce_restrictions',
            'domain': 'healthcare',
            'restrictions': [
                'no_unapproved_use_promotion',
                'require_fair_balance',
                'require_important_safety_information',
                'no_overstatement_of_efficacy'
            ],
            'severity': 'high',
            'enforcement_level': 'mandatory',
            'conditions': [
                {'type': 'domain_match', 'domain': 'healthcare'},
                {'type': 'entity_present', 'entity_type': 'DRUG_NAME'}
            ]
        })
        
        return rules
    
    def _get_iso_21434_domain_rules(self):
        """Get ISO 21434-specific domain rules."""
        rules = []
        
        # Automotive domain rules for ISO 21434
        rules.append({
            'id': 'iso_21434_domain_001',
            'name': 'Automotive Cybersecurity Information Restrictions',
            'description': 'Restrictions on automotive cybersecurity information under ISO 21434',
            'type': 'domain_specific',
            'subtype': 'content_restrictions',
            'action': 'enforce_restrictions',
            'domain': 'automotive',
            'restrictions': [
                'no_detailed_vulnerability_disclosure',
                'no_exploitation_instructions',
                'require_responsible_disclosure_statement'
            ],
            'severity': 'high',
            'enforcement_level': 'mandatory',
            'conditions': [
                {'type': 'domain_match', 'domain': 'automotive'},
                {'type': 'concept_present', 'concept': 'cybersecurity'}
            ]
        })
        
        return rules