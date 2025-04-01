import re
import json
import datetime
import numpy as np
from collections import defaultdict

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from io import BytesIO
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not available. PDF report generation will not be functional.")

class ComplianceProofTracer:
    """
    Traces the chain of reasoning for compliance decisions
    to provide explainable and auditable compliance proofs.
    """
    
    def __init__(self, regulatory_framework):
        """
        Initialize compliance proof tracer
        
        Args:
            regulatory_framework: The regulatory framework to generate proofs for
        """
        self.framework = regulatory_framework
        self.reasoning_chain = []
        self.proof_steps = []
        self.entity_cache = {}
        self.premise_patterns = self._initialize_premise_patterns()
        self.inference_rules = self._initialize_inference_rules()
        
    def _initialize_premise_patterns(self):
        """
        Initialize patterns for extracting premises based on regulatory framework
        
        Returns:
            Dictionary of premise patterns
        """
        # Common patterns across frameworks
        common_patterns = {
            "personal_data": {
                "pattern": r'\b(name|address|email|phone|birth\s*date|ssn|social\s*security|passport|id\s*number)\b',
                "type": "entity_presence"
            },
            "consent": {
                "pattern": r'\b(consent|permission|authorize|agreed|opt[- ]in)\b',
                "type": "state"
            },
            "purpose_statement": {
                "pattern": r'(purpose|reason|goal|objective).{1,50}(is|for|to)(.{1,100})',
                "type": "purpose_declaration"
            }
        }
        
        # Framework-specific patterns
        if self.framework == "GDPR":
            gdpr_patterns = {
                "data_subject_rights": {
                    "pattern": r'\b(right\s*to\s*access|right\s*to\s*erasure|right\s*to\s*be\s*forgotten|right\s*to\s*rectification|data\s*portability)\b',
                    "type": "rights_reference"
                },
                "data_transfer": {
                    "pattern": r'\b(transfer|send|transmit|share).{1,30}(data|information).{1,50}(outside|third|party|country|international)\b',
                    "type": "data_movement"
                },
                "legitimate_interest": {
                    "pattern": r'\b(legitimate\s*interest|legal\s*obligation|vital\s*interest|public\s*interest|official\s*authority)\b',
                    "type": "legal_basis"
                }
            }
            return {**common_patterns, **gdpr_patterns}
            
        elif self.framework == "HIPAA":
            hipaa_patterns = {
                "phi": {
                    "pattern": r'\b(health|medical|treatment|diagnosis|medication|condition|patient).{1,50}(information|data|record|history)\b',
                    "type": "sensitive_data"
                },
                "authorization": {
                    "pattern": r'\b(authorization|consent|permission).{1,50}(signed|obtained|given|provided|documented)\b',
                    "type": "authorization_evidence"
                },
                "minimum_necessary": {
                    "pattern": r'\b(minimum|necessary|essential|required).{1,50}(information|data)\b',
                    "type": "data_limitation"
                },
                "safeguards": {
                    "pattern": r'\b(encrypt|secure|protect|safeguard|password|access\s*control)\b',
                    "type": "security_measure"
                }
            }
            return {**common_patterns, **hipaa_patterns}
            
        elif self.framework == "CCPA":
            ccpa_patterns = {
                "sale_of_data": {
                    "pattern": r'\b(sell|sale|sold).{1,50}(personal|information|data)\b',
                    "type": "data_transaction"
                },
                "opt_out": {
                    "pattern": r'\b(opt[-\s]out|do\s*not\s*sell|right\s*to\s*opt[-\s]out)\b',
                    "type": "consumer_choice"
                },
                "minor_consent": {
                    "pattern": r'\b(minor|child|under\s*16|under\s*13).{1,50}(consent|opt[-\s]in|permission|authorization)\b',
                    "type": "minor_protection"
                }
            }
            return {**common_patterns, **ccpa_patterns}
        
        # Default to common patterns if framework not recognized
        return common_patterns
        
    def _initialize_inference_rules(self):
        """
        Initialize inference rules for the regulatory framework
        
        Returns:
            List of inference rules
        """
        # Common inference rules
        common_rules = [
            {
                "id": "personal_data_present",
                "premises": ["personal_data"],
                "conclusion": "contains_personal_data",
                "confidence_factor": 0.9,
                "description": "Text contains personal data identifiers"
            },
            {
                "id": "consent_mentioned",
                "premises": ["consent"],
                "conclusion": "consent_referenced",
                "confidence_factor": 0.7,
                "description": "Text references user consent"
            }
        ]
        
        # Framework-specific rules
        if self.framework == "GDPR":
            gdpr_rules = [
                {
                    "id": "data_with_consent",
                    "premises": ["contains_personal_data", "consent_referenced"],
                    "conclusion": "personal_data_with_consent",
                    "confidence_factor": 0.8,
                    "description": "Personal data is processed with consent reference"
                },
                {
                    "id": "data_with_legitimate_interest",
                    "premises": ["contains_personal_data", "legitimate_interest"],
                    "conclusion": "personal_data_with_legitimate_interest",
                    "confidence_factor": 0.75,
                    "description": "Personal data is processed with legitimate interest basis"
                },
                {
                    "id": "compliant_with_consent",
                    "premises": ["personal_data_with_consent", "purpose_statement"],
                    "conclusion": "gdpr_compliant",
                    "confidence_factor": 0.85,
                    "description": "Processing appears GDPR compliant with consent and purpose"
                },
                {
                    "id": "compliant_with_legitimate_interest",
                    "premises": ["personal_data_with_legitimate_interest", "purpose_statement"],
                    "conclusion": "gdpr_compliant",
                    "confidence_factor": 0.8,
                    "description": "Processing appears GDPR compliant with legitimate interest"
                },
                {
                    "id": "data_without_basis",
                    "premises": ["contains_personal_data"],
                    "excluded_premises": ["consent_referenced", "legitimate_interest"],
                    "conclusion": "gdpr_non_compliant",
                    "confidence_factor": 0.7,
                    "description": "Personal data without clear legal basis"
                },
                {
                    "id": "international_transfer_compliance",
                    "premises": ["data_transfer", "safeguards_mentioned"],
                    "conclusion": "compliant_transfer",
                    "confidence_factor": 0.75,
                    "description": "International transfer with safeguards mentioned"
                }
            ]
            return common_rules + gdpr_rules
            
        elif self.framework == "HIPAA":
            hipaa_rules = [
                {
                    "id": "phi_with_authorization",
                    "premises": ["phi", "authorization"],
                    "conclusion": "phi_with_authorization",
                    "confidence_factor": 0.85,
                    "description": "PHI with patient authorization"
                },
                {
                    "id": "phi_with_minimum_necessary",
                    "premises": ["phi", "minimum_necessary"],
                    "conclusion": "minimum_necessary_applied",
                    "confidence_factor": 0.8,
                    "description": "Minimum necessary principle applied to PHI"
                },
                {
                    "id": "phi_with_safeguards",
                    "premises": ["phi", "safeguards"],
                    "conclusion": "phi_with_safeguards",
                    "confidence_factor": 0.8,
                    "description": "PHI with security safeguards"
                },
                {
                    "id": "hipaa_compliant",
                    "premises": ["phi_with_authorization", "minimum_necessary_applied", "phi_with_safeguards"],
                    "conclusion": "hipaa_compliant",
                    "confidence_factor": 0.9,
                    "description": "Processing appears HIPAA compliant"
                },
                {
                    "id": "phi_without_authorization",
                    "premises": ["phi"],
                    "excluded_premises": ["authorization"],
                    "conclusion": "hipaa_non_compliant",
                    "confidence_factor": 0.8,
                    "description": "PHI without authorization"
                }
            ]
            return common_rules + hipaa_rules
            
        elif self.framework == "CCPA":
            ccpa_rules = [
                {
                    "id": "sale_with_opt_out",
                    "premises": ["sale_of_data", "opt_out"],
                    "conclusion": "sale_with_opt_out_option",
                    "confidence_factor": 0.85,
                    "description": "Data sale with opt-out option"
                },
                {
                    "id": "minor_data_with_consent",
                    "premises": ["personal_data", "minor_consent"],
                    "conclusion": "compliant_minor_data",
                    "confidence_factor": 0.9,
                    "description": "Minor's data with proper consent"
                },
                {
                    "id": "ccpa_compliant_sale",
                    "premises": ["sale_with_opt_out_option"],
                    "conclusion": "ccpa_compliant",
                    "confidence_factor": 0.8,
                    "description": "Data sale appears CCPA compliant with opt-out"
                },
                {
                    "id": "ccpa_non_compliant_sale",
                    "premises": ["sale_of_data"],
                    "excluded_premises": ["opt_out"],
                    "conclusion": "ccpa_non_compliant",
                    "confidence_factor": 0.75,
                    "description": "Data sale without opt-out option"
                }
            ]
            return common_rules + ccpa_rules
        
        # Default to common rules if framework not recognized
        return common_rules
        
    def trace_compliance(self, text, rules, context=None):
        """
        Trace compliance reasoning and generate proof
        
        Args:
            text: Text to analyze for compliance
            rules: Compliance rules to apply
            context: Additional context for evaluation
            
        Returns:
            Dictionary with compliance results and proof
        """
        # Reset reasoning chain for new trace
        self.reasoning_chain = []
        self.proof_steps = []
        
        # Extract premises from text
        premises = self._extract_premises(text, context)
        
        # Record initial premises
        self.reasoning_chain.append({
            "step": "premise_extraction",
            "premises": premises,
            "explanation": "Extracted initial premises from text"
        })
        
        # Add context as premises if available
        if context:
            context_premises = self._extract_premises_from_context(context)
            premises.extend(context_premises)
            
            self.reasoning_chain.append({
                "step": "context_integration",
                "premises": context_premises,
                "explanation": "Added premises from context"
            })
        
        # Apply rules with reasoning
        derived_facts = premises.copy()
        conclusion_explanations = {}
        
        # Apply inference rules until no new facts can be derived
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while iteration < max_iterations:
            new_facts_derived = False
            
            for rule in self.inference_rules:
                # Check if rule can be applied
                if self._can_apply_rule(rule, derived_facts):
                    # Apply the rule
                    inference_result = self._apply_inference_rule(rule, derived_facts)
                    conclusion = inference_result["conclusion"]
                    
                    # Check if this is a new fact
                    if conclusion not in derived_facts:
                        derived_facts.append(conclusion)
                        conclusion_explanations[conclusion] = inference_result["explanation"]
                        new_facts_derived = True
                        
                        # Record this inference step
                        self.reasoning_chain.append({
                            "step": f"inference_{iteration}",
                            "rule": rule["id"],
                            "conclusion": conclusion,
                            "confidence": inference_result["confidence"],
                            "explanation": inference_result["explanation"]
                        })
            
            # If no new facts were derived, we're done
            if not new_facts_derived:
                break
                
            iteration += 1
        
        # Check final compliance based on derived facts
        compliance_result = self._determine_compliance(derived_facts)
        
        # Generate final proof
        proof = {
            "compliant": compliance_result["compliant"],
            "confidence": compliance_result["confidence"],
            "reasoning_chain": self.reasoning_chain,
            "derived_facts": derived_facts,
            "explanations": conclusion_explanations,
            "final_conclusion": compliance_result["conclusion"],
            "explanation": compliance_result["explanation"]
        }
        
        return proof
        
    def _extract_premises(self, text, context=None):
        """
        Extract premises from text for inference
        
        Args:
            text: Text to analyze
            context: Additional context information
            
        Returns:
            List of premises extracted from text
        """
        if not text:
            return []
            
        premises = []
        
        # Apply each premise pattern
        for premise_type, pattern_info in self.premise_patterns.items():
            pattern = pattern_info["pattern"]
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Extract match information
                matched_text = match.group(0)
                start, end = match.span()
                
                # Extract capture groups if any
                groups = match.groups()
                captured_values = []
                if groups:
                    captured_values = [g for g in groups if g]
                
                # Add as premise
                premises.append(premise_type)
                
                # Record detailed information about this premise
                self.proof_steps.append({
                    "type": "premise_extraction",
                    "premise": premise_type,
                    "matched_text": matched_text,
                    "position": (start, end),
                    "captured_values": captured_values,
                    "pattern_type": pattern_info["type"]
                })
                
                # Only count each premise type once
                break
        
        # Extract entities and their relationships
        entities = self._extract_entities(text)
        for entity in entities:
            entity_type = entity["type"]
            
            # Add entity-specific premises
            if entity_type == "Person":
                premises.append("person_mentioned")
            elif entity_type == "Organization":
                premises.append("organization_mentioned")
            elif entity_type == "Location":
                premises.append("location_mentioned")
            elif entity_type == "Date":
                premises.append("date_mentioned")
            
            # Record proof step
            self.proof_steps.append({
                "type": "entity_extraction",
                "entity_type": entity_type,
                "value": entity["text"],
                "premise": f"{entity_type.lower()}_mentioned"
            })
        
        # Additional analysis for specific frameworks
        if self.framework == "GDPR":
            # Check for specific GDPR language
            gdpr_terms = [
                "data controller", "data processor", "supervisory authority",
                "data protection officer", "dpo", "data subject", "personal data",
                "special category", "profiling", "pseudonymization"
            ]
            for term in gdpr_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    premises.append(f"gdpr_term_{term.replace(' ', '_')}")
                    
                    self.proof_steps.append({
                        "type": "terminology_detection",
                        "term": term,
                        "framework": "GDPR",
                        "premise": f"gdpr_term_{term.replace(' ', '_')}"
                    })
        
        # Remove duplicates while preserving order
        unique_premises = []
        for premise in premises:
            if premise not in unique_premises:
                unique_premises.append(premise)
        
        return unique_premises
        
    def _extract_premises_from_context(self, context):
        """
        Extract premises from context object
        
        Args:
            context: Context dictionary
            
        Returns:
            List of premises from context
        """
        premises = []
        
        # Common context fields
        if context.get("has_consent", False):
            premises.append("explicit_consent")
            
            self.proof_steps.append({
                "type": "context_integration",
                "context_key": "has_consent",
                "value": True,
                "premise": "explicit_consent"
            })
        
        if context.get("purpose"):
            premises.append("explicit_purpose")
            
            self.proof_steps.append({
                "type": "context_integration",
                "context_key": "purpose",
                "value": context["purpose"],
                "premise": "explicit_purpose"
            })
        
        # Framework-specific context
        if self.framework == "GDPR":
            if context.get("legal_basis"):
                legal_basis = context["legal_basis"]
                premises.append(f"legal_basis_{legal_basis}")
                
                self.proof_steps.append({
                    "type": "context_integration",
                    "context_key": "legal_basis",
                    "value": legal_basis,
                    "premise": f"legal_basis_{legal_basis}"
                })
        
        elif self.framework == "HIPAA":
            if context.get("authorization", False):
                premises.append("patient_authorization")
                
                self.proof_steps.append({
                    "type": "context_integration",
                    "context_key": "authorization",
                    "value": True,
                    "premise": "patient_authorization"
                })
                
            if context.get("treatment_purpose", False):
                premises.append("treatment_purpose")
                
                self.proof_steps.append({
                    "type": "context_integration",
                    "context_key": "treatment_purpose",
                    "value": True,
                    "premise": "treatment_purpose"
                })
        
        return premises
        
    def _extract_entities(self, text):
        """
        Extract entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities
        """
        # Check if we've already processed this text
        if text in self.entity_cache:
            return self.entity_cache[text]
            
        # Simple entity extraction patterns
        entity_patterns = {
            "Person": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "Email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "Phone": r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b',
            "Date": r'\b(?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12][0-9]|3[01])[/.-](?:19|20)\d{2}\b',
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "CreditCard": r'\b(?:\d{4}[- ]?){3}\d{4}\b',
            "Address": r'\b\d+\s+[A-Za-z]+\s+(?:St|Ave|Rd|Blvd|Dr)(?:\.|eet|enue|oad|oulevard|ive)?\b',
            "Organization": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*\s+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co)\b'
        }
        
        entities = []
        
        for entity_type, pattern in entity_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "type": entity_type,
                    "text": match.group(0),
                    "position": (match.start(), match.end())
                })
        
        # Cache the results
        self.entity_cache[text] = entities
        
        return entities
        
    def _can_apply_rule(self, rule, facts):
        """
        Check if a rule can be applied given the current facts
        
        Args:
            rule: Rule to check
            facts: Current derived facts
            
        Returns:
            True if rule can be applied, False otherwise
        """
        # Check that all required premises are present
        required_premises = rule.get("premises", [])
        for premise in required_premises:
            if premise not in facts:
                return False
        
        # Check that excluded premises are absent
        excluded_premises = rule.get("excluded_premises", [])
        for premise in excluded_premises:
            if premise in facts:
                return False
                
        # Check if conclusion already derived
        conclusion = rule.get("conclusion")
        if conclusion in facts:
            return False
            
        return True
        
    def _apply_inference_rule(self, rule, facts):
        """
        Apply inference rule to derive new conclusions
        
        Args:
            rule: Inference rule to apply
            facts: Current known facts
            
        Returns:
            Dictionary with inference results
        """
        conclusion = rule["conclusion"]
        confidence = rule["confidence_factor"]
        description = rule.get("description", "")
        
        # Get premises that contributed to this conclusion
        used_premises = [premise for premise in rule["premises"] if premise in facts]
        
        # Build explanation
        if description:
            explanation = description
        else:
            premises_text = ", ".join(used_premises)
            explanation = f"Derived '{conclusion}' from: {premises_text}"
        
        # Record proof step
        self.proof_steps.append({
            "type": "inference",
            "rule": rule["id"],
            "premises": used_premises,
            "conclusion": conclusion,
            "confidence": confidence,
            "explanation": explanation
        })
        
        return {
            "conclusion": conclusion,
            "confidence": confidence,
            "explanation": explanation,
            "premises_used": used_premises,
            "rule_id": rule["id"]
        }
        
    def _determine_compliance(self, facts):
        """
        Determine overall compliance based on derived facts
        
        Args:
            facts: Derived facts from inference
            
        Returns:
            Dictionary with compliance determination
        """
        # Framework-specific compliance determination
        if self.framework == "GDPR":
            if "gdpr_compliant" in facts:
                return {
                    "compliant": True,
                    "confidence": 0.85,
                    "conclusion": "gdpr_compliant",
                    "explanation": "Processing appears compliant with GDPR requirements"
                }
            elif "gdpr_non_compliant" in facts:
                return {
                    "compliant": False,
                    "confidence": 0.8,
                    "conclusion": "gdpr_non_compliant",
                    "explanation": "Processing appears non-compliant with GDPR requirements"
                }
            elif "contains_personal_data" in facts:
                # Personal data but no clear compliance determination
                return {
                    "compliant": False,
                    "confidence": 0.7,
                    "conclusion": "gdpr_uncertain",
                    "explanation": "Contains personal data but insufficient evidence of GDPR compliance"
                }
            else:
                # No personal data, so GDPR may not apply
                return {
                    "compliant": True,
                    "confidence": 0.9,
                    "conclusion": "gdpr_not_applicable",
                    "explanation": "No personal data detected, GDPR may not apply"
                }
                
        elif self.framework == "HIPAA":
            if "hipaa_compliant" in facts:
                return {
                    "compliant": True,
                    "confidence": 0.85,
                    "conclusion": "hipaa_compliant",
                    "explanation": "Processing appears compliant with HIPAA requirements"
                }
            elif "hipaa_non_compliant" in facts:
                return {
                    "compliant": False,
                    "confidence": 0.8,
                    "conclusion": "hipaa_non_compliant",
                    "explanation": "Processing appears non-compliant with HIPAA requirements"
                }
            elif "phi" in facts:
                # PHI but no clear compliance determination
                return {
                    "compliant": False,
                    "confidence": 0.7,
                    "conclusion": "hipaa_uncertain",
                    "explanation": "Contains PHI but insufficient evidence of HIPAA compliance"
                }
            else:
                # No PHI, so HIPAA may not apply
                return {
                    "compliant": True,
                    "confidence": 0.9,
                    "conclusion": "hipaa_not_applicable",
                    "explanation": "No PHI detected, HIPAA may not apply"
                }
                
        elif self.framework == "CCPA":
            if "ccpa_compliant" in facts:
                return {
                    "compliant": True,
                    "confidence": 0.85,
                    "conclusion": "ccpa_compliant",
                    "explanation": "Processing appears compliant with CCPA requirements"
                }
            elif "ccpa_non_compliant" in facts:
                return {
                    "compliant": False,
                    "confidence": 0.8,
                    "conclusion": "ccpa_non_compliant",
                    "explanation": "Processing appears non-compliant with CCPA requirements"
                }
            elif "sale_of_data" in facts:
                # Data sale but no clear compliance determination
                return {
                    "compliant": False,
                    "confidence": 0.7,
                    "conclusion": "ccpa_uncertain",
                    "explanation": "Contains data sale but insufficient evidence of CCPA compliance"
                }
            else:
                # No data sale, so specific CCPA sale provisions may not apply
                return {
                    "compliant": True,
                    "confidence": 0.8,
                    "conclusion": "ccpa_sale_not_applicable",
                    "explanation": "No data sale detected, specific CCPA sale provisions may not apply"
                }
        
        # Default case
        if "contains_personal_data" in facts:
            return {
                "compliant": False,
                "confidence": 0.6,
                "conclusion": "generic_non_compliant",
                "explanation": "Contains personal data but uncertain compliance status"
            }
        else:
            return {
                "compliant": True,
                "confidence": 0.7,
                "conclusion": "generic_compliant",
                "explanation": "No obvious compliance issues detected"
            }
            
    def _apply_rule_with_proof(self, rule, text, context=None):
        """
        Apply a compliance rule and provide reasoning proof
        
        Args:
            rule: Compliance rule to apply
            text: Text to evaluate
            context: Additional context for evaluation
            
        Returns:
            Dictionary with rule application results and proof
        """
        # Extract rule components
        rule_id = rule.get("id", "unknown_rule")
        rule_type = rule.get("type", "unknown")
        
        # Extract premises from text and context
        premises = self._extract_premises(text, context)
        
        # Initialize result
        result = {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "compliant": False,
            "confidence": 0.0,
            "explanation": "",
            "proof_chain": []
        }
        
        # Record initial evidence
        result["proof_chain"].append({
            "step": "evidence_collection",
            "premises": premises,
            "explanation": "Extracted initial evidence from text and context"
        })
        
        # Apply rule based on type
        if rule_type == "entity_disclosure":
            # Check for unauthorized entity disclosure
            entity_type = rule.get("entity_type", "")
            authorization_required = rule.get("authorization_required", True)
            
            # Check for entity presence
            entity_present = False
            entity_evidence = []
            
            for premise in premises:
                if entity_type in premise:
                    entity_present = True
                    entity_evidence.append(premise)
            
            # Record entity analysis
            result["proof_chain"].append({
                "step": "entity_analysis",
                "entity_type": entity_type,
                "entity_present": entity_present,
                "evidence": entity_evidence,
                "explanation": f"Analyzed text for {entity_type} presence"
            })
            
            if not entity_present:
                # No entity, so rule doesn't apply
                result["compliant"] = True
                result["confidence"] = 0.9
                result["explanation"] = f"No {entity_type} detected, rule does not apply"
                return result
            
            # Check for authorization
            has_authorization = False
            authorization_evidence = []
            
            if context and context.get("has_authorization", False):
                has_authorization = True
                authorization_evidence.append("explicit_authorization_context")
            
            for premise in premises:
                if "consent" in premise or "authorization" in premise:
                    has_authorization = True
                    authorization_evidence.append(premise)
            
            # Record authorization analysis
            result["proof_chain"].append({
                "step": "authorization_analysis",
                "has_authorization": has_authorization,
                "evidence": authorization_evidence,
                "explanation": "Analyzed text and context for authorization evidence"
            })
            
            # Determine compliance
            if authorization_required and not has_authorization:
                result["compliant"] = False
                result["confidence"] = 0.85
                result["explanation"] = f"Unauthorized disclosure of {entity_type}"
            else:
                result["compliant"] = True
                result["confidence"] = 0.8
                result["explanation"] = f"Authorized disclosure of {entity_type}"
        
        elif rule_type == "purpose_limitation":
            # Check for purpose limitation compliance
            allowed_purposes = rule.get("allowed_purposes", [])
            
            # Check for purpose statement
            has_purpose = False
            purpose_evidence = []
            stated_purpose = ""
            
            if context and context.get("purpose"):
                has_purpose = True
                stated_purpose = context["purpose"]
                purpose_evidence.append("explicit_purpose_context")
            
            for premise, step in zip(premises, self.proof_steps):
                if "purpose" in premise:
                    has_purpose = True
                    purpose_evidence.append(premise)
                    
                    # Extract the purpose from proof steps
                    if step.get("captured_values"):
                        stated_purpose = step["captured_values"][-1]
            
            # Record purpose analysis
            result["proof_chain"].append({
                "step": "purpose_analysis",
                "has_purpose": has_purpose,
                "stated_purpose": stated_purpose,
                "evidence": purpose_evidence,
                "explanation": "Analyzed text and context for purpose statements"
            })
            
            if not has_purpose:
                # No purpose statement, non-compliant with purpose limitation
                result["compliant"] = False
                result["confidence"] = 0.7
                result["explanation"] = "No clear purpose statement found"
                return result
            
            # Check if purpose matches allowed purposes
            purpose_allowed = False
            if not allowed_purposes:
                # No restrictions on purpose
                purpose_allowed = True
            else:
                # Check if stated purpose matches any allowed purpose
                purpose_allowed = any(
                    allowed.lower() in stated_purpose.lower() 
                    for allowed in allowed_purposes
                )
            
            # Record purpose limitation analysis
            result["proof_chain"].append({
                "step": "purpose_limitation_analysis",
                "purpose_allowed": purpose_allowed,
                "allowed_purposes": allowed_purposes,
                "explanation": "Checked if stated purpose aligns with allowed purposes"
            })
            
            # Determine compliance
            if purpose_allowed:
                result["compliant"] = True
                result["confidence"] = 0.8
                result["explanation"] = "Purpose appears to align with allowed purposes"
            else:
                result["compliant"] = False
                result["confidence"] = 0.75
                result["explanation"] = "Purpose does not align with allowed purposes"
        
        elif rule_type == "data_minimization":
            # Check for data minimization compliance
            necessary_data_types = rule.get("necessary_data_types", [])
            
            # Extract data types present in text
            data_types_present = []
            for premise in premises:
                if "personal_data" in premise or "phi" in premise:
                    data_types_present.append(premise)
            
            # Record data presence analysis
            result["proof_chain"].append({
                "step": "data_presence_analysis",
                "data_types_present": data_types_present,
                "explanation": "Analyzed text for personal data presence"
            })
            
            if not data_types_present:
                # No personal data, so rule doesn't apply
                result["compliant"] = True
                result["confidence"] = 0.9
                result["explanation"] = "No personal data detected, rule does not apply"
                return result
            
            # Check if minimum necessary principle is mentioned
            minimum_necessary_mentioned = False
            for premise in premises:
                if "minimum" in premise or "necessary" in premise:
                    minimum_necessary_mentioned = True
                    break
            
            # Record minimum necessary analysis
            result["proof_chain"].append({
                "step": "minimum_necessary_analysis",
                "minimum_necessary_mentioned": minimum_necessary_mentioned,
                "explanation": "Checked for minimum necessary principle"
            })
            
            # Compare data types present with necessary data types
            excessive_data = False
            if necessary_data_types:
                for data_type in data_types_present:
                    if data_type not in necessary_data_types:
                        excessive_data = True
                        break
            
            # Determine compliance
            if excessive_data and not minimum_necessary_mentioned:
                result["compliant"] = False
                result["confidence"] = 0.7
                result["explanation"] = "Excessive data collection without minimum necessary principle"
            elif excessive_data and minimum_necessary_mentioned:
                result["compliant"] = False
                result["confidence"] = 0.6
                result["explanation"] = "Potential excessive data despite minimum necessary mention"
            elif minimum_necessary_mentioned:
                result["compliant"] = True
                result["confidence"] = 0.8
                result["explanation"] = "Minimum necessary principle applied to data collection"
            else:
                result["compliant"] = True
                result["confidence"] = 0.6
                result["explanation"] = "No clear evidence of excessive data collection"
        
        else:
            # Generic rule handling
            # Simplified implementation that checks for specific conditions
            conditions_met = 0
            conditions_total = 0
            
            if rule.get("conditions"):
                conditions = rule["conditions"]
                conditions_total = len(conditions)
                
                for condition in conditions:
                    if condition in premises:
                        conditions_met += 1
            
            # Record conditions analysis
            result["proof_chain"].append({
                "step": "conditions_analysis",
                "conditions_total": conditions_total,
                "conditions_met": conditions_met,
                "explanation": f"Checked {conditions_met} of {conditions_total} conditions"
            })
            
            # Determine compliance based on proportion of conditions met
            if conditions_total > 0:
                compliance_ratio = conditions_met / conditions_total
                
                if compliance_ratio >= 0.8:
                    result["compliant"] = True
                    result["confidence"] = 0.9
                    result["explanation"] = f"Met {conditions_met} of {conditions_total} conditions"
                elif compliance_ratio >= 0.5:
                    result["compliant"] = True
                    result["confidence"] = 0.7
                    result["explanation"] = f"Met {conditions_met} of {conditions_total} conditions"
                else:
                    result["compliant"] = False
                    result["confidence"] = 0.8
                    result["explanation"] = f"Only met {conditions_met} of {conditions_total} conditions"
            else:
                # No specific conditions, default to compliant with low confidence
                result["compliant"] = True
                result["confidence"] = 0.5
                result["explanation"] = "No specific conditions to evaluate"
        
        return result
        
    def generate_compliance_report(self, text, rules, context=None):
        """
        Generate comprehensive compliance report with proof chain
        
        Args:
            text: Text to analyze
            rules: Compliance rules to apply
            context: Additional context
            
        Returns:
            Dictionary with compliance report
        """
        # Trace compliance reasoning
        compliance_proof = self.trace_compliance(text, rules, context)
        
        # Apply each rule with detailed proof
        rule_results = []
        for rule in rules:
            rule_result = self._apply_rule_with_proof(rule, text, context)
            rule_results.append(rule_result)
        
        # Count rule compliance
        compliant_rules = [r for r in rule_results if r["compliant"]]
        non_compliant_rules = [r for r in rule_results if not r["compliant"]]
        
        # Overall compliance determination
        if non_compliant_rules:
            overall_compliant = False
            compliance_message = f"Failed {len(non_compliant_rules)} of {len(rules)} rules"
            confidence = sum(r["confidence"] for r in non_compliant_rules) / len(non_compliant_rules)
        else:
            overall_compliant = True
            compliance_message = f"Passed all {len(rules)} rules"
            confidence = sum(r["confidence"] for r in compliant_rules) / len(compliant_rules) if compliant_rules else 0.5
        
        # Generate report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "framework": self.framework,
            "text_analyzed": text[:500] + "..." if len(text) > 500 else text,
            "overall_compliance": {
                "compliant": overall_compliant,
                "confidence": confidence,
                "message": compliance_message
            },
            "rule_results": rule_results,
            "reasoning_proof": compliance_proof,
            "report_id": f"{self.framework.lower()}_report_{int(datetime.datetime.now().timestamp())}"
        }
        
        # Generate PDF if requested
        pdf_report = self._generate_pdf_report(report)
        if pdf_report:
            report["pdf_report"] = pdf_report
        
        return report
        
    def _generate_pdf_report(self, report_data):
        """
        Generate PDF compliance report
        
        Args:
            report_data: Report data to include in PDF
            
        Returns:
            Bytes of PDF report
        """
        if not REPORTLAB_AVAILABLE:
            return None
            
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles["Heading1"]
        heading2_style = styles["Heading2"]
        normal_style = styles["Normal"]
        
        # Add custom styles
        styles.add(ParagraphStyle(
            name='RightAlign',
            parent=styles['Normal'],
            alignment=2,  # 2 is right alignment
        ))
        
        styles.add(ParagraphStyle(
            name='Compliant',
            parent=styles['Normal'],
            textColor=colors.green,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='NonCompliant',
            parent=styles['Normal'],
            textColor=colors.red,
            fontName='Helvetica-Bold'
        ))
        
        # Title
        elements.append(Paragraph(f"{self.framework} Compliance Report", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Date
        date_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated: {date_text}", styles["RightAlign"]))
        elements.append(Spacer(1, 0.25*inch))
        
        # Summary
        elements.append(Paragraph("Compliance Summary", heading2_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Overall compliance status
        overall = report_data["overall_compliance"]
        compliance_style = "Compliant" if overall["compliant"] else "NonCompliant"
        elements.append(Paragraph(f"Status: {overall['message']}", styles[compliance_style]))
        elements.append(Paragraph(f"Confidence: {overall['confidence']:.2f}", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Rule results
        elements.append(Paragraph("Rule Results", heading2_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Create rule results table
        rule_data = [["Rule ID", "Status", "Confidence", "Explanation"]]
        
        for rule in report_data["rule_results"]:
            status = "✓ Compliant" if rule["compliant"] else "✗ Non-Compliant"
            rule_data.append([
                rule["rule_id"],
                status,
                f"{rule['confidence']:.2f}",
                rule["explanation"]
            ])
        
        # Create table
        rule_table = Table(rule_data, colWidths=[1.5*inch, 1*inch, 0.8*inch, 3*inch])
        
        # Style the table
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        
        # Add row-specific styling
        for i, rule in enumerate(report_data["rule_results"], 1):
            if rule["compliant"]:
                table_style.add('TEXTCOLOR', (1, i), (1, i), colors.green)
            else:
                table_style.add('TEXTCOLOR', (1, i), (1, i), colors.red)
        
        rule_table.setStyle(table_style)
        elements.append(rule_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Reasoning Chain
        if "reasoning_proof" in report_data and "reasoning_chain" in report_data["reasoning_proof"]:
            elements.append(Paragraph("Reasoning Process", heading2_style))
            elements.append(Spacer(1, 0.1*inch))
            
            for i, step in enumerate(report_data["reasoning_proof"]["reasoning_chain"]):
                elements.append(Paragraph(f"Step {i+1}: {step['step']}", styles["Heading3"]))
                
                if "premises" in step:
                    premises_text = ", ".join(step["premises"])
                    elements.append(Paragraph(f"<b>Premises:</b> {premises_text}", normal_style))
                
                if "rule" in step:
                    elements.append(Paragraph(f"<b>Rule:</b> {step['rule']}", normal_style))
                
                if "conclusion" in step:
                    elements.append(Paragraph(f"<b>Conclusion:</b> {step['conclusion']}", normal_style))
                
                if "explanation" in step:
                    elements.append(Paragraph(f"<b>Explanation:</b> {step['explanation']}", normal_style))
                
                elements.append(Spacer(1, 0.1*inch))
        
        # Build the PDF
        doc.build(elements)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data