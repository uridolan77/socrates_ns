from src.orchestration.application_processor import GatewayCompliantLanguageModelProcessor

class SemanticComplianceMonitor:
    """
    Monitors semantic compliance during text generation and evaluation
    by tracking entity relationships, context, and regulatory requirements.
    """
    def __init__(self, compliance_config):
        self.config = compliance_config
        self.entity_patterns = self._initialize_entity_patterns()
        self.sensitive_topics = self._initialize_sensitive_topics()
        self.relationship_rules = self._initialize_relationship_rules()
        self.domain_specific_rules = self._initialize_domain_specific_rules()
        
        # Tracking state
        self.detected_entities = {}
        self.topic_scores = {}
        self.semantic_state = {}
        
    def _initialize_entity_patterns(self):
        """Initialize patterns for entity detection"""
        # In a production system, this would be loaded from a database or config
        return {
            "PII": {
                "patterns": [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # SSN
                    r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'  # Credit card
                ],
                "sensitivity": "high"
            },
            "PHI": {
                "patterns": [
                    r'\b(?:patient|medical|health)\s+(?:record|data|information)\b',
                    r'\bdiagnosis\b',
                    r'\btreatment\s+plan\b'
                ],
                "sensitivity": "high"
            },
            "LOCATION": {
                "patterns": [
                    r'\b\d{5}(?:-\d{4})?\b',  # ZIP code
                    r'\b\d+\s+[A-Za-z0-9\s,]+(?:Road|Street|Avenue|Blvd)\b'  # Address
                ],
                "sensitivity": "medium"
            }
        }
    
    def _initialize_sensitive_topics(self):
        """Initialize sensitive topic detection patterns"""
        return {
            "violence": {
                "keywords": ["kill", "attack", "weapon", "violent", "bomb", "explosive"],
                "threshold": 0.6,
                "severity": "high"
            },
            "hate_speech": {
                "keywords": ["hate", "slur", "racist", "discrimination"],
                "threshold": 0.5,
                "severity": "high"
            },
            "financial_advice": {
                "keywords": ["invest", "stock", "financial advice", "buy shares", "securities"],
                "threshold": 0.7,
                "severity": "medium"
            },
            "medical_advice": {
                "keywords": ["treatment", "cure", "diagnose", "prescription", "medical advice"],
                "threshold": 0.7,
                "severity": "medium"
            }
        }
    
    def _initialize_relationship_rules(self):
        """Initialize rules for entity relationships"""
        return {
            "PII_disclosure": {
                "description": "PII should not be disclosed with specific identifiable context",
                "trigger_entities": ["PII", "LOCATION"],
                "condition": "co-occurrence",
                "max_distance": 200,  # characters
                "severity": "high"
            },
            "medical_advice_with_disclaimer": {
                "description": "Medical information should include appropriate disclaimers",
                "trigger_entities": ["PHI"],
                "condition": "requires_disclaimer",
                "disclaimer_pattern": r'(?:not medical advice|consult|healthcare professional)',
                "severity": "medium"
            }
        }
    
    def _initialize_domain_specific_rules(self):
        """Initialize domain-specific compliance rules"""
        return {
            "finance": {
                "disclaimers_required": True,
                "disclaimer_pattern": r'(?:not financial advice|for informational purposes|consult financial advisor)',
                "prohibited_phrases": [
                    "guaranteed returns", "risk-free investment", "certain profit"
                ]
            },
            "healthcare": {
                "disclaimers_required": True,
                "disclaimer_pattern": r'(?:not medical advice|consult healthcare professional|talk to your doctor)',
                "prohibited_phrases": [
                    "guaranteed cure", "miracle treatment", "100% effective"
                ]
            }
        }
    
    def evaluate_output(self, text, context=None, framework_ids=None):
        """
        Evaluate compliance of generated text output.
        
        Note: In a production system, entity extraction would be done using
        sophisticated NLP techniques. This implementation uses simple
        pattern matching as a placeholder.
        
        Args:
            text: Generated text to evaluate
            context: Optional context information
            framework_ids: Optional specific regulatory frameworks to check
            
        Returns:
            Dict with compliance evaluation results
        """
        # Reset state for new evaluation
        self.detected_entities = {}
        self.topic_scores = {}
        
        # Extract entities (simplified implementation)
        self._extract_entities(text)
        
        # Detect sensitive topics
        self._detect_sensitive_topics(text)
        
        # Check entity relationships
        relationship_violations = self._check_entity_relationships(text)
        
        # Check domain-specific rules
        domain = context.get('domain') if context else None
        domain_violations = self._check_domain_specific_rules(text, domain)
        
        # Check framework-specific rules if provided
        framework_violations = []
        if framework_ids:
            framework_violations = self._check_framework_rules(text, framework_ids, context)
        
        # Combine all violations
        all_violations = relationship_violations + domain_violations + framework_violations
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(all_violations)
        
        # Determine overall compliance
        strict_mode = self.config.get('strict_mode', False)
        if strict_mode:
            is_compliant = len(all_violations) == 0
        else:
            # In non-strict mode, only high/critical severity violations cause non-compliance
            is_compliant = not any(v.get('severity') in ['high', 'critical'] for v in all_violations)
        
        return {
            'is_compliant': is_compliant,
            'compliance_score': compliance_score,
            'violations': all_violations,
            'entities': self.detected_entities,
            'topics': self.topic_scores
        }
    
    def _extract_entities(self, text):
        """
        Extract entities from text using pattern matching.
        
        Note: A production implementation would use NER models or other
        NLP techniques for more accurate entity extraction.
        """
        import re
        
        for entity_type, config in self.entity_patterns.items():
            self.detected_entities[entity_type] = []
            
            for pattern in config["patterns"]:
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    self.detected_entities[entity_type].append({
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "sensitivity": config["sensitivity"]
                    })
    
    def _detect_sensitive_topics(self, text):
        """
        Detect sensitive topics in text using keyword matching.
        
        Note: A production implementation would use more sophisticated
        topic modeling or classification techniques.
        """
        text_lower = text.lower()
        
        for topic, config in self.sensitive_topics.items():
            # Count keyword occurrences
            keyword_count = sum(text_lower.count(k.lower()) for k in config["keywords"])
            
            # Calculate topic score (simple heuristic)
            text_length = len(text.split())
            topic_score = min(1.0, keyword_count / max(1, text_length / 50))
            
            self.topic_scores[topic] = {
                "score": topic_score,
                "threshold": config["threshold"],
                "severity": config["severity"],
                "exceeds_threshold": topic_score >= config["threshold"]
            }
    
    def _check_entity_relationships(self, text):
        """Check for problematic entity relationships"""
        violations = []
        
        for rule_id, rule in self.relationship_rules.items():
            trigger_entities = rule["trigger_entities"]
            
            # Get all instances of trigger entities
            entity_instances = []
            for entity_type in trigger_entities:
                if entity_type in self.detected_entities:
                    entity_instances.extend(self.detected_entities[entity_type])
            
            if not entity_instances:
                continue
                
            # Check relationship conditions
            if rule["condition"] == "co-occurrence" and len(entity_instances) >= 2:
                # Check if entities are within max_distance
                max_distance = rule.get("max_distance", float("inf"))
                
                # Sort entities by position
                sorted_entities = sorted(entity_instances, key=lambda e: e["start"])
                
                # Check distances between consecutive entities
                for i in range(len(sorted_entities) - 1):
                    distance = sorted_entities[i+1]["start"] - sorted_entities[i]["end"]
                    if distance <= max_distance:
                        violations.append({
                            "rule_id": rule_id,
                            "description": rule["description"],
                            "severity": rule["severity"],
                            "entities": [sorted_entities[i], sorted_entities[i+1]],
                            "type": "entity_relationship"
                        })
            
            elif rule["condition"] == "requires_disclaimer":
                import re
                # Check if disclaimer is present
                disclaimer_pattern = rule.get("disclaimer_pattern", "")
                if disclaimer_pattern and not re.search(disclaimer_pattern, text, re.IGNORECASE):
                    # If entities present but no disclaimer
                    if entity_instances:
                        violations.append({
                            "rule_id": rule_id,
                            "description": rule["description"],
                            "severity": rule["severity"],
                            "entity_count": len(entity_instances),
                            "type": "missing_disclaimer"
                        })
        
        return violations
    
    def _check_domain_specific_rules(self, text, domain=None):
        """Check domain-specific compliance rules"""
        violations = []
        
        if not domain or domain not in self.domain_specific_rules:
            return violations
        
        domain_rules = self.domain_specific_rules[domain]
        
        # Check for required disclaimers
        if domain_rules.get("disclaimers_required", False):
            import re
            disclaimer_pattern = domain_rules.get("disclaimer_pattern", "")
            if disclaimer_pattern and not re.search(disclaimer_pattern, text, re.IGNORECASE):
                violations.append({
                    "rule_id": f"{domain}_disclaimer_required",
                    "description": f"Content in {domain} domain requires appropriate disclaimers",
                    "severity": "medium",
                    "type": "domain_requirement"
                })
        
        # Check for prohibited phrases
        prohibited_phrases = domain_rules.get("prohibited_phrases", [])
        for phrase in prohibited_phrases:
            if phrase.lower() in text.lower():
                violations.append({
                    "rule_id": f"{domain}_prohibited_phrase",
                    "description": f"Content contains prohibited phrase: '{phrase}'",
                    "severity": "high",
                    "type": "prohibited_content",
                    "phrase": phrase
                })
        
        return violations
    
    def _check_framework_rules(self, text, framework_ids, context=None):
        """Check compliance against specific regulatory frameworks"""
        # In a production system, this would query framework-specific rules
        # This is a placeholder implementation
        return []
    
    def _calculate_compliance_score(self, violations):
        """Calculate compliance score based on violations"""
        if not violations:
            return 1.0
        
        # Calculate weighted score based on violation severity
        severity_weights = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.6,
            "critical": 1.0
        }
        
        total_penalty = sum(severity_weights.get(v.get("severity", "medium"), 0.3) for v in violations)
        
        # Cap the penalty at 1.0
        total_penalty = min(1.0, total_penalty)
        
        return 1.0 - total_penalty


class Framework:
    """Simple framework implementation for example purposes"""
    def __init__(self, id, name):
        self.id = id
        self.name = name
    
    def get_constraints(self, context=None):
        """Get constraints from this framework"""
        # Placeholder constraints
        if self.id == "FINREG":
            return [
                {
                    "id": "financial_advice_disclaimer",
                    "type": "required_disclaimer",
                    "description": "Financial content requires appropriate disclaimers",
                    "severity": "medium"
                },
                {
                    "id": "investment_claims",
                    "type": "prohibited_content",
                    "description": "Avoid guarantees about investment returns",
                    "terms": ["guaranteed returns", "risk-free investment"],
                    "severity": "high"
                }
            ]
        elif self.id == "HIPAA":
            return [
                {
                    "id": "phi_protection",
                    "type": "entity_protection",
                    "description": "Protected health information must be safeguarded",
                    "entities": ["PHI"],
                    "severity": "high"
                },
                {
                    "id": "medical_advice_disclaimer",
                    "type": "required_disclaimer",
                    "description": "Medical content requires appropriate disclaimers",
                    "severity": "medium"
                }
            ]
        else:
            return [
                {
                    "id": "harmful_content",
                    "type": "prohibited_content",
                    "description": "Content must not include harmful instructions",
                    "terms": ["how to hack", "how to steal"],
                    "severity": "high"
                }
            ]


async def example_usage():
    """
    Example usage of the compliant language model processor.
    
    This demonstrates how to set up and use the compliance components
    to generate text that adheres to regulatory requirements.
    """
    # Configure compliance components
    compliance_config = {
        "strict_mode": True,
        "temperature": 0.7,
        "prohibited_prompt_terms": ["pornography", "illegal activities", "hacking instructions"],
        "entity_sensitivity": {
            "PII": "high",
            "PHI": "high",
            "LOCATION": "medium"
        }
    }
    
    # Initialize compliance monitor
    compliance_monitor = SemanticComplianceMonitor(compliance_config)
    
    # Initialize language model (placeholder)
    language_model = "gpt-4"  # In a real implementation, this would be a model instance
    
    # Initialize processor
    processor = GatewayCompliantLanguageModelProcessor(
        language_model=language_model,
        compliance_monitor=compliance_monitor,
        compliance_config=compliance_config
    )
    
    # Example prompts
    prompts = [
        "Can you give me financial advice about the best stocks to invest in?",
        "Tell me about common symptoms of the cold and what to do about them.",
        "Write a story about a person who lives in New York and their daily routine."
    ]
    
    # Process each prompt with different contexts
    contexts = [
        {"domain": "finance"},
        {"domain": "healthcare"},
        {"domain": "general"}
    ]
    
    # Generate compliant responses
    for prompt, context in zip(prompts, contexts):
        print(f"\nProcessing prompt in {context['domain']} domain:")
        print(f"Prompt: {prompt}")
        
        result = await processor.generate_compliant_text(
            prompt=prompt,
            context=context,
            max_tokens=100,
            compliance_mode="relaxed"  # Allow post-processing fixes
        )
        
        if result["is_compliant"]:
            print(f"Generated compliant text (score: {result['compliance_score']:.2f}):")
            print(result["text"])
            if result.get("modified", False):
                print("Note: Text was modified to ensure compliance")
        else:
            print("Failed to generate compliant text:")
            print(f"Compliance error: {result.get('compliance_error', 'Unknown error')}")
            if "violations" in result:
                print("Violations:")
                for violation in result["violations"]:
                    print(f"- {violation.get('description', 'Unknown violation')}")
        
        print(f"Generation time: {result['generation_time']:.2f}s")
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Requests processed: {processor.metrics['requests_processed']}")
    print(f"Compliant responses: {processor.metrics['compliant_responses']}")
    print(f"Rejected requests: {processor.metrics['rejected_requests']}")
    print(f"Average response time: {processor.metrics['avg_response_time']:.2f}s")


# Run the example
import asyncio
asyncio.run(example_usage())