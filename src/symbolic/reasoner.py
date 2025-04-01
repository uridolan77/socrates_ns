class SymbolicReasoner:
    """
    Symbolic reasoning engine for compliance verification that translates 
    regulatory requirements into formal logic representations.
    """
    def __init__(self, config=None, knowledge_base=None):
        self.config = config or {}
        self.knowledge_base = knowledge_base
        self.rules = self._load_rules()
        self.rule_cache = {}  # Cache for rule applicability
        
    def _load_rules(self):
        """
        Load symbolic reasoning rules from configured source.
        
        Notes:
            In a production system, this would load rules from a database, 
            file system, or API endpoint. For this implementation, we use 
            sample rules defined in code.
        
        Returns:
            Dictionary mapping rule IDs to rule definitions
        """
        # This would fetch from a database or file in a real system
        # Using sample rules for demonstration
        rules = {
            "gdpr_data_minimization": {
                "id": "gdpr_data_minimization",
                "framework": "GDPR",
                "description": "Personal data must be adequate, relevant and limited to what is necessary",
                "severity": "high",
                "symbolic_representation": "∀x (PersonalData(x) → (Adequate(x) ∧ Relevant(x) ∧ Necessary(x)))",
                "pattern": r"\b(?:collect|store|process|use)\b.{0,30}\b(?:all|every|any|extensive)\b.{0,30}\b(?:data|information)\b",
                "keywords": ["collect", "store", "process", "extensive", "all data"]
            },
            "gdpr_consent": {
                "id": "gdpr_consent",
                "framework": "GDPR",
                "description": "Processing based on consent must be demonstrably given",
                "severity": "critical",
                "symbolic_representation": "∀x (ProcessingActivity(x) ∧ BasedOnConsent(x) → ∃y (Consent(y) ∧ Demonstrable(y) ∧ GivenFor(y, x)))",
                "pattern": r"\b(?:without|no|implicit)\b.{0,30}\b(?:consent|permission|authorization)\b",
                "keywords": ["consent", "permission", "authorization", "explicit"]
            },
            "hipaa_phi_disclosure": {
                "id": "hipaa_phi_disclosure",
                "framework": "HIPAA",
                "description": "Protected health information should only be disclosed with authorization",
                "severity": "critical",
                "symbolic_representation": "∀x (PHI(x) → (Disclosed(x) → ∃y (Authorization(y) ∧ CoversDisclosure(y, x))))",
                "pattern": r"\b(?:health|medical|patient).{0,50}\b(?:disclose|share|reveal)\b",
                "keywords": ["PHI", "health information", "medical data", "disclose", "share"]
            }
        }
        
        print(f"Loaded {len(rules)} symbolic reasoning rules")
        return rules
        
    def evaluate_compliance(self, text, applicable_rules, context=None):
        """
        Evaluate whether text complies with applicable rules using symbolic reasoning.
        
        Notes:
            A production implementation would use formal logic reasoning engines.
            This simplified version uses pattern matching and keyword detection.
        
        Args:
            text: Text to evaluate
            applicable_rules: List of applicable rule IDs
            context: Optional contextual information
            
        Returns:
            Dict with compliance evaluation results
        """
        import re
        
        if not text or not applicable_rules:
            return {
                "is_compliant": True,
                "violations": [],
                "reasoning_steps": [],
                "compliance_score": 1.0
            }
            
        # Track violations and reasoning steps
        violations = []
        reasoning_steps = []
        
        # Check each applicable rule
        for rule_id in applicable_rules:
            rule = self.rules.get(rule_id)
            if not rule:
                continue
                
            # This simplified implementation uses pattern matching
            # A real implementation would parse the symbolic representation and apply formal logic
            
            # Check for pattern matches
            pattern = rule.get("pattern")
            if pattern:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    violations.append({
                        "rule_id": rule_id,
                        "severity": rule.get("severity", "medium"),
                        "description": rule.get("description", ""),
                        "matched_text": match.group(0),
                        "position": (match.start(), match.end())
                    })
                    
                    reasoning_steps.append({
                        "rule_id": rule_id,
                        "reasoning": f"Pattern '{pattern}' matched at position {match.start()}: '{match.group(0)}'",
                        "conclusion": "Rule violation detected"
                    })
            
            # Check for keywords (as a backup detection method)
            if not any(v["rule_id"] == rule_id for v in violations):
                keywords = rule.get("keywords", [])
                matched_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
                
                if matched_keywords:
                    violations.append({
                        "rule_id": rule_id,
                        "severity": rule.get("severity", "medium"),
                        "description": rule.get("description", ""),
                        "matched_keywords": matched_keywords
                    })
                    
                    reasoning_steps.append({
                        "rule_id": rule_id,
                        "reasoning": f"Keywords detected: {', '.join(matched_keywords)}",
                        "conclusion": "Potential rule violation based on keywords"
                    })
            
            # No violation detected, add compliance reasoning step
            if not any(v["rule_id"] == rule_id for v in violations):
                reasoning_steps.append({
                    "rule_id": rule_id,
                    "reasoning": "No violation patterns or keywords detected",
                    "conclusion": "Rule compliance verified"
                })
        
        # Calculate compliance score based on violations
        if not violations:
            compliance_score = 1.0
        else:
            # Weight by severity
            severity_weights = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            violation_penalty = sum(severity_weights.get(v["severity"], 0.5) for v in violations)
            compliance_score = max(0.0, 1.0 - (violation_penalty / len(applicable_rules)))
        
        return {
            "is_compliant": len(violations) == 0,
            "violations": violations,
            "reasoning_steps": reasoning_steps,
            "compliance_score": compliance_score
        }
    
    def get_applicable_rules(self, text, frameworks=None, context=None):
        """
        Determine which rules are applicable to the given text and context.
        
        Notes:
            In a production system, this would use more sophisticated logic
            including semantic analysis and contextual inference to determine
            rule applicability. This implementation uses simpler heuristics.
        
        Args:
            text: Text to analyze
            frameworks: Optional list of regulatory frameworks to consider
            context: Optional contextual information
            
        Returns:
            List of applicable rule IDs
        """
        # Generate cache key based on text, frameworks, and context
        cache_key = self._generate_cache_key(text, frameworks, context)
        
        # Check cache first
        if cache_key in self.rule_cache:
            return self.rule_cache[cache_key]
        
        applicable_rules = []
        
        # If specific frameworks are provided, only consider rules from those frameworks
        framework_filter = set(frameworks) if frameworks else None
        
        for rule_id, rule in self.rules.items():
            # Apply framework filter if specified
            if framework_filter and rule.get("framework") not in framework_filter:
                continue
            
            # Check domain applicability
            if context and "domain" in context:
                rule_domains = rule.get("applicable_domains", ["general"])
                if "all" not in rule_domains and context["domain"] not in rule_domains:
                    continue
            
            # Basic relevance check: look for any rule keywords in the text
            # A real implementation would use semantic similarity, context analysis, etc.
            keywords = rule.get("keywords", [])
            
            # If any keyword is found in the text, consider the rule applicable
            if any(keyword.lower() in text.lower() for keyword in keywords):
                applicable_rules.append(rule_id)
                continue
            
            # Check the rule pattern for potential matches
            pattern = rule.get("pattern")
            if pattern:
                import re
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        applicable_rules.append(rule_id)
                        continue
                except re.error:
                    # Log pattern error but continue
                    print(f"Invalid regex pattern in rule {rule_id}: {pattern}")
            
            # More sophisticated rule applicability logic would be implemented here
            # For example, checking entity types, semantic similarity, etc.
        
        # Sort rules by severity for prioritized evaluation
        applicable_rules.sort(
            key=lambda rule_id: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                self.rules.get(rule_id, {}).get("severity", "medium"), 2
            )
        )
        
        # Cache the result
        self.rule_cache[cache_key] = applicable_rules
        
        return applicable_rules
    
    def _generate_cache_key(self, text, frameworks, context):
        """Generate a cache key for rule applicability caching"""
        import hashlib
        
        # Use first 100 chars of text
        text_sample = text[:100] if text else ""
        
        # Create a string representation of frameworks
        frameworks_str = ",".join(sorted(frameworks)) if frameworks else "all"
        
        # Extract relevant context fields (domain and content_type)
        context_str = ""
        if context:
            domain = context.get("domain", "general")
            content_type = context.get("content_type", "text")
            context_str = f"{domain}:{content_type}"
        
        # Combine and hash
        combined = f"{text_sample}|{frameworks_str}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()