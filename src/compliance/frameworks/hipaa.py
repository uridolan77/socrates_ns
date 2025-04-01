import re

class HIPAAComplianceProcessor:
    """HIPAA compliance verification processor"""
    
    def __init__(self):
        """Initialize HIPAA compliance processor"""
        # Define PHI types and patterns
        self.phi_types = {
            "name": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
            "mrn": r'\bMRN\s*:?\s*\d{5,10}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "address": r'\b\d+\s+[A-Za-z]+\s+(?:St|Ave|Rd|Blvd|Dr|Lane|Way)(?:\s+[A-Za-z]+)?(?:\s+[A-Za-z]+)?,\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b',
            "phone": r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "date": r'\b(?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12][0-9]|3[01])[/.-](?:19|20)\d{2}\b',
            "age": r'\bage\s*:?\s*\d{1,3}\b|\b\d{1,3}\s+years\s+old\b',
            "facility": r'\b(?:hospital|clinic|center|medical center|healthcare facility|practice)\b'
        }
        
        # Define HIPAA rules
        self.hipaa_rules = [
            {
                "id": "phi_disclosure",
                "title": "PHI Disclosure",
                "description": "Protected Health Information should not be disclosed without proper authorization",
                "verification": self._verify_phi_disclosure
            },
            {
                "id": "minimum_necessary",
                "title": "Minimum Necessary Rule",
                "description": "Only the minimum necessary PHI should be used or disclosed",
                "verification": self._verify_minimum_necessary
            },
            {
                "id": "authorization",
                "title": "Patient Authorization",
                "description": "PHI disclosure requires patient authorization unless an exception applies",
                "verification": self._verify_authorization
            },
            {
                "id": "safeguards",
                "title": "Administrative Safeguards",
                "description": "Appropriate safeguards must be in place to protect PHI",
                "verification": self._verify_safeguards
            }
        ]
    
    def verify_compliance(self, text, context=None):
        """
        Verify HIPAA compliance of the provided text
        
        Args:
            text: Text to verify
            context: Additional context like authorization status
            
        Returns:
            Dictionary with compliance results
        """
        if not text:
            return {"compliant": True, "score": 1.0, "details": []}
        
        # Initialize results
        results = {
            "compliant": True,
            "score": 1.0,
            "details": []
        }
        
        # Check each HIPAA rule
        rule_results = []
        for rule in self.hipaa_rules:
            rule_compliant, rule_score, rule_reason = rule["verification"](text, context)
            
            rule_results.append({
                "rule_id": rule["id"],
                "title": rule["title"],
                "compliant": rule_compliant,
                "score": rule_score,
                "reason": rule_reason
            })
            
            # Update overall compliance
            if not rule_compliant:
                results["compliant"] = False
        
        # Calculate weighted average compliance score
        if rule_results:
            total_score = sum(r["score"] for r in rule_results)
            results["score"] = total_score / len(rule_results)
        
        results["details"] = rule_results
        return results
    
    def _verify_phi_disclosure(self, text, context=None):
        """
        Verify if the text contains PHI disclosure
        
        Args:
            text: Text to verify
            context: Additional context
            
        Returns:
            Tuple of (compliant, score, reason)
        """
        # Check for authorization
        has_authorization = context.get("has_authorization", False) if context else False
        
        # Detect PHI in text
        phi_found = {}
        for phi_type, pattern in self.phi_types.items():
            matches = re.findall(pattern, text)
            if matches:
                phi_found[phi_type] = matches
        
        # Check if PHI is disclosed
        if not phi_found:
            return True, 1.0, "No PHI detected in text"
        
        # Check if disclosure is authorized
        if has_authorization:
            return True, 0.9, f"PHI disclosure is authorized. Found: {', '.join(phi_found.keys())}"
        
        # PHI disclosed without authorization
        return False, 0.3, f"Unauthorized PHI disclosure detected: {', '.join(phi_found.keys())}"
    
    def _verify_minimum_necessary(self, text, context=None):
        """
        Verify if only minimum necessary PHI is disclosed
        
        Args:
            text: Text to verify
            context: Additional context
            
        Returns:
            Tuple of (compliant, score, reason)
        """
        # Get required PHI types for the purpose
        required_phi = context.get("required_phi", []) if context else []
        
        # Detect PHI in text
        phi_found = {}
        for phi_type, pattern in self.phi_types.items():
            matches = re.findall(pattern, text)
            if matches:
                phi_found[phi_type] = matches
        
        # If no PHI found, compliance is perfect
        if not phi_found:
            return True, 1.0, "No PHI detected in text"
        
        # If no required PHI specified, we can't determine minimum necessary
        if not required_phi:
            # If 3 or fewer PHI types, consider it reasonable
            if len(phi_found) <= 3:
                return True, 0.8, f"Limited PHI disclosure ({len(phi_found)} types)"
            else:
                return False, 0.6, f"Extensive PHI disclosure ({len(phi_found)} types) without specified requirements"
        
        # Check if disclosed PHI exceeds required PHI
        excessive_phi = [phi for phi in phi_found.keys() if phi not in required_phi]
        if excessive_phi:
            return False, 0.5, f"Excessive PHI disclosure beyond requirements: {', '.join(excessive_phi)}"
        
        return True, 0.9, "Only necessary PHI is disclosed"
    
    def _verify_authorization(self, text, context=None):
        """
        Verify if proper authorization exists for PHI disclosure
        
        Args:
            text: Text to verify
            context: Additional context
            
        Returns:
            Tuple of (compliant, score, reason)
        """
        # Check for explicit authorization
        has_authorization = context.get("has_authorization", False) if context else False
        
        # Detect PHI in text
        phi_found = {}
        for phi_type, pattern in self.phi_types.items():
            matches = re.findall(pattern, text)
            if matches:
                phi_found[phi_type] = matches
        
        # If no PHI found, no authorization needed
        if not phi_found:
            return True, 1.0, "No PHI detected, authorization not required"
        
        # Check if text mentions authorization
        authorization_keywords = [
            "authorized", "authorized by patient", "consent", "patient consented",
            "permission", "approved disclosure", "with approval"
        ]
        mentions_authorization = any(keyword in text.lower() for keyword in authorization_keywords)
        
        # Check for exception keywords
        exception_keywords = [
            "emergency", "required by law", "public health", "health oversight",
            "judicial proceeding", "law enforcement", "research", "serious threat"
        ]
        mentions_exception = any(keyword in text.lower() for keyword in exception_keywords)
        
        # Determine compliance
        if has_authorization:
            return True, 1.0, "Valid authorization exists for PHI disclosure"
        elif mentions_authorization:
            return True, 0.8, "Text indicates authorization exists, but not verified"
        elif mentions_exception:
            return True, 0.7, "Text indicates an exception to authorization requirement may apply"
        else:
            return False, 0.3, "PHI disclosure without indication of authorization"
    
    def _verify_safeguards(self, text, context=None):
        """
        Verify if appropriate safeguards are mentioned
        
        Args:
            text: Text to verify
            context: Additional context
            
        Returns:
            Tuple of (compliant, score, reason)
        """
        # Check for safeguard indications in text
        safeguard_keywords = [
            "encrypted", "secure", "protected", "confidential",
            "access control", "authorization", "authentication",
            "security", "privacy", "safeguard"
        ]
        
        safeguards_mentioned = [kw for kw in safeguard_keywords if kw in text.lower()]
        
        # Detect PHI in text
        phi_present = False
        for pattern in self.phi_types.values():
            if re.search(pattern, text):
                phi_present = True
                break
        
        # If no PHI, safeguards aren't required
        if not phi_present:
            return True, 1.0, "No PHI detected, safeguards not required"
        
        # Check if safeguards are mentioned
        if safeguards_mentioned:
            if len(safeguards_mentioned) >= 3:
                return True, 0.9, f"Multiple safeguards mentioned: {', '.join(safeguards_mentioned[:3])}"
            else:
                return True, 0.7, f"Limited safeguards mentioned: {', '.join(safeguards_mentioned)}"
        else:
            # Context may include security environment info
            secure_environment = context.get("secure_environment", False) if context else False
            if secure_environment:
                return True, 0.8, "No safeguards mentioned, but secure environment indicated in context"
            else:
                return False, 0.5, "PHI present without mentioned safeguards"