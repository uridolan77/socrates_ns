import collections
import numpy as np
import json
import re
import logging
import traceback

class CosineSimilarityCalculator:
    """Simple cosine similarity calculator"""
    def calculate(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
            
        # Ensure vectors are normalized
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        return np.dot(vec1_norm, vec2_norm)

class ComplianceBoundaryClassifier:
    """Classifier for compliance boundaries in regulatory space"""
    def __init__(self, regulatory_dim, num_frameworks):
        self.dim = regulatory_dim
        self.num_frameworks = num_frameworks
        
        # Initialize classifier parameters
        self.boundaries = self._initialize_boundaries()
        
    def check_compliance(self, embedding, framework_id):
        """Check if embedding complies with framework boundaries"""
        if framework_id not in self.boundaries:
            return True, 1.0, None
            
        boundary = self.boundaries[framework_id]
        
        # Calculate distance to boundary
        distance = self._calculate_boundary_distance(embedding, boundary)
        
        # Determine compliance and violation type
        if distance > 0:
            return True, distance, None
        else:
            # Negative distance indicates non-compliance
            violation_type = self._determine_violation_type(embedding, boundary, framework_id)
            return False, abs(distance), violation_type
    
    def _initialize_boundaries(self):
        """Initialize compliance boundaries for frameworks"""
        # In a real system, these would be learned from labeled data
        # Simple placeholder implementation
        boundaries = {}
        
        # Create random hyperplanes for each framework
        for i in range(self.num_frameworks):
            framework_id = f"framework_{i}"
            
            # Random normal vector for hyperplane
            normal = np.random.randn(self.dim)
            normal = normal / np.linalg.norm(normal)
            
            # Random offset (distance from origin)
            offset = np.random.uniform(0.1, 0.5)
            
            boundaries[framework_id] = {
                "normal": normal,
                "offset": offset,
                "violation_types": self._initialize_violation_types()
            }
            
        return boundaries
    
    def _initialize_violation_types(self):
        """Initialize violation type detectors"""
        # In a real system, these would be more sophisticated
        # Simple placeholder
        return ["content_violation", "disclosure_violation", "procedural_violation"]
    
    def _calculate_boundary_distance(self, embedding, boundary):
        """Calculate signed distance to boundary"""
        # Ensure embedding is normalized
        embedding = embedding / np.linalg.norm(embedding)
        
        # Calculate signed distance (positive = compliant side, negative = non-compliant)
        return np.dot(embedding, boundary["normal"]) - boundary["offset"]
    
    def _determine_violation_type(self, embedding, boundary, framework_id):
        """Determine type of compliance violation"""
        # Simple placeholder implementation
        # In a real system, this would use more sophisticated classification
        
        # Return random violation type for demonstration
        return np.random.choice(boundary["violation_types"])
    
class LRUCache:
    """Enhanced LRU Cache with size tracking and statistics"""
    
    def __init__(self, maxsize=128, stats_window=100):
        self.cache = {}
        self.maxsize = maxsize
        self.order = []
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_rate": 0.0,
            "size_bytes": 0
        }
        self.access_history = []  # Track last N accesses
        self.stats_window = stats_window
    
    def get(self, key):
        """Get an item from cache with stats tracking"""
        if key in self.cache:
            # Move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            
            # Update stats
            self.stats["hits"] += 1
            self._record_access(key, True)
            
            return self.cache[key]
        else:
            # Update stats
            self.stats["misses"] += 1
            self._record_access(key, False)
            
            return None
    
    def __setitem__(self, key, value):
        """Add/update an item in the cache"""
        if key in self.cache:
            # Update existing entry
            old_size = self._get_item_size(self.cache[key])
            new_size = self._get_item_size(value)
            self.stats["size_bytes"] += (new_size - old_size)
            
            # Update
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # Add new entry
            self.cache[key] = value
            self.order.append(key)
            self.stats["size_bytes"] += self._get_item_size(value)
            
            # Check size limit
            while len(self.cache) > self.maxsize:
                # Remove least recently used
                lru_key = self.order.pop(0)
                self.stats["size_bytes"] -= self._get_item_size(self.cache[lru_key])
                del self.cache[lru_key]
                self.stats["evictions"] += 1
    
    def _record_access(self, key, hit):
        """Record access for hit rate calculation"""
        self.access_history.append(hit)
        if len(self.access_history) > self.stats_window:
            self.access_history.pop(0)
        
        # Update hit rate
        if self.access_history:
            self.stats["hit_rate"] = sum(self.access_history) / len(self.access_history)
    
    def _get_item_size(self, item):
        """Estimate memory size of cached item"""
        import sys
        
        # For strings
        if isinstance(item, str):
            return len(item) * 2  # Approximate for Python strings
        
        # For tensors
        if hasattr(item, 'element_size') and hasattr(item, 'nelement'):
            return item.element_size() * item.nelement()
        
        # For dictionaries
        if isinstance(item, dict):
            return sum(self._get_item_size(k) + self._get_item_size(v) 
                      for k, v in item.items())
        
        # For lists
        if isinstance(item, list):
            return sum(self._get_item_size(x) for x in item)
        
        # Default approximation
        return sys.getsizeof(item)

class TokenConfidenceTracker:
    """Tracks confidence levels of token predictions"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.token_history = []
        self.uncertainty_threshold = 0.3
        
    def update(self, logits, text):
        """Update confidence tracking with new logits"""
        # Convert logits to probabilities
        probs = self._softmax(logits)
        
        # Get top tokens and their probabilities
        top_indices = np.argsort(probs)[-5:]  # Top 5 tokens
        top_probs = probs[top_indices]
        
        # Calculate confidence metrics
        entropy = -np.sum(top_probs * np.log(top_probs + 1e-10))
        max_prob = np.max(top_probs)
        
        # Update history
        self.token_history.append({
            "entropy": entropy,
            "max_prob": max_prob,
            "top_tokens": top_indices.tolist(),
            "text_position": len(text)
        })
        
        # Keep fixed window size
        if len(self.token_history) > self.window_size:
            self.token_history.pop(0)
    
    def get_uncertain_tokens(self):
        """Get tokens with high uncertainty"""
        uncertain_tokens = set()
        
        for entry in self.token_history:
            if entry["entropy"] > self.uncertainty_threshold or entry["max_prob"] < (1 - self.uncertainty_threshold):
                uncertain_tokens.update(entry["top_tokens"])
                
        return uncertain_tokens
    
    def _softmax(self, logits):
        """Convert logits to probabilities using softmax"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

class ViolationAnalyzer:
    """Analyzes compliance violations and generates remediation suggestions"""
    def __init__(self, config):
        self.config = config
        
    def generate_remediation(self, violations, content_type, context=None):
        """Generate remediation suggestions for violations"""
        if not violations:
            return []
            
        suggestions = []
        
        for violation in violations:
            rule_id = violation.get("rule_id", "unknown")
            severity = violation.get("severity", "medium")
            
            # Generate suggestion based on violation type
            suggestion = {
                "rule_id": rule_id,
                "severity": severity,
                "suggestion": self._get_suggestion_for_rule(rule_id, content_type)
            }
            
            suggestions.append(suggestion)
            
        return suggestions
    
    def _get_suggestion_for_rule(self, rule_id, content_type):
        """Get remediation suggestion for specific rule"""
        # In a real system, this would use rule metadata to generate appropriate suggestions
        # Simple placeholder implementation
        if content_type == "prompt":
            return "Consider rephrasing your prompt to avoid potentially problematic content."
        else:
            return "The generated content may need modification to ensure regulatory compliance."

class SensitiveTokenDetector:
    """Detects sensitive tokens that may indicate compliance issues"""
    def __init__(self, config):
        self.config = config
        self.sensitive_patterns = config.get("sensitive_patterns", [])
        
    def detect_sensitive_tokens(self, tokens, context):
        """Detect tokens that may indicate sensitive content"""
        # Implementation would detect tokens related to sensitive topics
        # This is a placeholder
        return []

class TextSemanticAnalyzer:
    """Analyzes semantic content of text for compliance monitoring"""
    def __init__(self, config):
        self.config = config
        
    def analyze(self, text):
        """Analyze text semantics"""
        # In a real implementation, this would use language model or embeddings
        # Simplified placeholder implementation
        return {
            "embedding": np.random.randn(64),  # Random embedding vector
            "topics": {"general": 0.8},  # Topic distribution
            "sentiment": 0.0,  # Neutral sentiment
            "formality": 0.5  # Medium formality
        }

class OptimizedSemanticAnalyzer:
    """Optimized semantic analysis for text"""
    def __init__(self, config):
        self.config = config
        
    def analyze(self, text):
        """Analyze text for semantic content"""
        # In a real implementation, this would use a language model
        # Placeholder implementation
        return {
            "embedding": np.random.randn(64),
            "topics": {"general": 0.8},
            "sentiment": 0.0,
            "entities": []
        }

class TopicModel:
    """Simple topic modeling for text"""
    def __init__(self, config):
        self.config = config
        
    def extract_topics(self, text):
        """Extract topics from text"""
        # In a real implementation, this would use a trained topic model
        # Simplified placeholder implementation
        return {"general": 0.8}

class EntityTracker:
    """Tracks entities mentioned in text"""
    def __init__(self, config):
        self.config = config
        
    def extract_entities(self, text):
        """Extract entities from text"""
        # In a real implementation, this would use NER
        # Simplified placeholder implementation
        return []

class RelationTracker:
    """Tracks relations between entities"""
    def __init__(self, config):
        self.config = config
        
    def extract_relations(self, text, entities):
        """Extract relations between entities"""
        # In a real implementation, this would use relation extraction
        # Simplified placeholder implementation
        return {}

class ContextualRuleInterpreter:
    """Interprets rules in context"""
    def __init__(self, config):
        self.config = config
        
    def get_contextual_rules(self, content_type, context=None):
        """Get rules applicable in current context"""
        # This is a placeholder implementation
        return []
    

class ProofFormatter:
    """Formats compliance proofs for different output formats"""
    def format_as_json(self, proof_trace):
        """Format proof trace as JSON"""
        return json.dumps(proof_trace, indent=2)
        
    def format_as_text(self, proof_trace):
        """Format proof trace as human-readable text"""
        text = f"Compliance Verification Proof\n"
        text += f"===========================\n\n"
        
        # Add input summary
        text += f"Input: {proof_trace.get('input', 'Unknown input')[:100]}...\n\n"
        
        # Add frameworks
        text += f"Frameworks: {', '.join(proof_trace.get('frameworks', []))}\n"
        text += f"Mode: {proof_trace.get('mode', 'standard')}\n\n"
        
        # Add steps
        text += "Verification Steps:\n"
        for i, step in enumerate(proof_trace.get('steps', []), 1):
            text += f"{i}. {step.get('description', 'Step')}\n"
            if 'intermediate_conclusion' in step:
                text += f"   Conclusion: {step['intermediate_conclusion']}\n"
            text += "\n"
            
        # Add final conclusion
        conclusion = proof_trace.get('conclusion', {})
        text += f"Final Result: {'Compliant' if conclusion.get('is_compliant', False) else 'Non-compliant'}\n"
        text += f"Compliance Score: {conclusion.get('compliance_score', 0.0):.2f}\n"
        text += f"Justification: {conclusion.get('justification', 'No justification provided')}\n"
        
        return text
        
    def format_as_html(self, proof_trace):
        """Format proof trace as HTML"""
        # Simplified HTML format implementation
        html = "<div class='compliance-proof'>\n"
        html += f"<h3>Compliance Verification Result</h3>\n"
        
        # Add conclusion
        conclusion = proof_trace.get('conclusion', {})
        is_compliant = conclusion.get('is_compliant', False)
        html += f"<div class='result {is_compliant and 'compliant' or 'non-compliant'}'>\n"
        html += f"<p><strong>Result:</strong> {'Compliant' if is_compliant else 'Non-compliant'}</p>\n"
        html += f"<p><strong>Score:</strong> {conclusion.get('compliance_score', 0.0):.2f}</p>\n"
        html += f"<p><strong>Justification:</strong> {conclusion.get('justification', 'No justification provided')}</p>\n"
        html += "</div>\n"
        
        html += "</div>"
        return html

class TopicDetector:
    """Detects topics in text"""
    def __init__(self, config):
        self.config = config
        
    def detect_topics(self, text):
        """Detect topics in text"""
        # In a real implementation, this would use a topic model
        # Placeholder implementation
        return {"general": 0.8}


class RegexPatternMatcher:
    """Matches regex patterns in content"""
    def __init__(self, patterns):
        self.patterns = self._compile_patterns(patterns)
        
    def check_patterns(self, content, context=None):
        """Check content against patterns"""
        # In a real implementation, this would check against compiled patterns
        # Placeholder implementation
        return {
            "is_compliant": True,
            "filtered_input": content,
            "issues": []
        }
        
    def _compile_patterns(self, patterns):
        """Compile regex patterns"""
        compiled = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern))
            except Exception as e:
                logging.warning(f"Invalid pattern: {pattern}, Error: {str(e)}")
        return compiled

class ComplianceTopicAnalyzer:
    """Analyzes topics for compliance issues"""
    def __init__(self, config):
        self.config = config
        
    def analyze_topics(self, content, context=None):
        """Analyze content topics for compliance"""
        # In a real implementation, this would check topics against sensitive areas
        # Placeholder implementation
        return {
            "is_compliant": True,
            "filtered_input": content,
            "issues": []
        }

class SensitiveDataDetector:
    """Detects sensitive data in content"""
    def __init__(self, config):
        self.config = config
        
    def detect(self, content, context=None):
        """Detect sensitive data in content"""
        # In a real implementation, this would check for PII, etc.
        # Placeholder implementation
        return {
            "is_compliant": True,
            "filtered_input": content,
            "issues": []
        }
    
import re
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComplianceIssue:
    """Data class for storing compliance issues found during filtering."""
    rule_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    location: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentComplianceDetector:
    """
    Detects content that violates compliance policies.
    """
    def __init__(self, config):
        self.config = config
        self.policy_rules = config.get("policy_rules", {})
        self.compliance_threshold = config.get("compliance_threshold", 0.7)
        self.rules_enabled = config.get("enabled_rules", [])
        
    def check_compliance(self, text, context=None):
        """
        Check if the provided text complies with all defined policy rules.
        
        Args:
            text: Input text to check
            context: Optional context information
            
        Returns:
            Dict with compliance results
        """
        if not text:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        issues = []
        is_compliant = True
        
        # Process each enabled rule
        for rule_id in self.rules_enabled:
            if rule_id not in self.policy_rules:
                continue
                
            rule = self.policy_rules[rule_id]
            violation_found = self._check_rule(text, rule, context)
            
            if violation_found:
                is_compliant = False
                issues.append(ComplianceIssue(
                    rule_id=rule_id,
                    severity=rule.get("severity", "medium"),
                    description=rule.get("description", "Policy violation detected"),
                    metadata={"rule_name": rule.get("name", "")}
                ))
                
                # Stop checking if configured to break on first violation
                if self.config.get("break_on_first_violation", False):
                    break
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": text,
            "issues": [issue.__dict__ for issue in issues],
            "modified": False
        }
        
    def _check_rule(self, text, rule, context=None):
        """Check if text violates a specific rule."""
        rule_type = rule.get("type", "keyword")
        
        if rule_type == "keyword":
            keywords = rule.get("keywords", [])
            return any(keyword.lower() in text.lower() for keyword in keywords)
            
        elif rule_type == "regex":
            pattern = rule.get("pattern", "")
            try:
                return bool(re.search(pattern, text, re.IGNORECASE))
            except re.error:
                logging.error(f"Invalid regex pattern: {pattern}")
                return False
                
        elif rule_type == "custom":
            # Placeholder for custom rule implementation
            # In a real system, this might call an external API or model
            return False
            
        return False


class RegexPatternMatcher:
    """
    Matches text against predefined regex patterns for policy enforcement.
    """
    def __init__(self, patterns):
        self.patterns = []
        
        # Compile regex patterns
        for pattern_def in patterns:
            try:
                compiled = re.compile(pattern_def["pattern"], re.IGNORECASE)
                pattern_def["compiled"] = compiled
                self.patterns.append(pattern_def)
            except re.error as e:
                logging.error(f"Failed to compile regex pattern '{pattern_def.get('name')}': {str(e)}")
    
    def check_patterns(self, text, context=None):
        """
        Check text against configured regex patterns.
        
        Args:
            text: Input text to check
            context: Optional context information
            
        Returns:
            Dict with pattern matching results
        """
        if not text:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        issues = []
        is_compliant = True
        
        for pattern_def in self.patterns:
            compiled = pattern_def.get("compiled")
            if not compiled:
                continue
                
            matches = list(compiled.finditer(text))
            if matches:
                is_compliant = False
                for match in matches:
                    issues.append(ComplianceIssue(
                        rule_id=pattern_def.get("id", str(uuid.uuid4())),
                        severity=pattern_def.get("severity", "medium"),
                        description=pattern_def.get("description", "Pattern match detected"),
                        location={"start": match.start(), "end": match.end()},
                        metadata={"pattern_name": pattern_def.get("name", "")}
                    ))
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": text,
            "issues": [issue.__dict__ for issue in issues],
            "modified": False
        }


class ComplianceTopicAnalyzer:
    """
    Analyzes text for sensitive topics that may require compliance review.
    """
    def __init__(self, config):
        self.config = config
        self.sensitive_topics = config.get("sensitive_topics", [])
        self.topic_threshold = config.get("topic_threshold", 0.6)
        
    def analyze_topics(self, text, context=None):
        """
        Analyze text for sensitive topics.
        
        Args:
            text: Input text to analyze
            context: Optional context information
            
        Returns:
            Dict with topic analysis results
        """
        if not text or not self.sensitive_topics:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        issues = []
        is_compliant = True
        
        # Simple keyword-based topic detection
        # In a real system, this would likely use a more sophisticated ML model
        for topic in self.sensitive_topics:
            topic_id = topic.get("id", str(uuid.uuid4()))
            keywords = topic.get("keywords", [])
            threshold = topic.get("threshold", self.topic_threshold)
            
            # Count keyword occurrences
            keyword_count = sum(text.lower().count(keyword.lower()) for keyword in keywords)
            
            # Crude calculation of topic relevance
            topic_score = min(1.0, keyword_count / max(10, len(text.split()) / 5))
            
            if topic_score >= threshold:
                is_compliant = False
                issues.append(ComplianceIssue(
                    rule_id=topic_id,
                    severity=topic.get("severity", "medium"),
                    description=f"Sensitive topic detected: {topic.get('name', 'Unnamed')}",
                    metadata={"topic_score": topic_score, "topic_name": topic.get("name", "")}
                ))
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": text,
            "issues": [issue.__dict__ for issue in issues],
            "modified": False
        }


class SensitiveDataDetector:
    """
    Detects sensitive data patterns like PII in text.
    """
    def __init__(self, config):
        self.config = config
        
        # Initialize PII detection patterns
        self.pii_patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
            "phone": re.compile(r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        }
        
        # Add custom patterns from config
        custom_patterns = config.get("pii_patterns", {})
        for pattern_name, pattern_string in custom_patterns.items():
            try:
                self.pii_patterns[pattern_name] = re.compile(pattern_string)
            except re.error:
                logging.error(f"Invalid PII regex pattern for {pattern_name}: {pattern_string}")
    
    def detect(self, text, context=None):
        """
        Detect sensitive data in text.
        
        Args:
            text: Input text to analyze
            context: Optional context information
            
        Returns:
            Dict with detection results
        """
        if not text:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        issues = []
        is_compliant = True
        modified_text = text
        modified = False
        
        # Detect PII based on patterns
        for pii_type, pattern in self.pii_patterns.items():
            matches = list(pattern.finditer(text))
            
            for match in matches:
                is_compliant = False
                
                # Create issue
                issues.append(ComplianceIssue(
                    rule_id=f"pii_{pii_type}",
                    severity="high",
                    description=f"Detected {pii_type} in text",
                    location={"start": match.start(), "end": match.end()},
                    metadata={"pii_type": pii_type}
                ))
                
                # Redact PII if configured
                if self.config.get("redact_pii", False):
                    redaction = self.config.get("redaction_string", "[REDACTED]")
                    modified_text = modified_text[:match.start()] + redaction + modified_text[match.end():]
                    modified = True
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": modified_text,
            "issues": [issue.__dict__ for issue in issues],
            "modified": modified
        }