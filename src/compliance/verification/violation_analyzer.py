import logging
from collections import defaultdict, Counter
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Set
import json
from dataclasses import dataclass
import time
import textblob
from textblob import TextBlob
import spacy
import datetime
# import text  # Removed as it could not be resolved

@dataclass
class ViolationSummary:
    """Summary of compliance violations found during filtering."""
    violation_count: int
    severity_counts: Dict[str, int]  # Counts by severity level
    categories: Dict[str, int]       # Counts by violation category
    top_rules: List[Dict[str, Any]]  # Most frequently triggered rules
    primary_severity: str            # Most severe violation level
    timestamp: float                 # When the analysis was performed
    suggestion: Optional[str] = None # Suggested remediation


class ViolationAnalyzer:
    """
    Analyzes compliance violations to provide insights and remediation suggestions.
    
    This class processes violation data from various filter components, categorizes them,
    identifies patterns, generates reports, and provides suggestions for addressing violations.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Configure analyzer settings
        self.min_violations_for_pattern = config.get("min_violations_for_pattern", 3)
        self.severity_weights = config.get("severity_weights", {
            "critical": 100,
            "high": 50,
            "medium": 10,
            "low": 1
        })
        
        # Configure rule metadata
        self.rule_metadata = self._load_rule_metadata(config.get("rule_metadata", {}))
        
        # Configure remediation templates
        self.remediation_templates = config.get("remediation_templates", {})
        
        # Configure category mapping
        self.category_mapping = config.get("category_mapping", {})
        
        # Configure violation thresholds
        self.violation_thresholds = config.get("violation_thresholds", {
            "critical": 1,
            "high": 2,
            "medium": 5,
            "low": 10
        })
        
        # Historical tracking (could be connected to a database in a real system)
        self.historical_violations = defaultdict(list)
        self.violation_trends = defaultdict(list)
        
        # Pattern recognition settings
        self.pattern_recognition_enabled = config.get("pattern_recognition_enabled", True)
        self.pattern_matchers = self._initialize_pattern_matchers()
        
    def analyze_violations(self, filter_results, context=None):
        """
        Analyze compliance violations from filter results.
        
        Args:
            filter_results: Results from compliance filtering
            context: Optional context information
            
        Returns:
            Dict with analysis results
        """
        violations = filter_results.get("issues", [])
        
        if not violations:
            return {
                "violation_count": 0,
                "has_violations": False,
                "summary": None
            }
            
        # Extract violations by severity
        violations_by_severity = self._group_by_severity(violations)
        
        # Extract violations by category
        violations_by_category = self._group_by_category(violations)
        
        # Identify the most triggered rules
        top_rules = self._identify_top_rules(violations)
        
        # Determine primary severity
        primary_severity = self._determine_primary_severity(violations_by_severity)
        
        # Generate violation summary
        summary = ViolationSummary(
            violation_count=len(violations),
            severity_counts={severity: len(violations_list) for severity, violations_list in violations_by_severity.items()},
            categories={category: len(violations_list) for category, violations_list in violations_by_category.items()},
            top_rules=top_rules[:5],  # Top 5 rules
            primary_severity=primary_severity,
            timestamp=time.time()
        )
        
        # Generate remediation suggestion
        suggestion = self._generate_remediation_suggestion(violations, context, filter_results.get("filtered_input", ""))
        summary.suggestion = suggestion
        
        # Update historical tracking
        self._update_historical_data(violations, context)
        
        # Detect patterns if enabled
        patterns = {}
        if self.pattern_recognition_enabled:
            patterns = self._detect_violation_patterns(violations, context)
            
        # Create full analysis results
        analysis_results = {
            "violation_count": len(violations),
            "has_violations": True,
            "summary": summary.__dict__,
            "violations_by_severity": {k: [v.__dict__ if hasattr(v, '__dict__') else v for v in violations_by_severity[k]] for k in violations_by_severity},
            "violations_by_category": {k: [v.__dict__ if hasattr(v, '__dict__') else v for v in violations_by_category[k]] for k in violations_by_category},
            "patterns": patterns,
            "impact_assessment": self._assess_impact(violations, context)
        }
        
        return analysis_results
        
    def generate_violation_report(self, analysis_results, format="json"):
        """
        Generate a formatted report of violation analysis.
        
        Args:
            analysis_results: Results from analyze_violations
            format: Report format (json, text, html)
            
        Returns:
            Formatted report
        """
        if not analysis_results.get("has_violations", False):
            return "No compliance violations detected."
            
        summary = analysis_results.get("summary", {})
        
        if format == "json":
            # Return JSON format report
            return json.dumps(analysis_results, indent=2)
            
        elif format == "html":
            # Generate HTML report
            # In a real implementation, this would use a template engine
            html_parts = [
                "<html><head><title>Compliance Violation Report</title></head><body>",
                f"<h1>Compliance Violation Report</h1>",
                f"<h2>Summary</h2>",
                f"<p>Total violations: {summary.get('violation_count', 0)}</p>",
                f"<p>Primary severity: {summary.get('primary_severity', 'Unknown')}</p>",
                "<h3>Violations by Severity</h3>",
                "<ul>"
            ]
            
            for severity, count in summary.get("severity_counts", {}).items():
                html_parts.append(f"<li>{severity}: {count}</li>")
                
            html_parts.append("</ul>")
            html_parts.append("<h3>Violations by Category</h3>")
            html_parts.append("<ul>")
            
            for category, count in summary.get("categories", {}).items():
                html_parts.append(f"<li>{category}: {count}</li>")
                
            html_parts.append("</ul>")
            
            if summary.get("suggestion"):
                html_parts.append("<h3>Suggested Remediation</h3>")
                html_parts.append(f"<p>{summary.get('suggestion')}</p>")
                
            html_parts.append("</body></html>")
            
            return "".join(html_parts)
            
        else:  # Default to text format
            # Generate plain text report
            text_parts = [
                "COMPLIANCE VIOLATION REPORT",
                "===========================",
                f"Total violations: {summary.get('violation_count', 0)}",
                f"Primary severity: {summary.get('primary_severity', 'Unknown')}",
                "",
                "VIOLATIONS BY SEVERITY",
                "----------------------"
            ]
            
            for severity, count in summary.get("severity_counts", {}).items():
                text_parts.append(f"{severity}: {count}")
                
            text_parts.append("")
            text_parts.append("VIOLATIONS BY CATEGORY")
            text_parts.append("---------------------")
            
            for category, count in summary.get("categories", {}).items():
                text_parts.append(f"{category}: {count}")
                
            text_parts.append("")
            
            if "top_rules" in summary:
                text_parts.append("TOP TRIGGERED RULES")
                text_parts.append("-----------------")
                
                for rule in summary.get("top_rules", []):
                    text_parts.append(f"- {rule.get('rule_id', 'Unknown')}: {rule.get('count', 0)} violations")
                    
                text_parts.append("")
                
            if summary.get("suggestion"):
                text_parts.append("SUGGESTED REMEDIATION")
                text_parts.append("---------------------")
                text_parts.append(summary.get("suggestion"))
                
            return "\n".join(text_parts)
            
    def get_remediation_suggestions(self, violations, context=None):
        """
        Get specific remediation suggestions for the given violations.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            Dict with remediation suggestions
        """
        if not violations:
            return {"suggestions": []}
            
        violations_by_category = self._group_by_category(violations)
        suggestions = []
        
        # Generate category-specific suggestions
        for category, category_violations in violations_by_category.items():
            if category in self.remediation_templates:
                template = self.remediation_templates[category]
                
                # Extract all unique rule IDs in this category
                rule_ids = set(v.get("rule_id", "") for v in category_violations)
                rule_names = [self.rule_metadata.get(rule_id, {}).get("name", "Unknown rule") for rule_id in rule_ids if rule_id]
                
                # Format the suggestion
                suggestion = template.format(
                    count=len(category_violations),
                    rules=", ".join(rule_names),
                    category=category
                )
                
                suggestions.append({
                    "category": category,
                    "suggestion": suggestion,
                    "priority": self._get_category_priority(category),
                    "violation_count": len(category_violations)
                })
                
        # Generate rule-specific suggestions
        rule_violations = defaultdict(list)
        for violation in violations:
            rule_id = violation.get("rule_id", "")
            if rule_id:
                rule_violations[rule_id].append(violation)
                
        for rule_id, rule_violations_list in rule_violations.items():
            if rule_id in self.rule_metadata and "remediation_template" in self.rule_metadata[rule_id]:
                template = self.rule_metadata[rule_id]["remediation_template"]
                rule_name = self.rule_metadata[rule_id].get("name", "Unknown rule")
                
                # Extract example content from violations
                example_texts = []
                for v in rule_violations_list[:3]:  # Take up to 3 examples
                    if "metadata" in v and "matched_content" in v["metadata"]:
                        example_texts.append(v["metadata"]["matched_content"])
                        
                # Format the suggestion
                suggestion = template.format(
                    count=len(rule_violations_list),
                    rule=rule_name,
                    examples=", ".join(f'"{text}"' for text in example_texts) if example_texts else "N/A"
                )
                
                suggestions.append({
                    "rule_id": rule_id,
                    "suggestion": suggestion,
                    "priority": self._get_rule_priority(rule_id),
                    "violation_count": len(rule_violations_list)
                })
                
        # Sort suggestions by priority, then by violation count
        suggestions.sort(key=lambda x: (-x["priority"], -x["violation_count"]))
        
        return {"suggestions": suggestions}
        
    def assess_violation_severity(self, violations):
        """
        Assess the overall severity of the given violations.
        
        Args:
            violations: List of compliance violations
            
        Returns:
            Dict with severity assessment
        """
        if not violations:
            return {
                "overall_severity": "none",
                "severity_score": 0,
                "violation_count": 0
            }
            
        # Count violations by severity
        severity_counts = Counter(v.get("severity", "medium") for v in violations)
        
        # Calculate weighted severity score
        severity_score = sum(
            count * self.severity_weights.get(severity, 1)
            for severity, count in severity_counts.items()
        )
        
        # Determine overall severity based on thresholds
        overall_severity = "none"
        for severity, threshold in sorted(self.violation_thresholds.items(), key=lambda x: -self.severity_weights.get(x[0], 0)):
            if severity_counts.get(severity, 0) >= threshold:
                overall_severity = severity
                break
                
        return {
            "overall_severity": overall_severity,
            "severity_score": severity_score,
            "severity_counts": dict(severity_counts),
            "violation_count": len(violations)
        }
        
    def _load_rule_metadata(self, rule_metadata):
        """Load and process rule metadata."""
        processed_metadata = {}
        
        for rule_id, metadata in rule_metadata.items():
            processed_metadata[rule_id] = {
                "name": metadata.get("name", f"Rule {rule_id}"),
                "category": metadata.get("category", "general"),
                "description": metadata.get("description", ""),
                "severity": metadata.get("severity", "medium"),
                "remediation_template": metadata.get("remediation_template", ""),
                "priority": metadata.get("priority", 0),
                "tags": metadata.get("tags", [])
            }
            
        return processed_metadata
        
    def _initialize_pattern_matchers(self):
        """Initialize pattern matching functions."""
        return {
            "repeated_violations": self._match_repeated_violations,
            "sequential_violations": self._match_sequential_violations,
            "contextual_patterns": self._match_contextual_patterns
        }
        
    def _group_by_severity(self, violations):
        """Group violations by severity level."""
        result = defaultdict(list)
        
        for violation in violations:
            severity = violation.get("severity", "medium")
            result[severity].append(violation)
            
        return result
        
    def _group_by_category(self, violations):
        """Group violations by category."""
        result = defaultdict(list)
        
        for violation in violations:
            rule_id = violation.get("rule_id", "")
            
            # Look up category from rule metadata
            if rule_id in self.rule_metadata:
                category = self.rule_metadata[rule_id].get("category", "general")
            else:
                # Try to determine category from rule_id
                category = self._infer_category_from_rule_id(rule_id)
                
            result[category].append(violation)
            
        return result
        
    def _infer_category_from_rule_id(self, rule_id):
        """Infer violation category from rule ID if not in metadata."""
        # Check explicit mapping first
        if rule_id in self.category_mapping:
            return self.category_mapping[rule_id]
            
        # Try to infer from rule ID prefix
        for prefix, category in self.category_mapping.items():
            if rule_id.startswith(prefix):
                return category
                
        # Default category
        return "general"
        
    def _identify_top_rules(self, violations):
        """Identify the most frequently triggered rules."""
        rule_counts = Counter(v.get("rule_id", "") for v in violations if "rule_id" in v)
        
        top_rules = []
        for rule_id, count in rule_counts.most_common():
            if not rule_id:
                continue
                
            rule_info = {
                "rule_id": rule_id,
                "count": count
            }
            
            # Add metadata if available
            if rule_id in self.rule_metadata:
                rule_info.update({
                    "name": self.rule_metadata[rule_id].get("name", ""),
                    "category": self.rule_metadata[rule_id].get("category", ""),
                    "severity": self.rule_metadata[rule_id].get("severity", "")
                })
                
            top_rules.append(rule_info)
            
        return top_rules
        
    def _determine_primary_severity(self, violations_by_severity):
        """Determine the primary severity level of violations."""
        # Order of severity from highest to lowest
        severity_order = ["critical", "high", "medium", "low"]
        
        for severity in severity_order:
            if severity in violations_by_severity and violations_by_severity[severity]:
                return severity
                
        return "low"  # Default
        
    def _generate_remediation_suggestion(self, violations, context, input_text):
        """Generate an overall remediation suggestion based on violations."""
        if not violations:
            return None
            
        # Get detailed suggestions
        suggestions_result = self.get_remediation_suggestions(violations, context)
        suggestions = suggestions_result.get("suggestions", [])
        
        if not suggestions:
            # Fallback general suggestion
            return "Review and modify content to address compliance issues."
            
        # Take the highest priority suggestion
        top_suggestion = suggestions[0]["suggestion"]
        
        # If there are multiple high-priority suggestions, combine them
        if len(suggestions) > 1 and suggestions[1]["priority"] == suggestions[0]["priority"]:
            top_suggestions = [s["suggestion"] for s in suggestions[:3] if s["priority"] == suggestions[0]["priority"]]
            return " ".join(top_suggestions)
            
        return top_suggestion
        
    def _update_historical_data(self, violations, context):
        """Update historical violation data for trend analysis."""
        # In a real system, this might store data in a database
        timestamp = time.time()
        
        # Group violations by rule ID
        for violation in violations:
            rule_id = violation.get("rule_id", "")
            if rule_id:
                self.historical_violations[rule_id].append({
                    "timestamp": timestamp,
                    "severity": violation.get("severity", "medium"),
                    "context_info": self._extract_context_summary(context)
                })
                
        # Keep only recent history (last 100 entries per rule)
        for rule_id in self.historical_violations:
            if len(self.historical_violations[rule_id]) > 100:
                self.historical_violations[rule_id] = self.historical_violations[rule_id][-100:]
                
        # Update violation trends
        self._update_violation_trends()
        
    def _extract_context_summary(self, context):
        """Extract a summary of context information for historical tracking."""
        if not context:
            return {}
            
        # Extract relevant context info, avoiding storing sensitive data
        summary = {}
        
        if "domain" in context:
            summary["domain"] = context["domain"]
            
        if "user_info" in context:
            # Only store non-sensitive user info
            user_summary = {}
            user_info = context["user_info"]
            
            safe_user_fields = ["role", "access_level", "account_type"]
            for field in safe_user_fields:
                if field in user_info:
                    user_summary[field] = user_info[field]
                    
            summary["user_info"] = user_summary
            
        if "metadata" in context:
            # Only store select metadata
            metadata = context["metadata"]
            metadata_summary = {}
            
            safe_metadata_fields = ["source", "channel", "request_type"]
            for field in safe_metadata_fields:
                if field in metadata:
                    metadata_summary[field] = metadata[field]
                    
            summary["metadata"] = metadata_summary
            
        return summary
        
    def _update_violation_trends(self):
        """Update violation trend analysis."""
        # Calculate current violation distribution
        rule_counts = {
            rule_id: len(violations) 
            for rule_id, violations in self.historical_violations.items()
            if violations  # Only rules with violations
        }
        
        timestamp = time.time()
        
        # Add current snapshot to trends
        self.violation_trends["rule_distribution"].append({
            "timestamp": timestamp,
            "distribution": rule_counts
        })
        
        # Keep only recent trends (last 100 entries)
        if len(self.violation_trends["rule_distribution"]) > 100:
            self.violation_trends["rule_distribution"] = self.violation_trends["rule_distribution"][-100:]
                
    def _detect_violation_patterns(self, violations, context):
        """Detect patterns in violation data."""
        patterns = {}
        
        for pattern_type, matcher in self.pattern_matchers.items():
            pattern_results = matcher(violations, context)
            if pattern_results:
                patterns[pattern_type] = pattern_results
        
        return patterns

    def _initialize_pattern_matchers(self):
        """Initialize pattern matching functions."""
        return {
            "repeated_violations": self._match_repeated_violations,
            "sequential_violations": self._match_sequential_violations,
            "contextual_patterns": self._match_contextual_patterns,
            "content_type_patterns": self._analyze_content_type_patterns,
            "platform_patterns": self._analyze_platform_patterns,
            "time_patterns": self._analyze_time_patterns,
            "location_patterns": self._analyze_location_patterns,
            "multi_context_patterns": self._analyze_multi_context_patterns
        }
            
    def _match_repeated_violations(self, violations, context):
        """Match pattern: repeated violations of the same rule."""
        rule_counts = Counter(v.get("rule_id", "") for v in violations if "rule_id" in v)
        
        repeated_violations = [
            {
                "rule_id": rule_id,
                "count": count,
                "name": self.rule_metadata.get(rule_id, {}).get("name", "Unknown rule"),
                "pattern_type": "repeated_violation"
            }
            for rule_id, count in rule_counts.items()
            if count >= self.min_violations_for_pattern and rule_id
        ]
        
        return repeated_violations if repeated_violations else None
        
    def _match_sequential_violations(self, violations, context):
        """
        Match pattern: violations that follow a sequential pattern over time or in content.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of sequential violation patterns or None if none found
        """
        if not violations or len(violations) < 3:  # Need at least 3 violations for a meaningful sequence
            return None
            
        # Look for sequences in rule IDs, categories, or severity levels
        sequential_patterns = []
        
        # 1. Sort violations by position/timestamp if available
        sorted_violations = sorted(
            violations,
            key=lambda v: v.get("metadata", {}).get("position", 0) 
            if "metadata" in v and "position" in v.get("metadata", {})
            else 0
        )
        
        # 2. Check for rule ID sequences
        rule_sequence = self._extract_sequence([v.get("rule_id", "") for v in sorted_violations])
        if rule_sequence and len(rule_sequence) >= 3:
            sequential_patterns.append({
                "type": "rule_sequence",
                "sequence": rule_sequence,
                "confidence": 0.8,
                "description": f"Sequence of rule violations: {', '.join(rule_sequence)}"
            })
        
        # 3. Check for severity escalation
        severity_values = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        severity_sequence = [severity_values.get(v.get("severity", "medium"), 2) for v in sorted_violations]
        
        # Check if severities are increasing
        is_escalating = all(severity_sequence[i] <= severity_sequence[i+1] for i in range(len(severity_sequence)-1))
        has_significant_change = max(severity_sequence) - min(severity_sequence) >= 2
        
        if is_escalating and has_significant_change and len(severity_sequence) >= 3:
            sequential_patterns.append({
                "type": "severity_escalation",
                "starting_severity": sorted_violations[0].get("severity", "medium"),
                "ending_severity": sorted_violations[-1].get("severity", "medium"),
                "violation_count": len(sorted_violations),
                "confidence": 0.7,
                "description": f"Escalating severity from {sorted_violations[0].get('severity', 'medium')} to {sorted_violations[-1].get('severity', 'medium')}"
            })
        
        # 4. Check for category sequences
        if hasattr(self, 'rule_metadata'):
            categories = []
            for violation in sorted_violations:
                rule_id = violation.get("rule_id", "")
                if rule_id in self.rule_metadata:
                    category = self.rule_metadata[rule_id].get("category", "general")
                    categories.append(category)
                else:
                    categories.append("unknown")
                    
            category_sequence = self._extract_sequence(categories)
            if category_sequence and len(category_sequence) >= 3:
                sequential_patterns.append({
                    "type": "category_sequence",
                    "sequence": category_sequence,
                    "confidence": 0.6,
                    "description": f"Sequence of violation categories: {', '.join(category_sequence)}"
                })
        
        return sequential_patterns if sequential_patterns else None
    
    def _extract_sequence(self, items):
        """Extract the longest non-repeating sequence from a list of items"""
        if not items or len(items) < 3:
            return None
            
        # Find sequences of 3 or more non-repeating items
        sequences = []
        current_sequence = [items[0]]
        
        for i in range(1, len(items)):
            if items[i] != items[i-1]:  # Different from previous
                current_sequence.append(items[i])
            else:  # Repeating item, end current sequence
                if len(current_sequence) >= 3:
                    sequences.append(current_sequence)
                current_sequence = [items[i]]
        
        # Add final sequence if valid
        if len(current_sequence) >= 3:
            sequences.append(current_sequence)
        
        # Return the longest sequence found
        if sequences:
            return max(sequences, key=len)
        return None
        
    def _match_contextual_patterns(self, violations, context):
        """Match pattern: violations that occur in specific contexts."""
        # This is a placeholder for more sophisticated contextual pattern detection
        # In a real system, this would analyze patterns related to specific contexts
        return None
        
    def _assess_impact(self, violations, context):
        """Assess the potential impact of the violations."""
        severity_assessment = self.assess_violation_severity(violations)
        
        # Base impact on severity
        impact_level = severity_assessment["overall_severity"]
        
        # Consider context to adjust impact assessment
        if context:
            domain = context.get("domain", "general")
            
            # In some domains, certain violations have higher impact
            domain_sensitivities = self.config.get("domain_sensitivities", {}).get(domain, {})
            
            for violation in violations:
                rule_id = violation.get("rule_id", "")
                if rule_id in domain_sensitivities:
                    rule_impact = domain_sensitivities[rule_id]
                    # Escalate impact if domain-sensitive rule is violated
                    if rule_impact == "high" and impact_level in ["low", "medium"]:
                        impact_level = "high"
                    elif rule_impact == "critical" and impact_level != "critical":
                        impact_level = "critical"
        
        return {
            "impact_level": impact_level,
            "severity_assessment": severity_assessment
        }
        
    def _get_category_priority(self, category):
        """Get priority level for a violation category."""
        category_priorities = self.config.get("category_priorities", {})
        return category_priorities.get(category, 0)
        
    def _get_rule_priority(self, rule_id):
        """Get priority level for a rule."""
        if rule_id in self.rule_metadata:
            return self.rule_metadata[rule_id].get("priority", 0)
        return 0

    def _match_sequential_violations(self, violations, context):
        """
        Match violations that follow a sequential pattern.
        
        Identifies patterns where violations occur in a specific order or sequence,
        which may indicate systematic issues or common violation paths.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of sequential violation patterns or None if none found
        """
        if len(violations) < 2:
            return None
            
        # Group violations by content or session if available
        grouped_violations = {}
        
        # First try to group by content
        if context and 'content_id' in context:
            content_id = context['content_id']
            grouped_violations[content_id] = violations
        # Then try by session
        elif context and 'session_id' in context:
            session_id = context['session_id']
            grouped_violations[session_id] = violations
        # Finally, just use a default group if no better grouping is available
        else:
            grouped_violations['default'] = violations
        
        sequential_patterns = []
        
        for group_id, group_violations in grouped_violations.items():
            # Sort violations by timestamp if available
            sorted_violations = sorted(
                group_violations,
                key=lambda v: v.get('timestamp', 0)
            )
            
            # Find sequence patterns using sliding window approach
            # Look for sequences of 2-4 violations
            for window_size in range(2, min(5, len(sorted_violations) + 1)):
                # Use frequency counting to identify repeated sequences
                sequence_counts = {}
                
                for i in range(len(sorted_violations) - window_size + 1):
                    # Create a sequence key using rule IDs
                    sequence = tuple(v.get('rule_id', str(i)) for v, i in 
                                zip(sorted_violations[i:i+window_size], range(window_size)))
                    
                    if sequence not in sequence_counts:
                        sequence_counts[sequence] = 0
                    sequence_counts[sequence] += 1
                
                # Filter sequences that occur multiple times
                for sequence, count in sequence_counts.items():
                    if count >= self.min_violations_for_pattern:
                        # Found a repeated sequence
                        rules = [rule_id for rule_id in sequence]
                        
                        # Get rule metadata for better description
                        rule_names = []
                        for rule_id in rules:
                            if rule_id in self.rule_metadata:
                                rule_names.append(self.rule_metadata[rule_id].get('name', rule_id))
                            else:
                                rule_names.append(f"Rule {rule_id}")
                        
                        sequential_patterns.append({
                            'rules': rules,
                            'rule_names': rule_names,
                            'count': count,
                            'group_id': group_id,
                            'pattern_type': 'sequential_violation',
                            'description': f"Sequential pattern: {' → '.join(rule_names)} (occurs {count} times)"
                        })
            
            # Look for time-based patterns - violations that consistently occur within timeframes
            if len(sorted_violations) >= 2 and all('timestamp' in v for v in sorted_violations):
                time_clusters = self._cluster_by_time_proximity(sorted_violations)
                
                for cluster in time_clusters:
                    if len(cluster) >= self.min_violations_for_pattern:
                        # Extract rule IDs and names
                        rules = [v.get('rule_id', 'unknown') for v in cluster]
                        rule_names = []
                        for rule_id in rules:
                            if rule_id in self.rule_metadata:
                                rule_names.append(self.rule_metadata[rule_id].get('name', rule_id))
                            else:
                                rule_names.append(f"Rule {rule_id}")
                        
                        sequential_patterns.append({
                            'rules': rules,
                            'rule_names': rule_names,
                            'count': len(cluster),
                            'group_id': group_id,
                            'pattern_type': 'time_proximity_pattern',
                            'description': f"Time-proximity pattern: {', '.join(rule_names)} occur together"
                        })
        
        return sequential_patterns if sequential_patterns else None

    def _cluster_by_time_proximity(self, violations, max_time_diff=60):
        """
        Helper function to cluster violations by time proximity.
        
        Args:
            violations: List of violations with timestamps
            max_time_diff: Maximum time difference (in seconds) to consider violations as clustered
            
        Returns:
            List of violation clusters
        """
        # Ensure violations are sorted by timestamp
        sorted_violations = sorted(violations, key=lambda v: v.get('timestamp', 0))
        
        clusters = []
        current_cluster = [sorted_violations[0]]
        
        for i in range(1, len(sorted_violations)):
            curr_time = sorted_violations[i].get('timestamp', 0)
            prev_time = sorted_violations[i-1].get('timestamp', 0)
            
            # Check if time difference is within threshold
            if curr_time - prev_time <= max_time_diff:
                # Add to current cluster
                current_cluster.append(sorted_violations[i])
            else:
                # Start a new cluster
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [sorted_violations[i]]
        
        # Add the last cluster if it has at least 2 violations
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
            
        return clusters

    def _match_contextual_patterns(self, violations, context):
        """
        Match violations that occur in specific contexts.
        
        Identifies patterns related to specific contexts such as domain, user role,
        content type, etc., which can help identify context-specific compliance issues.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of contextual violation patterns or None if none found
        """
        if not context or not violations:
            return None
        
        contextual_patterns = []
        
        # Identify context-specific patterns
        
        # 1. Domain-specific patterns
        if 'domain' in context:
            domain = context['domain']
            domain_violations = self._analyze_domain_patterns(violations, domain)
            if domain_violations:
                contextual_patterns.extend(domain_violations)
        
        # 2. User role patterns
        if 'user_info' in context and 'role' in context['user_info']:
            user_role = context['user_info']['role']
            role_violations = self._analyze_role_patterns(violations, user_role)
            if role_violations:
                contextual_patterns.extend(role_violations)
        
        # 3. Content type patterns
        if 'content_type' in context:
            content_type = context['content_type']
            content_type_violations = self._analyze_content_type_patterns(violations, content_type)
            if content_type_violations:
                contextual_patterns.extend(content_type_violations)
        
        # 4. Device or platform patterns
        if 'platform' in context or 'device' in context:
            platform = context.get('platform', context.get('device'))
            platform_violations = self._analyze_platform_patterns(violations, platform)
            if platform_violations:
                contextual_patterns.extend(platform_violations)
        
        # 5. Time-based patterns (time of day, day of week)
        if 'timestamp' in context:
            import datetime
            try:
                dt = datetime.datetime.fromisoformat(context['timestamp'])
                time_patterns = self._analyze_time_patterns(violations, dt)
                if time_patterns:
                    contextual_patterns.extend(time_patterns)
            except (ValueError, TypeError):
                pass
        
        # 6. Location-based patterns
        if 'location' in context or 'country' in context or 'region' in context:
            location = context.get('location', context.get('country', context.get('region')))
            location_patterns = self._analyze_location_patterns(violations, location)
            if location_patterns:
                contextual_patterns.extend(location_patterns)
        
        # 7. Multi-dimensional context patterns (combinations of contexts)
        multi_context_patterns = self._analyze_multi_context_patterns(violations, context)
        if multi_context_patterns:
            contextual_patterns.extend(multi_context_patterns)
        
        return contextual_patterns if contextual_patterns else None

    def _analyze_domain_patterns(self, violations, domain):
        """Analyze domain-specific violation patterns"""
        # Group violations by rule ID
        rule_violations = {}
        for violation in violations:
            rule_id = violation.get('rule_id', 'unknown')
            if rule_id not in rule_violations:
                rule_violations[rule_id] = []
            rule_violations[rule_id].append(violation)
        
        # Check if any rules have a high frequency in this domain
        domain_patterns = []
        
        for rule_id, rule_viols in rule_violations.items():
            if len(rule_viols) >= self.min_violations_for_pattern:
                # Check if this rule has domain-specific sensitivity
                domain_sensitivity = self.config.get('domain_sensitivities', {}).get(domain, {}).get(rule_id)
                
                if domain_sensitivity:
                    # This rule has specific domain sensitivity
                    domain_patterns.append({
                        'rule_id': rule_id,
                        'count': len(rule_viols),
                        'domain': domain,
                        'pattern_type': 'domain_specific_pattern',
                        'name': self.rule_metadata.get(rule_id, {}).get('name', f"Rule {rule_id}"),
                        'description': f"Domain-specific pattern: {rule_id} frequently violated in {domain} domain"
                    })
        
        return domain_patterns

    def _analyze_role_patterns(self, violations, user_role):
        """Analyze user role specific violation patterns"""
        # Similar implementation to domain patterns but for user roles
        # Group violations by rule ID
        rule_violations = {}
        for violation in violations:
            rule_id = violation.get('rule_id', 'unknown')
            if rule_id not in rule_violations:
                rule_violations[rule_id] = []
            rule_violations[rule_id].append(violation)
        
        # Check if any rules have a high frequency for this user role
        role_patterns = []
        
        for rule_id, rule_viols in rule_violations.items():
            if len(rule_viols) >= self.min_violations_for_pattern:
                # Check if this rule has role-specific sensitivity
                role_sensitivity = self.config.get('role_sensitivities', {}).get(user_role, {}).get(rule_id)
                
                if role_sensitivity:
                    # This rule has specific role sensitivity
                    role_patterns.append({
                        'rule_id': rule_id,
                        'count': len(rule_viols),
                        'user_role': user_role,
                        'pattern_type': 'role_specific_pattern',
                        'name': self.rule_metadata.get(rule_id, {}).get('name', f"Rule {rule_id}"),
                        'description': f"Role-specific pattern: {rule_id} frequently violated by {user_role} role"
                    })
        
        return role_patterns

    def _analyze_content_type_patterns(self, violations, context):
        """
        Analyze patterns specific to different content types (e.g., text, images, code).
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of detected content type patterns
        """
        if not violations or not context:
            return None
            
        patterns = []
        content_type = context.get("content_type", "unknown")
        
        # Group violations by content type
        content_type_violations = defaultdict(list)
        
        for violation in violations:
            metadata = violation.get("metadata", {})
            violation_content_type = metadata.get("content_type", content_type)
            content_type_violations[violation_content_type].append(violation)
        
        # Analyze patterns for each content type
        for ct, ct_violations in content_type_violations.items():
            if len(ct_violations) >= self.min_violations_for_pattern:
                # Analyze common rules within this content type
                rule_counts = Counter(v.get("rule_id", "") for v in ct_violations)
                common_rules = [rule for rule, count in rule_counts.items() 
                            if count >= 3 and rule]  # At least 3 occurrences
                
                if common_rules:
                    patterns.append({
                        "type": "content_type_pattern",
                        "content_type": ct,
                        "common_rules": common_rules,
                        "violation_count": len(ct_violations),
                        "confidence": min(1.0, len(ct_violations) / 10)  # Scale confidence
                    })
        
        return patterns if patterns else None

    def _analyze_platform_patterns(self, violations, context):
        """
        Analyze patterns specific to different platforms or environments.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of detected platform patterns
        """
        if not violations or not context:
            return None
            
        patterns = []
        
        # Extract platform information from context
        platform = context.get("platform", {})
        platform_type = platform.get("type") if isinstance(platform, dict) else str(platform)
        
        if not platform_type:
            return None
        
        # Group violations by severity for this platform
        severity_counts = Counter(v.get("severity", "medium") for v in violations)
        
        # Check if this platform has a disproportionate number of high-severity violations
        high_severity_ratio = (severity_counts.get("high", 0) + severity_counts.get("critical", 0)) / \
                            max(1, len(violations))
        
        if high_severity_ratio > 0.4:  # 40% or more are high severity
            patterns.append({
                "type": "platform_risk_pattern",
                "platform": platform_type,
                "high_severity_ratio": high_severity_ratio,
                "violation_count": len(violations),
                "confidence": min(1.0, high_severity_ratio + 0.3)
            })
        
        # Check for platform-specific rule violations
        platform_specific_rules = self._get_platform_specific_rules(platform_type)
        if platform_specific_rules:
            platform_violations = [v for v in violations if v.get("rule_id", "") in platform_specific_rules]
            
            if len(platform_violations) >= 2:
                patterns.append({
                    "type": "platform_specific_pattern",
                    "platform": platform_type,
                    "rules": [v.get("rule_id") for v in platform_violations],
                    "violation_count": len(platform_violations),
                    "confidence": min(1.0, len(platform_violations) / len(platform_specific_rules))
                })
        
        return patterns if patterns else None
        
    def _get_platform_specific_rules(self, platform_type):
        """Get rules specific to a platform type"""
        # This would be populated from configuration or derived from rule metadata
        platform_rules = {
            "web": ["web_accessibility", "csrf_protection", "xss_prevention"],
            "mobile": ["mobile_privacy", "data_collection", "notification_consent"],
            "desktop": ["local_storage", "system_integration", "update_mechanism"],
            "iot": ["device_security", "data_transmission", "physical_access"]
        }
        
        return platform_rules.get(platform_type.lower(), [])
    
    def _analyze_time_patterns(self, violations, context):
        """
        Analyze temporal patterns in violations, identifying time-based trends.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of detected time-based patterns
        """
        if not violations:
            return None
            
        patterns = []
        
        # Check if we have historical data to work with
        if not hasattr(self, 'historical_violations') or not self.historical_violations:
            return None
        
        # Get current timestamp
        current_time = time.time()
        
        # Analyze frequency of violations over time
        time_grouped_violations = defaultdict(list)
        
        # Group historical violations by time period (daily, weekly)
        for rule_id, rule_violations in self.historical_violations.items():
            for violation in rule_violations:
                # Group by day (86400 seconds in a day)
                day_bucket = int(violation.get("timestamp", 0) / 86400)
                time_grouped_violations[day_bucket].append(violation)
        
        # Calculate violation frequency by day
        day_frequencies = {day: len(violations) for day, violations in time_grouped_violations.items()}
        
        # Check for recent spikes in violations
        recent_days = sorted(day_frequencies.keys())[-7:]  # Last 7 days
        if recent_days:
            recent_avg = sum(day_frequencies.get(day, 0) for day in recent_days) / len(recent_days)
            historical_days = [d for d in day_frequencies.keys() if d not in recent_days]
            
            if historical_days:
                historical_avg = sum(day_frequencies.get(day, 0) for day in historical_days) / len(historical_days)
                
                # Detect significant increase
                if recent_avg > historical_avg * 1.5:
                    patterns.append({
                        "type": "time_frequency_pattern",
                        "pattern": "increasing_violations",
                        "recent_daily_avg": recent_avg,
                        "historical_daily_avg": historical_avg,
                        "increase_factor": recent_avg / historical_avg if historical_avg > 0 else float('inf'),
                        "confidence": min(1.0, (recent_avg - historical_avg) / max(1, historical_avg))
                    })
        
        # Detect time-of-day patterns if timestamp data is detailed enough
        hour_violations = defaultdict(int)
        for bucket, violations in time_grouped_violations.items():
            for violation in violations:
                timestamp = violation.get("timestamp", 0)
                hour = datetime.datetime.fromtimestamp(timestamp).hour
                hour_violations[hour] += 1
        
        # Check for concentrated violations during specific hours
        total_violations = sum(hour_violations.values())
        if total_violations > 0:
            for hour, count in hour_violations.items():
                ratio = count / total_violations
                if ratio > 0.25:  # More than 25% of violations happen in this hour
                    patterns.append({
                        "type": "time_of_day_pattern",
                        "hour": hour,
                        "violation_ratio": ratio,
                        "violation_count": count,
                        "confidence": min(1.0, ratio + 0.3)
                    })
        
        return patterns if patterns else None

    def _analyze_location_patterns(self, violations, context):
        """
        Analyze patterns related to geographical or logical locations.
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of detected location-based patterns
        """
        if not violations or not context:
            return None
            
        patterns = []
        
        # Extract location information
        location_info = context.get("location", {})
        if not location_info:
            return None
        
        # Normalize location information
        country = location_info.get("country") if isinstance(location_info, dict) else None
        region = location_info.get("region") if isinstance(location_info, dict) else None
        
        if not country and not region:
            return None
        
        # Get regulatory requirements specific to this location
        location_specific_rules = self._get_location_specific_rules(country, region)
        
        # Check for location-specific violations
        location_violations = []
        for violation in violations:
            rule_id = violation.get("rule_id", "")
            if rule_id in location_specific_rules:
                location_violations.append(violation)
        
        if location_violations:
            # Calculate compliance risk for this location
            location_risk = len(location_violations) / len(location_specific_rules)
            
            patterns.append({
                "type": "location_compliance_pattern",
                "country": country,
                "region": region,
                "violation_count": len(location_violations),
                "rule_count": len(location_specific_rules),
                "compliance_risk": location_risk,
                "rules_violated": [v.get("rule_id") for v in location_violations],
                "confidence": min(1.0, location_risk + 0.2)
            })
        
        # Check for regional violation clusters
        if hasattr(self, 'historical_violations') and region:
            region_violations = self._get_historical_violations_by_region(region)
            if region_violations:
                region_rule_counts = Counter(v.get("rule_id", "") for v in region_violations)
                common_rules = [rule for rule, count in region_rule_counts.items() 
                            if count >= 3 and rule]  # At least 3 occurrences
                
                if common_rules:
                    patterns.append({
                        "type": "regional_violation_pattern",
                        "region": region,
                        "common_rules": common_rules,
                        "violation_count": len(region_violations),
                        "confidence": min(1.0, len(common_rules) / 10 + 0.5)
                    })
        
        return patterns if patterns else None
            
    def _get_location_specific_rules(self, country, region):
        """Get rules specific to a location"""
        # This would be populated from configuration or derived from rule metadata
        # In a real implementation, this would look up actual regulatory requirements by region
        location_rules = {
            "us": ["us_privacy_law", "ccpa", "coppa"],
            "eu": ["gdpr", "eprivacy_directive", "dsa"],
            "uk": ["uk_gdpr", "dpa_2018", "pecr"],
            "canada": ["pipeda", "casl", "provincial_privacy_laws"],
            "australia": ["privacy_act", "spam_act", "consumer_law"],
            "california": ["ccpa", "cpra", "shine_the_light"],
            "new_york": ["shield_act", "ny_privacy_law"]
        }
        
        # Combine country and region rules
        rules = set()
        
        if country and country.lower() in location_rules:
            rules.update(location_rules[country.lower()])
            
        if region and region.lower() in location_rules:
            rules.update(location_rules[region.lower()])
            
        return list(rules)
        
    def _get_historical_violations_by_region(self, region):
        """Get historical violations for a specific region"""
        # In a real implementation, this would query stored violation data
        region_violations = []
        
        # Simplified implementation - extract violations with matching region
        for rule_id, violations in self.historical_violations.items():
            for violation in violations:
                violation_context = violation.get("context_info", {})
                violation_location = violation_context.get("location", {})
                violation_region = violation_location.get("region", "") if isinstance(violation_location, dict) else ""
                
                if violation_region.lower() == region.lower():
                    region_violations.append(violation)
        
        return region_violations

    def _analyze_multi_context_patterns(self, violations, context):
        """
        Analyze patterns that span multiple context dimensions (e.g., content type + platform + time).
        
        Args:
            violations: List of compliance violations
            context: Optional context information
            
        Returns:
            List of detected multi-dimensional patterns
        """
        if not violations or not context:
            return None
            
        patterns = []
        
        # Extract key context dimensions
        content_type = context.get("content_type", "unknown")
        platform = context.get("platform", {})
        platform_type = platform.get("type", "unknown") if isinstance(platform, dict) else "unknown"
        user_info = context.get("user_info", {})
        user_role = user_info.get("role", "unknown") if isinstance(user_info, dict) else "unknown"
        
        # Group violations by rule ID
        rule_violations = defaultdict(list)
        for violation in violations:
            rule_id = violation.get("rule_id", "")
            if rule_id:
                rule_violations[rule_id].append(violation)
        
        # Identify rules that appear across different context dimensions
        cross_context_rules = []
        
        for rule_id, rule_viols in rule_violations.items():
            # Check if we have historical data for this rule
            if hasattr(self, 'historical_violations') and rule_id in self.historical_violations:
                historical_viols = self.historical_violations[rule_id]
                
                # Count contexts where this rule appears
                contexts_seen = set()
                
                for violation in historical_viols:
                    violation_context = violation.get("context_info", {})
                    
                    ctx_type = violation_context.get("content_type", "unknown")
                    ctx_platform = violation_context.get("platform", "unknown")
                    
                    # Create context signature
                    context_sig = f"{ctx_type}|{ctx_platform}"
                    contexts_seen.add(context_sig)
                
                # If rule appears across multiple contexts, it's a cross-context pattern
                if len(contexts_seen) >= 3:  # Appears in at least 3 different contexts
                    cross_context_rules.append({
                        "rule_id": rule_id,
                        "context_count": len(contexts_seen),
                        "contexts": list(contexts_seen),
                        "current_violations": len(rule_viols)
                    })
        
        if cross_context_rules:
            # Sort by number of contexts
            cross_context_rules.sort(key=lambda x: x["context_count"], reverse=True)
            
            patterns.append({
                "type": "cross_context_pattern",
                "rules": cross_context_rules,
                "confidence": min(1.0, len(cross_context_rules) / 5 + 0.4)  # Scale confidence
            })
        
        # Identify patterns related to user roles + content types
        if hasattr(self, 'historical_violations') and user_role != "unknown":
            role_content_violations = defaultdict(int)
            
            # Analyze historical violations by role and content type
            for rule_id, historical_viols in self.historical_violations.items():
                for violation in historical_viols:
                    violation_context = violation.get("context_info", {})
                    violation_user = violation_context.get("user_info", {})
                    violation_role = violation_user.get("role", "") if isinstance(violation_user, dict) else ""
                    violation_content = violation_context.get("content_type", "")
                    
                    if violation_role and violation_content:
                        role_content_pair = f"{violation_role}|{violation_content}"
                        role_content_violations[role_content_pair] += 1
            
            # Check if current user role + content type has high violation frequency
            current_pair = f"{user_role}|{content_type}"
            if current_pair in role_content_violations and role_content_violations[current_pair] >= 5:
                patterns.append({
                    "type": "role_content_pattern",
                    "role": user_role,
                    "content_type": content_type,
                    "historical_violations": role_content_violations[current_pair],
                    "current_violations": len(violations),
                    "confidence": min(1.0, role_content_violations[current_pair] / 20 + 0.5)
                })
        
        return patterns if patterns else None

    def _identify_new_exemptions(self, violations, original_input, context):
        """
        Identify patterns that might qualify for compliance exemptions.
        
        Analyzes patterns to identify potential new exemptions to rules, which can
        help reduce false positives and improve rule precision over time.
        
        Args:
            violations: List of compliance violations
            original_input: Original input content
            context: Optional context information
            
        Returns:
            List of potential exemption patterns
        """
        potential_exemptions = []
        
        # Skip if no violations or input
        if not violations or not original_input:
            return potential_exemptions
        
        # 1. Educational context exemptions
        if context and context.get('purpose') == 'educational':
            educational_exemptions = self._identify_educational_exemptions(violations, original_input)
            potential_exemptions.extend(educational_exemptions)
        
        # 2. Scientific/research context exemptions
        if context and context.get('purpose') in ['research', 'scientific']:
            research_exemptions = self._identify_research_exemptions(violations, original_input)
            potential_exemptions.extend(research_exemptions)
        
        # 3. Quoted content exemptions
        quoted_exemptions = self._identify_quoted_content_exemptions(violations, original_input)
        potential_exemptions.extend(quoted_exemptions)
        
        # 4. Legal/compliance discussion exemptions
        legal_exemptions = self._identify_legal_discussion_exemptions(violations, original_input)
        potential_exemptions.extend(legal_exemptions)
        
        # 5. Statistical/aggregate data exemptions
        statistical_exemptions = self._identify_statistical_exemptions(violations, original_input)
        potential_exemptions.extend(statistical_exemptions)
        
        # 6. Fictional/creative content exemptions
        if context and context.get('content_type') in ['fiction', 'creative', 'artistic']:
            fictional_exemptions = self._identify_fictional_exemptions(violations, original_input)
            potential_exemptions.extend(fictional_exemptions)
        
        # 7. Historical content exemptions
        historical_exemptions = self._identify_historical_exemptions(violations, original_input)
        potential_exemptions.extend(historical_exemptions)
        
        # 8. Redacted/anonymized content exemptions
        anonymized_exemptions = self._identify_anonymized_exemptions(violations, original_input)
        potential_exemptions.extend(anonymized_exemptions)
        
        # 9. Rule-specific contextual exemptions
        for violation in violations:
            rule_id = violation.get('rule_id')
            if not rule_id:
                continue
                
            rule_exemptions = self._identify_rule_specific_exemptions(rule_id, violation, original_input, context)
            if rule_exemptions:
                potential_exemptions.extend(rule_exemptions)
        
        return potential_exemptions

    def _identify_educational_exemptions(self, violations, original_input):
        """Identify exemption patterns in educational contexts"""
        exemptions = []
        
        # Check for educational indicators
        educational_indicators = [
            r'for\s+educational\s+purposes',
            r'in\s+an?\s+educational\s+context',
            r'for\s+teaching\s+purposes',
            r'as\s+a\s+learning\s+example',
            r'for\s+instructional\s+purposes'
        ]
        
        import re
        for indicator in educational_indicators:
            if re.search(indicator, original_input, re.IGNORECASE):
                # Found educational context indicator
                for violation in violations:
                    rule_id = violation.get('rule_id')
                    if not rule_id:
                        continue
                    
                    # Check if this rule can have educational exemptions
                    if self._rule_allows_educational_exemption(rule_id):
                        exemptions.append({
                            'rule_id': rule_id,
                            'pattern': indicator,
                            'match': re.search(indicator, original_input, re.IGNORECASE).group(0),
                            'exemption_type': 'educational_context',
                            'description': f"Educational context exemption for rule {rule_id}",
                            'confidence': 0.8
                        })
        
        return exemptions

    def _rule_allows_educational_exemption(self, rule_id):
        """Check if a rule allows educational exemptions"""
        # This would check rule metadata to see if educational exemptions are allowed
        # For now, default to True for most rules except highly sensitive ones
        high_risk_rules = ['pii_disclosure', 'phi_disclosure', 'financial_account_numbers']
        return rule_id not in high_risk_rules

    def _identify_research_exemptions(self, violations, original_input):
        """Identify exemption patterns in research contexts"""
        # Similar implementation to educational exemptions
        return []  # Placeholder - implement similar to educational exemptions

    def _identify_quoted_content_exemptions(self, violations, original_input):
        """Identify exemption patterns for quoted content"""
        exemptions = []
        
        # Check for quotation patterns
        import re
        
        # Find all quoted segments
        quote_patterns = [
            r'"([^"]+)"',           # Double quotes
            r"'([^']+)'",           # Single quotes
            r'\["""([^"""]+)\]"""', # Smart quotes
            r"[«]([^»])+[»]",       # Guillemets
            r"[「]([^」])+[」]"      # CJK quotes
        ]
        
        for pattern in quote_patterns:
            for match in re.finditer(pattern, original_input):
                quoted_text = match.group(1)
                quote_span = (match.start(), match.end())
                
                # Check if any violations occur within this quote
                for violation in violations:
                    if 'location' in violation:
                        viol_start = violation['location'].get('start', 0)
                        viol_end = violation['location'].get('end', 0)
                        
                        # Check if violation is contained within quote
                        if quote_span[0] <= viol_start and viol_end <= quote_span[1]:
                            rule_id = violation.get('rule_id', 'unknown')
                            
                            # Check if this rule allows quote exemptions
                            if self._rule_allows_quote_exemption(rule_id):
                                exemptions.append({
                                    'rule_id': rule_id,
                                    'pattern': pattern,
                                    'match': quoted_text,
                                    'exemption_type': 'quoted_content',
                                    'description': f"Quoted content exemption for rule {rule_id}",
                                    'confidence': 0.9,
                                    'location': {'start': quote_span[0], 'end': quote_span[1]}
                                })
        
        return exemptions

    def _rule_allows_quote_exemption(self, rule_id):
        """Check if a rule allows quote exemptions"""
        # Similar to educational exemptions, check rule metadata
        return True  # Placeholder - implement rule-specific logic

    def _identify_legal_discussion_exemptions(self, violations, original_input):
        """Identify exemption patterns for legal/compliance discussions"""
        # Implementation for legal discussion exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_statistical_exemptions(self, violations, original_input):
        """Identify exemption patterns for statistical/aggregate data"""
        # Implementation for statistical exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_fictional_exemptions(self, violations, original_input):
        """Identify exemption patterns for fictional content"""
        # Implementation for fictional content exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_historical_exemptions(self, violations, original_input):
        """Identify exemption patterns for historical content"""
        # Implementation for historical content exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_anonymized_exemptions(self, violations, original_input):
        """Identify exemption patterns for anonymized content"""
        # Implementation for anonymized content exemptions
        return []  # Placeholder - implement similar to quoted content

    def _identify_rule_specific_exemptions(self, rule_id, violation, original_input, context):
        """Identify rule-specific exemption patterns"""
        # This would contain rule-specific logic for identifying exemptions
        # Different rules may have different kinds of valid exemptions
        
        # Check rule metadata for exemption patterns
        rule_meta = self.rule_metadata.get(rule_id, {})
        exemption_patterns = rule_meta.get('exemption_patterns', [])
        
        exemptions = []
        
        import re
        for pattern in exemption_patterns:
            pattern_regex = pattern.get('regex')
            if pattern_regex and re.search(pattern_regex, original_input, re.IGNORECASE):
                exemptions.append({
                    'rule_id': rule_id,
                    'pattern': pattern_regex,
                    'match': re.search(pattern_regex, original_input, re.IGNORECASE).group(0),
                    'exemption_type': pattern.get('type', 'rule_specific'),
                    'description': pattern.get('description', f"Rule-specific exemption for {rule_id}"),
                    'confidence': pattern.get('confidence', 0.7)
                })
        
        return exemptions

    def _generate_rule_specific_alternatives(self, text, rule_id, rule_issues, context):
        """
        Generate alternative formulations specific to a rule violation.
        
        Uses sophisticated techniques like synonym replacement, restructuring,
        or embeddings-based reformulation to address specific rule violations.
        
        Args:
            text: Original text content
            rule_id: Rule identifier
            rule_issues: Issues related to this rule
            context: Optional context information
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # Get rule metadata
        rule_meta = self.rule_metadata.get(rule_id, {})
        rule_type = rule_meta.get('type', 'unknown')
        rule_name = rule_meta.get('name', f"Rule {rule_id}")
        
        # Different strategies based on rule type
        if rule_type == 'prohibited_term':
            # Replace prohibited terms with alternatives
            alternatives.extend(self._generate_term_replacement_alternatives(text, rule_id, rule_issues, context))
        
        elif rule_type == 'sensitive_data':
            # Anonymize or redact sensitive data
            alternatives.extend(self._generate_anonymization_alternatives(text, rule_id, rule_issues, context))
        
        elif rule_type == 'data_minimization':
            # Reduce unnecessary data
            alternatives.extend(self._generate_data_minimization_alternatives(text, rule_id, rule_issues, context))
        
        elif rule_type == 'disclaimer_required':
            # Add required disclaimers
            alternatives.extend(self._generate_disclaimer_alternatives(text, rule_id, rule_issues, context))
        
        elif rule_type == 'biased_language':
            # Replace biased language
            alternatives.extend(self._generate_bias_correction_alternatives(text, rule_id, rule_issues, context))
        
        # Additional rule-specific alternatives using embeddings if available
        embedding_alternatives = self._generate_embedding_based_alternatives(text, rule_id, rule_issues, context)
        alternatives.extend(embedding_alternatives)
        
        # Try transformer-based alternatives if available
        transformer_alternatives = self._generate_transformer_based_alternatives(text, rule_id, rule_issues, context)
        alternatives.extend(transformer_alternatives)
        
        return alternatives

    def _generate_term_replacement_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives by replacing prohibited terms with safer alternatives"""
        alternatives = []
        
        # Get prohibited terms for this rule
        rule_meta = self.rule_metadata.get(rule_id, {})
        prohibited_terms = rule_meta.get('prohibited_terms', [])
        
        # Get alternative terms for each prohibited term
        term_alternatives = rule_meta.get('term_alternatives', {})
        
        for issue in rule_issues:
            # Get the problematic text
            if 'metadata' in issue and 'matched_content' in issue['metadata']:
                matched_text = issue['metadata']['matched_content']
                
                # Find this text in the original content
                import re
                matches = list(re.finditer(re.escape(matched_text), text, re.IGNORECASE))
                
                for match in matches:
                    start, end = match.span()
                    
                    # Try term-specific alternatives
                    for term in prohibited_terms:
                        if term.lower() in matched_text.lower():
                            # Found a prohibited term, generate alternatives
                            
                            # Method 1: Use predefined alternatives if available
                            if term in term_alternatives:
                                for alt_term in term_alternatives[term]:
                                    # Replace the term
                                    alternative_text = text[:start] + text[start:end].replace(term, alt_term) + text[end:]
                                    
                                    alternatives.append({
                                        'text': alternative_text,
                                        'rule_id': rule_id,
                                        'confidence': 0.9,
                                        'type': 'term_replacement',
                                        'description': f"Replaced prohibited term '{term}' with '{alt_term}'"
                                    })
                            
                            # Method 2: Use thesaurus for synonyms
                            synonyms = self._get_synonyms(term)
                            for synonym in synonyms[:3]:  # Limit to top 3 synonyms
                                # Replace the term
                                alternative_text = text[:start] + text[start:end].replace(term, synonym) + text[end:]
                                
                                alternatives.append({
                                    'text': alternative_text,
                                    'rule_id': rule_id,
                                    'confidence': 0.7,
                                    'type': 'synonym_replacement',
                                    'description': f"Replaced prohibited term '{term}' with synonym '{synonym}'"
                                })
        
        return alternatives

    def _get_synonyms(self, term):
        """Get synonyms for a term using thesaurus or embeddings"""
        # This would use a thesaurus API or embedding model in a real implementation
        # For now, return some examples
        
        # Try using NLTK WordNet if available
        try:
            from nltk.corpus import wordnet
            synonyms = []
            
            # Get synonyms from WordNet
            for syn in wordnet.synsets(term):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != term and synonym not in synonyms:
                        synonyms.append(synonym)
            
            return synonyms[:5]  # Return top 5 synonyms
        except:
            # Fallback to a simple dictionary for common terms
            synonym_dict = {
                'issue': ['problem', 'concern', 'matter'],
                'bad': ['poor', 'suboptimal', 'concerning'],
                'good': ['positive', 'beneficial', 'favorable'],
                'customer': ['client', 'user', 'consumer'],
                'data': ['information', 'details', 'records'],
                'money': ['funds', 'financial resources', 'capital'],
                'problem': ['issue', 'challenge', 'difficulty'],
                'sensitive': ['confidential', 'private', 'protected']
            }
            
            return synonym_dict.get(term.lower(), [])

    def _generate_anonymization_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives by anonymizing sensitive data"""
        alternatives = []
        
        for issue in rule_issues:
            # Get the sensitive data
            if 'metadata' in issue and 'matched_content' in issue['metadata']:
                sensitive_text = issue['metadata']['matched_content']
                
                # Find this text in the original content
                import re
                matches = list(re.finditer(re.escape(sensitive_text), text, re.IGNORECASE))
                
                for match in matches:
                    start, end = match.span()
                    
                    # Method 1: Simple redaction
                    redacted_text = text[:start] + "[REDACTED]" + text[end:]
                    alternatives.append({
                        'text': redacted_text,
                        'rule_id': rule_id,
                        'confidence': 0.9,
                        'type': 'redaction',
                        'description': f"Redacted sensitive data"
                    })
                    
                    # Method 2: Type-specific anonymization
                    # Determine the type of sensitive data
                    data_type = self._determine_sensitive_data_type(sensitive_text)
                    
                    if data_type == 'email':
                        # For emails, keep domain but anonymize local part
                        if '@' in sensitive_text:
                            local, domain = sensitive_text.split('@', 1)
                            anonymized = f"[email]@{domain}"
                            alternative_text = text[:start] + anonymized + text[end:]
                            
                            alternatives.append({
                                'text': alternative_text,
                                'rule_id': rule_id,
                                'confidence': 0.9,
                                'type': 'email_anonymization',
                                'description': f"Anonymized email address"
                            })
                    
                    elif data_type == 'phone':
                        # For phone numbers, keep area code but anonymize the rest
                        if len(sensitive_text) >= 10:
                            # Try to extract area code (first 3 digits for US numbers)
                            digits = ''.join(c for c in sensitive_text if c.isdigit())
                            if len(digits) >= 10:
                                area_code = digits[:3]
                                anonymized = f"({area_code}) XXX-XXXX"
                                alternative_text = text[:start] + anonymized + text[end:]
                                
                                alternatives.append({
                                    'text': alternative_text,
                                    'rule_id': rule_id,
                                    'confidence': 0.8,
                                    'type': 'phone_anonymization',
                                    'description': f"Anonymized phone number"
                                })
                    
                    elif data_type == 'name':
                        # For names, replace with placeholder
                        anonymized = "[Person's Name]"
                        alternative_text = text[:start] + anonymized + text[end:]
                        
                        alternatives.append({
                            'text': alternative_text,
                            'rule_id': rule_id,
                            'confidence': 0.7,
                            'type': 'name_anonymization',
                            'description': f"Anonymized personal name"
                        })
                    
                    # Method 3: Pseudonymization (fake but realistic data)
                    pseudonymized_text = self._generate_pseudonym(sensitive_text, data_type)
                    if pseudonymized_text:
                        pseudo_text = text[:start] + pseudonymized_text + text[end:]
                        
                        alternatives.append({
                            'text': pseudo_text,
                            'rule_id': rule_id,
                            'confidence': 0.7,
                            'type': 'pseudonymization',
                            'description': f"Replaced sensitive data with pseudonym"
                        })
        
        return alternatives

    def _determine_sensitive_data_type(self, text):
        """Determine the type of sensitive data"""
        import re
        
        # Check for email pattern
        if re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            return 'email'
        
        # Check for phone pattern
        if re.match(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
            return 'phone'
        
        # Check for SSN pattern
        if re.match(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text):
            return 'ssn'
        
        # Check for credit card pattern
        if re.match(r'\b(?:\d{4}[-.\s]?){3}\d{4}\b', text):
            return 'credit_card'
        
        # Check for name pattern (capitalized words)
        if re.match(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b', text):
            return 'name'
        
        # Default to generic
        return 'generic'

    def _generate_pseudonym(self, text, data_type):
        """Generate a pseudonym for the sensitive data"""
        # This would use a more sophisticated approach in a real implementation
        
        if data_type == 'email':
            return "john.doe@example.com"
        elif data_type == 'phone':
            return "(555) 123-4567"
        elif data_type == 'name':
            return "Jane Smith"
        elif data_type == 'ssn':
            return "123-45-6789"
        elif data_type == 'credit_card':
            return "1234-5678-9012-3456"
        else:
            return "EXAMPLE_DATA"

    def _generate_data_minimization_alternatives(self, text, violation, context=None):
        """
        Generate alternative text that minimizes the amount of personal or sensitive data.
        
        Args:
            text: Original text containing excessive data
            violation: The compliance violation
            context: Optional context information
            
        Returns:
            List of alternative text options that minimize data
        """
        alternatives = []
        
        # Get violation details
        violation_metadata = violation.get("metadata", {})
        severity = violation.get("severity", "medium")
        
        # Extract affected segments if available
        affected_segments = violation_metadata.get("affected_segments", [])
        
        # Strategy 1: Replace specific PII with generic terms
        generic_alternative = self._replace_pii_with_generic_terms(text, affected_segments)
        if generic_alternative != text:
            alternatives.append({
                "text": generic_alternative,
                "strategy": "generic_replacement",
                "confidence": 0.85,
                "description": "Replaced specific personal details with generic terms"
            })
        
        # Strategy 2: Abstract details to higher category levels
        abstracted_alternative = self._abstract_details(text, affected_segments)
        if abstracted_alternative != text and abstracted_alternative != generic_alternative:
            alternatives.append({
                "text": abstracted_alternative,
                "strategy": "abstraction",
                "confidence": 0.75,
                "description": "Abstracted specific details to higher-level categories"
            })
        
        # Strategy 3: Remove non-essential information
        if severity == "high":
            minimal_alternative = self._remove_nonessential_info(text, affected_segments)
            if minimal_alternative != text:
                alternatives.append({
                    "text": minimal_alternative,
                    "strategy": "information_reduction",
                    "confidence": 0.9,
                    "description": "Removed non-essential information to minimize data exposure"
                })
        
        # Strategy 4: Use placeholder references instead of full details
        placeholder_alternative = self._use_reference_placeholders(text, affected_segments)
        if placeholder_alternative != text:
            alternatives.append({
                "text": placeholder_alternative,
                "strategy": "reference_placeholders",
                "confidence": 0.8,
                "description": "Used placeholder references instead of specific details"
            })
        
        return alternatives

    def _generate_embedding_based_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives using embedding-based similarity"""
        # This would use embeddings to find semantically similar alternatives
        # Requires a sentence embedding model
        
        alternatives = []
        
        # Try using sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')  # or another suitable model
            
            for issue in rule_issues:
                # Get the problematic text
                if 'metadata' in issue and 'matched_content' in issue['metadata']:
                    matched_text = issue['metadata']['matched_content']
                    
                    # Find this text in the original content
                    import re
                    matches = list(re.finditer(re.escape(matched_text), text, re.IGNORECASE))
                    
                    if matches:
                        # Get alternative segments from a precomputed set
                        # In a real implementation, these would be generated dynamically
                        alternative_segments = self._get_compliant_alternatives_for_rule(rule_id)
                        
                        if alternative_segments:
                            # Encode the matched text
                            matched_embedding = model.encode(matched_text)
                            
                            # Encode all alternatives
                            alternative_embeddings = model.encode(alternative_segments)
                            
                            # Calculate similarities
                            similarities = []
                            for i, alt_embedding in enumerate(alternative_embeddings):
                                similarity = self._cosine_similarity(matched_embedding, alt_embedding)
                                similarities.append((i, similarity))
                            
                            # Sort by similarity
                            similarities.sort(key=lambda x: x[1], reverse=True)
                            
                            # Take top 3 most similar alternatives
                            for i, similarity in similarities[:3]:
                                alternative_segment = alternative_segments[i]
                                
                                # Replace in the original text
                                for match in matches:
                                    start, end = match.span()
                                    alternative_text = text[:start] + alternative_segment + text[end:]
                                    
                                    alternatives.append({
                                        'text': alternative_text,
                                        'rule_id': rule_id,
                                        'confidence': similarity * 0.9,  # Scale by similarity
                                        'type': 'embedding_based_replacement',
                                        'description': f"Replaced text with semantically similar compliant alternative"
                                    })
            
            return alternatives
        except:
            # Embedding model not available
            return []

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _get_compliant_alternatives_for_rule(self, rule_id):
        """Get a set of compliant alternatives for a specific rule"""
        # This would be populated from a database of pre-verified compliant alternatives
        # For now, return some examples
        
        alternatives_by_rule = {
            'prohibited_term_001': [
                "appropriate language",
                "suitable wording",
                "acceptable terminology"
            ],
            'bias_001': [
                "all people",
                "everyone",
                "all individuals"
            ],
            'compliance_001': [
                "in accordance with regulations",
                "following proper procedures",
                "in compliance with policies"
            ]
        }
        
        return alternatives_by_rule.get(rule_id, [])

    def _generate_transformer_based_alternatives(self, text, rule_id, rule_issues, context):
        """Generate alternatives using transformer models"""
        # This would use a language model to generate compliant alternatives
        # Requires a language model like GPT or T5
        
        alternatives = []
        
        # Try using transformers if available
        try:
            from transformers import pipeline
            
            # Initialize a text generation pipeline
            generator = pipeline('text2text-generation', model='t5-small')
            
            for issue in rule_issues:
                # Get the problematic text
                if 'metadata' in issue and 'matched_content' in issue['metadata']:
                    matched_text = issue['metadata']['matched_content']
                    
                    # Find this text in the original content
                    import re
                    matches = list(re.finditer(re.escape(matched_text), text, re.IGNORECASE))
                    
                    if matches:
                        # Create prompt for the model
                        rule_name = self.rule_metadata.get(rule_id, {}).get('name', f"Rule {rule_id}")
                        prompt = f"Rewrite the following text to comply with {rule_name}: {matched_text}"
                        
                        # Generate alternative
                        result = generator(prompt, max_length=100, num_return_sequences=1)
                        
                        if result and len(result) > 0:
                            alternative_segment = result[0]['generated_text']
                            
                            # Replace in the original text
                            for match in matches:
                                start, end = match.span()
                                alternative_text = text[:start] + alternative_segment + text[end:]
                                
                                alternatives.append({
                                    'text': alternative_text,
                                    'rule_id': rule_id,
                                    'confidence': 0.7,  # Lower confidence since it's automated
                                    'type': 'transformer_based_replacement',
                                    'description': f"Used language model to generate compliant alternative"
                                })
            
            return alternatives
        except:
            # Transformer model not available
            return []

    def _apply_general_alternatives(self, text, issues, context):
        """
        Apply general strategies for generating alternatives when rule-specific
        strategies are not available.
        
        Uses NLP techniques like text simplification, general rewording, or removal
        of problematic segments to address compliance issues.
        
        Args:
            text: Original text content
            issues: Compliance issues detected
            context: Optional context information
            
        Returns:
            List of alternative formulations
        """
        alternatives = []
        
        # 1. Remove problematic segments
        removal_alt = self._apply_segment_removal(text, issues)
        if removal_alt:
            alternatives.append(removal_alt)
        
        # 2. Apply hedging language
        hedging_alt = self._apply_hedging_language(text, issues)
        if hedging_alt:
            alternatives.append(hedging_alt)
        
        # 3. Text simplification
        simplification_alt = self._apply_text_simplification(text, issues)
        if simplification_alt:
            alternatives.append(simplification_alt)
        
        # 4. General paraphrasing
        paraphrase_alt = self._apply_paraphrasing(text, issues)
        if paraphrase_alt:
            alternatives.append(paraphrase_alt)
        
        # 5. Change tone (more formal/neutral)
        tone_alt = self._apply_tone_change(text, issues)
        if tone_alt:
            alternatives.append(tone_alt)
        
        # 6. Add clarifying context
        context_alt = self._add_clarifying_context(text, issues)
        if context_alt:
            alternatives.append(context_alt)
        
        # 7. Restructure sentences
        restructure_alt = self._restructure_sentences(text, issues)
        if restructure_alt:
            alternatives.append(restructure_alt)
        
        return alternatives

    def _apply_segment_removal(self, text, issues):
        """Apply strategy: Remove problematic segments"""
        # Identify segments to remove
        segments_to_remove = []
        
        for issue in issues:
            if 'metadata' in issue and 'location' in issue['metadata']:
                start = issue['metadata']['location'].get('start', -1)
                end = issue['metadata']['location'].get('end', -1)
                
                if start >= 0 and end > start and end <= len(text):
                    segments_to_remove.append((start, end))
        
        if not segments_to_remove:
            # Try to extract from matched_content if location not available
            for issue in issues:
                if 'metadata' in issue and 'matched_content' in issue['metadata']:
                    matched_content = issue['metadata']['matched_content']
                    
                    # Find in text
                    import re
                    for match in re.finditer(re.escape(matched_content), text):
                        segments_to_remove.append(match.span())
        
        if segments_to_remove:
            # Sort segments in reverse order so removal doesn't affect indices
            segments_to_remove.sort(reverse=True)
            
            # Apply removals
            modified_text = text
            for start, end in segments_to_remove:
                modified_text = modified_text[:start] + modified_text[end:]
            
            # Only return if text was actually modified
            if modified_text != text:
                return {
                    'text': modified_text,
                    'confidence': 0.7,
                    'type': 'segment_removal',
                    'description': f"Removed {len(segments_to_remove)} problematic segments"
                }
        
        return None

    def _apply_hedging_language(self, text, issues):
        """Apply strategy: Add hedging language to questionable statements"""
        # Hedging phrases to insert before problematic statements
        hedging_phrases = [
            "It is commonly suggested that ",
            "Some sources indicate that ",
            "According to certain perspectives, ",
            "It might be considered that ",
            "In some contexts, ",
            "From one point of view, "
        ]
        
        # Find problematic statements
        problematic_statements = []
        
        # Try to identify sentence boundaries around issues
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for issue in issues:
            if 'metadata' in issue and 'matched_content' in issue['metadata']:
                matched_content = issue['metadata']['matched_content']
                
                # Find which sentence contains this content
                for i, sentence in enumerate(sentences):
                    if matched_content in sentence:
                        problematic_statements.append((i, sentence))
                        break
        
        if problematic_statements:
            # Deduplicate
            problematic_statements = list(set(problematic_statements))
            
            # Apply hedging to sentences
            import random
            modified_sentences = sentences.copy()
            
            for idx, _ in problematic_statements:
                if idx < len(modified_sentences):
                    # Choose a random hedging phrase
                    hedge = random.choice(hedging_phrases)
                    
                    # Apply hedging to start of sentence if it doesn't already have hedging
                    if not any(h.lower() in modified_sentences[idx].lower() for h in hedging_phrases):
                        # Capitalize first letter after hedging
                        sentence = modified_sentences[idx]
                        if sentence and sentence[0].isupper():
                            sentence = sentence[0].lower() + sentence[1:]
                        modified_sentences[idx] = hedge + sentence
            
            # Reconstruct text
            modified_text = " ".join(modified_sentences)
            
            # Only return if text was actually modified
            if modified_text != text:
                return {
                    'text': modified_text,
                    'confidence': 0.6,
                    'type': 'hedging_language',
                    'description': f"Added hedging language to {len(problematic_statements)} statements"
                }
        
        return None

    def _apply_text_simplification(self, text, issues):
        """Apply strategy: Simplify text to reduce complexity and potential issues"""
        # Text simplification requires more sophisticated NLP
        # This is a placeholder that would use text simplification models
        
        # Try using a simple rule-based approach for demonstration
        complex_words = {
            'utilize': 'use',
            'implement': 'use',
            'leverage': 'use',
            'facilitate': 'help',
            'furthermore': 'also',
            'additionally': 'also',
            'consequently': 'so',
            'subsequently': 'later',
            'nevertheless': 'still',
            'accordingly': 'so',
            'furthermore': 'also',
            'notwithstanding': 'despite',
            'aforementioned': 'this',
            'heretofore': 'until now'
        }
        
        # Simplify by replacing complex words
        simplified_text = text
        for complex_word, simple_word in complex_words.items():
            # Use word boundaries to avoid partial replacements
            simplified_text = re.sub(
                r'\b' + re.escape(complex_word) + r'\b', 
                simple_word, 
                simplified_text, 
                flags=re.IGNORECASE
            )
        
        # Only return if text was actually modified
        if simplified_text != text:
            return {
                'text': simplified_text,
                'confidence': 0.5,
                'type': 'text_simplification',
                'description': "Simplified text by replacing complex words with simpler alternatives"
            }
        
        return None

    def _apply_paraphrasing(self, text, issues):
        
        """Apply strategy: Paraphrase content to address issues"""
        # Full paraphrasing requires advanced NLP models
        # This is a placeholder that would use paraphrasing models
        
        # In a real implementation, this would use a text-to-text model
        # to paraphrase the content while preserving meaning
        try:
            from transformers import pipeline
            generator = pipeline('text2text-generation', model='t5-small')
            
            # Only paraphrase smaller texts due to model limitations
            if len(text) <= 512:
                # Generate paraphrase
                result = generator(f"paraphrase: {text}", max_length=512, num_return_sequences=1)
                
                if result and len(result) > 0:
                    paraphrased_text = result[0]['generated_text']
                    
                    # Only return if text was meaningfully modified
                    if paraphrased_text != text and len(paraphrased_text) > len(text) * 0.5:
                        return {
                            'text': paraphrased_text,
                            'confidence': 0.5,
                            'type': 'paraphrasing',
                            'description': "Paraphrased content to address compliance issues"
                        }
        except:
            # Paraphrasing model not available
            pass
        
        return None
    def _apply_tone_change(self, text, target_tone, context=None):
        """
        Adjust the tone of text while preserving meaning.
        
        Args:
            text: Original text content
            target_tone: Desired tone ('formal', 'casual', 'neutral', etc.)
            context: Optional context information
                
        Returns:
            Text with adjusted tone
        """
        if not text:
            return text
            
        # Analyze current tone
        current_tone = self._analyze_text_tone(text)
        
        # If already close to target tone, return original
        if current_tone == target_tone:
            return text
            
        # Break text into sentences for easier processing
        sentences = self._split_into_sentences(text)
        
        # Process each sentence based on target tone
        adjusted_sentences = []
        
        for sentence in sentences:
            if target_tone == "formal":
                adjusted = self._formalize_text(sentence, current_tone)
            elif target_tone == "casual":
                adjusted = self._casualize_text(sentence, current_tone)
            elif target_tone == "neutral":
                adjusted = self._neutralize_text(sentence, current_tone)
            elif target_tone == "technical":
                adjusted = self._technicalize_text(sentence, current_tone)
            elif target_tone == "simple":
                adjusted = self._simplify_text(sentence, current_tone)
            else:
                adjusted = sentence  # No change for unknown tones
                
            adjusted_sentences.append(adjusted)
        
        # Reconstruct text with original paragraph structure
        return self._reconstruct_text(text, sentences, adjusted_sentences)
        
    def _analyze_text_tone(self, text):
        """Analyze the current tone of text"""
        text_lower = text.lower()
        
        # Count tone indicators
        formal_indicators = self._count_indicators(text_lower, self._get_formal_indicators())
        casual_indicators = self._count_indicators(text_lower, self._get_casual_indicators())
        technical_indicators = self._count_indicators(text_lower, self._get_technical_indicators())
        
        # Calculate sentence complexity (proxy for formality)
        avg_sentence_length = self._calculate_avg_sentence_length(text)
        has_complex_sentences = avg_sentence_length > 20
        
        # Determine dominant tone
        if technical_indicators > 3 or (technical_indicators > 0 and formal_indicators > 2):
            return "technical"
        elif formal_indicators > casual_indicators and has_complex_sentences:
            return "formal"
        elif casual_indicators > formal_indicators:
            return "casual"
        else:
            return "neutral"
            
    def _count_indicators(self, text_lower, indicators):
        """Count occurrences of tone indicators in text"""
        count = 0
        for indicator in indicators:
            if indicator in text_lower:
                count += 1
        return count
        
    def _get_formal_indicators(self):
        """Get indicators of formal tone"""
        return [
            "furthermore", "moreover", "thus", "therefore", "consequently",
            "accordingly", "hereby", "herein", "wherein", "therein",
            "pursuant to", "in accordance with", "with respect to",
            "notwithstanding", "nevertheless", "however", "additionally",
            "subsequently", "shall", "must", "may not", "in conclusion"
        ]
        
    def _get_casual_indicators(self):
        """Get indicators of casual tone"""
        return [
            "yeah", "cool", "awesome", "basically", "kinda", "sorta",
            "you know", "like", "stuff", "things", "anyway", "anyways",
            "so", "just", "pretty", "really", "very", "totally", "literally",
            "actually", "btw", "fyi", "ok", "okay", "guys", "folks",
            "gonna", "wanna", "gotta", "dunno", "y'all", "yep", "nope"
        ]
        
    def _get_technical_indicators(self):
        """Get indicators of technical tone"""
        return [
            "implementation", "methodology", "functionality", "algorithm",
            "infrastructure", "architecture", "configuration", "parameters",
            "optimization", "specification", "interface", "protocol",
            "component", "module", "system", "framework", "paradigm",
            "technical", "analytical", "quantitative", "qualitative",
            "statistical", "empirical", "theoretical", "conceptual"
        ]
        
    def _calculate_avg_sentence_length(self, text):
        """Calculate average sentence length in words"""
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return 0
            
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)
        
    def _split_into_sentences(self, text):
        """Split text into sentences"""
        # Simple sentence splitting (could be improved with NLP)
        sentence_endings = re.compile(r'[.!?][\s)]')
        
        # Find all potential sentence endings
        end_positions = [m.start() + 1 for m in sentence_endings.finditer(text)]
        
        if not end_positions:
            return [text]
            
        # Build sentences
        sentences = []
        start = 0
        
        for end in end_positions:
            sentences.append(text[start:end].strip())
            start = end
            
        # Add the last sentence if needed
        if start < len(text):
            sentences.append(text[start:].strip())
            
        return sentences
        
    def _formalize_text(self, text, current_tone):
        """Make text more formal"""
        if current_tone == "formal":
            return text
            
        # Replace casual phrases with formal alternatives
        casual_to_formal = {
            "a lot of": "numerous",
            "lots of": "numerous",
            "kind of": "somewhat",
            "sort of": "somewhat",
            "really": "significantly",
            "very": "substantially",
            "pretty": "fairly",
            "big": "substantial",
            "huge": "significant",
            "get": "obtain",
            "got": "obtained",
            "show": "demonstrate",
            "think": "consider",
            "use": "utilize",
            "help": "assist",
            "start": "commence",
            "end": "conclude",
            "make sure": "ensure",
            "find out": "determine",
            "look into": "investigate",
            "deal with": "address",
            "keep up": "maintain",
            "set up": "establish",
            "check": "examine",
            "okay": "acceptable",
            "ok": "acceptable",
            "kids": "children",
            "guys": "individuals",
            "stuff": "materials",
            "things": "items",
            "a bit": "slightly",
            "I think": "I believe",
            "I feel": "I perceive",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "like": "such as",
            "maybe": "perhaps",
            "yeah": "yes",
            "yep": "yes",
            "nope": "no"
        }
        
        result = text
        
        # Replace casual phrases with formal equivalents
        for casual, formal in casual_to_formal.items():
            result = re.sub(r'\b' + re.escape(casual) + r'\b', formal, result, flags=re.IGNORECASE)
        
        # Remove excessive exclamation marks
        result = re.sub(r'!+', '.', result)
        
        # Fix starting sentences with conjunctions
        for conjunction in ["And", "But", "So", "Or"]:
            result = re.sub(r'^\s*' + conjunction + r'\b', "", result)
            result = re.sub(r'[.!?]\s+' + conjunction + r'\b', ".", result)
        
        # Avoid first person (casual) in formal writing
        if "I " in result or "I'm " in result or "I've " in result:
            result = result.replace("I ", "One ")
            result = result.replace("I'm ", "One is ")
            result = result.replace("I've ", "One has ")
        
        # Avoid contractions in formal writing
        for contraction, expanded in [("I'm", "I am"), ("you're", "you are"), 
                                    ("we're", "we are"), ("they're", "they are")]:
            result = result.replace(contraction, expanded)
            
        return result
        
    def _casualize_text(self, text, current_tone):
        """Make text more casual"""
        if current_tone == "casual":
            return text
            
        # Replace formal phrases with casual alternatives
        formal_to_casual = {
            "furthermore": "also",
            "moreover": "plus",
            "thus": "so",
            "therefore": "so",
            "consequently": "so",
            "however": "but",
            "nevertheless": "still",
            "approximately": "about",
            "obtain": "get",
            "purchase": "buy",
            "require": "need",
            "utilize": "use",
            "implement": "use",
            "demonstrate": "show",
            "determine": "figure out",
            "sufficient": "enough",
            "inquire": "ask",
            "numerous": "lots of",
            "individuals": "people",
            "assist": "help",
            "commence": "start",
            "conclude": "end",
            "terminate": "end",
            "comprehend": "understand",
            "additional": "more",
            "adequate": "enough",
            "subsequently": "later",
            "facilitate": "help",
            "initial": "first",
            "personnel": "people",
            "regarding": "about",
            "request": "ask",
            "attempt": "try",
            "endeavor": "try",
            "forward": "send",
            "in the event that": "if",
            "regarding": "about",
            "in the amount of": "for",
            "with regard to": "about",
            "despite the fact that": "though",
            "due to the fact that": "because",
            "in light of the fact that": "because",
            "in the event that": "if",
            "it is necessary that": "must",
            "it is important that": "should"
        }
        
        result = text
        
        # Replace formal phrases with casual equivalents
        for formal, casual in formal_to_casual.items():
            result = re.sub(r'\b' + re.escape(formal) + r'\b', casual, result, flags=re.IGNORECASE)
        
        # Convert passive voice to active (a more casual style)
        passive_patterns = [
            (r'(is|are|was|were) being ([\w]+ed)', "{subject} {tense} {verb}"),
            (r'(has|have|had) been ([\w]+ed)', "{subject} {tense} {verb}")
        ]
        
        # Add contractions for more casual tone
        result = result.replace("cannot", "can't")
        result = result.replace("will not", "won't")
        result = result.replace("do not", "don't")
        result = result.replace("is not", "isn't")
        result = result.replace("are not", "aren't")
        result = result.replace("was not", "wasn't")
        result = result.replace("were not", "weren't")
        result = result.replace("have not", "haven't")
        result = result.replace("has not", "hasn't")
        result = result.replace("would not", "wouldn't")
        result = result.replace("could not", "couldn't")
        result = result.replace("should not", "shouldn't")
        
        # Simplify complex sentences
        if len(result.split()) > 25:
            # Very simple sentence splitting - would be better with NLP
            result = result.replace("; ", ". ")
            result = result.replace(", which ", ". This ")
            result = result.replace(", and ", ". And ")
            
        # Add a casual opener for very formal text
        if current_tone == "formal" or current_tone == "technical":
            if not result.startswith("So ") and not result.startswith("Basically "):
                result = "Basically, " + result[0].lower() + result[1:]
                
        return result
        
    def _neutralize_text(self, text, current_tone):
        """Make text more neutral in tone"""
        if current_tone == "neutral":
            return text
            
        result = text
        
        if current_tone == "formal":
            # Make formal text more neutral
            formal_to_neutral = {
                "furthermore": "also",
                "moreover": "additionally",
                "thus": "therefore",
                "herein": "here",
                "wherein": "where",
                "aforementioned": "mentioned",
                "utilize": "use",
                "pursuant to": "according to",
                "in accordance with": "following",
                "commence": "begin",
                "terminate": "end"
            }
            
            for formal, neutral in formal_to_neutral.items():
                result = re.sub(r'\b' + re.escape(formal) + r'\b', neutral, result, flags=re.IGNORECASE)
                
        elif current_tone == "casual":
            # Make casual text more neutral
            casual_to_neutral = {
                "really": "",
                "very": "",
                "totally": "",
                "basically": "",
                "literally": "",
                "actually": "",
                "just": "",
                "kinda": "somewhat",
                "sorta": "somewhat",
                "guys": "people",
                "stuff": "items",
                "things": "factors",
                "lots of": "many",
                "a lot of": "many",
                "huge": "significant",
                "awesome": "excellent",
                "cool": "good",
                "yeah": "yes",
                "nope": "no"
            }
            
            for casual, neutral in casual_to_neutral.items():
                if neutral:
                    result = re.sub(r'\b' + re.escape(casual) + r'\b', neutral, result, flags=re.IGNORECASE)
                else:
                    # For empty replacements (removing words like "really"), be careful with spacing
                    result = re.sub(r'\b' + re.escape(casual) + r'\s', "", result, flags=re.IGNORECASE)
                    
            # Expand contractions
            result = result.replace("can't", "cannot")
            result = result.replace("won't", "will not")
            result = result.replace("don't", "do not")
            result = result.replace("isn't", "is not")
            result = result.replace("aren't", "are not")
            
        elif current_tone == "technical":
            # Make technical text more neutral
            technical_to_neutral = {
                "implementation": "process",
                "methodology": "method",
                "functionality": "function",
                "algorithm": "procedure",
                "infrastructure": "structure",
                "architecture": "design",
                "configuration": "setup",
                "parameters": "settings",
                "optimization": "improvement",
                "interface": "connection"
            }
            
            for technical, neutral in technical_to_neutral.items():
                result = re.sub(r'\b' + re.escape(technical) + r'\b', neutral, result, flags=re.IGNORECASE)
        
        return result
        
    def _technicalize_text(self, text, current_tone):
        """Make text more technical"""
        if current_tone == "technical":
            return text
            
        # Replace common terms with technical alternatives
        common_to_technical = {
            "use": "utilize",
            "make": "implement",
            "do": "execute",
            "look at": "analyze",
            "check": "validate",
            "fix": "resolve",
            "change": "modify",
            "improve": "optimize",
            "increase": "augment",
            "decrease": "reduce",
            "show": "demonstrate",
            "see": "observe",
            "tell": "communicate",
            "find": "identify",
            "help": "facilitate",
            "stop": "terminate",
            "start": "initiate",
            "end": "finalize",
            "connect": "interface",
            "need": "require",
            "problem": "issue",
            "solution": "resolution",
            "idea": "concept",
            "plan": "strategy",
            "group": "cluster",
            "part": "component",
            "tool": "utility",
            "software": "application",
            "hardware": "infrastructure",
            "settings": "configuration",
            "features": "functionality",
            "user": "end user",
            "computer": "system",
            "network": "infrastructure",
            "website": "web application",
            "app": "application",
            "bug": "defect",
            "testing": "quality assurance",
            "update": "upgrade"
        }
        
        result = text
        
        # Replace common terms with technical equivalents
        for common, technical in common_to_technical.items():
            result = re.sub(r'\b' + re.escape(common) + r'\b', technical, result, flags=re.IGNORECASE)
        
        # Add technical phrases for more technical tone
        if not any(phrase in result.lower() for phrase in ["in order to", "with respect to", "in accordance with"]):
            # Add a technical phrase if none exists
            if "for" in result:
                result = result.replace(" for ", " in order to ")
            elif "about" in result:
                result = result.replace(" about ", " with respect to ")
                
        # Expand contractions for more formal technical tone
        result = result.replace("can't", "cannot")
        result = result.replace("won't", "will not")
        result = result.replace("don't", "do not")
        
        # If sentence doesn't sound technical enough, add a technical opener
        lower_result = result.lower()
        technical_phrases = ["therefore", "thus", "consequently", "accordingly", "subsequently"]
        
        if not any(phrase in lower_result for phrase in technical_phrases):
            if result.startswith("This "):
                result = "Subsequently, this" + result[4:]
            elif result.startswith("The "):
                result = "Accordingly, the" + result[3:]
            elif result.startswith("We "):
                result = "Consequently, we" + result[2:]
            elif result.startswith("I "):
                result = "Based on analysis, I" + result[1:]
                
        return result
        
    def _simplify_text(self, text, current_tone):
        """Make text simpler and easier to understand"""
        if current_tone == "simple":
            return text
            
        # Replace complex terms with simpler alternatives
        complex_to_simple = {
            "utilize": "use",
            "implement": "use",
            "execute": "do",
            "facilitate": "help",
            "sufficient": "enough",
            "demonstrate": "show",
            "modify": "change",
            "optimize": "improve",
            "augment": "increase",
            "terminate": "end",
            "initiate": "start",
            "subsequently": "later",
            "previously": "before",
            "additionally": "also",
            "consequently": "so",
            "furthermore": "also",
            "approximately": "about",
            "numerous": "many",
            "obtain": "get",
            "purchase": "buy",
            "require": "need",
            "inquire": "ask",
            "comprehend": "understand",
            "endeavor": "try",
            "attempt": "try",
            "assist": "help",
            "regarding": "about",
            "nevertheless": "still",
            "notwithstanding": "even though",
            "accordingly": "so",
            "therefore": "so",
            "although": "though",
            "however": "but",
            "in addition": "also",
            "consequently": "so",
            "despite": "even though",
            "due to": "because of",
            "in order to": "to",
            "in the event that": "if",
            "prior to": "before",
            "subsequent to": "after",
            "with regard to": "about"
        }
        
        result = text
        
        # Replace complex words and phrases with simpler alternatives
        for complex_term, simple_term in complex_to_simple.items():
            result = re.sub(r'\b' + re.escape(complex_term) + r'\b', simple_term, result, flags=re.IGNORECASE)
        
        # Break long sentences
        sentences = self._split_into_sentences(result)
        simplified_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 15:
                # Break long sentences at logical points
                midpoint = len(words) // 2
                
                # Look for logical break points
                break_points = []
                for i in range(midpoint - 3, midpoint + 3):
                    if i > 0 and i < len(words) - 1:
                        if words[i].lower() in ["and", "but", "or", "because", "however", "so"]:
                            break_points.append(i)
                
                if break_points:
                    # Use the break point closest to midpoint
                    break_point = min(break_points, key=lambda x: abs(x - midpoint))
                    
                    # Split sentence at break point
                    first_half = " ".join(words[:break_point])
                    if not first_half.endswith("."):
                        first_half += "."
                        
                    second_half = words[break_point].capitalize() + " " + " ".join(words[break_point+1:])
                    
                    simplified_sentences.append(first_half)
                    simplified_sentences.append(second_half)
                else:
                    simplified_sentences.append(sentence)
            else:
                simplified_sentences.append(sentence)
        
        # Recombine simplified sentences
        result = " ".join(simplified_sentences)
        
        # Use active voice instead of passive
        passive_patterns = [
            (r'(is|are|was|were) ([\w]+ed) by', "SUBJECT VERB"),
            (r'(has|have|had) been ([\w]+ed)', "SUBJECT VERB")
        ]
        
        # This is a simplified approach - full passive->active conversion would require NLP
        for pattern, _ in passive_patterns:
            if re.search(pattern, result):
                # Flag that passive voice should be reviewed
                result += " [Simplified version would use active voice instead of passive voice.]"
                break
        
        # Add contractions for more natural simple language
        result = result.replace("cannot", "can't")
        result = result.replace("will not", "won't")
        result = result.replace("do not", "don't")
        result = result.replace("is not", "isn't")
        result = result.replace("are not", "aren't")
        
        return result
        
    def _reconstruct_text(self, original_text, original_sentences, adjusted_sentences):
        """Reconstruct text preserving original paragraph structure"""
        if len(original_sentences) != len(adjusted_sentences):
            # Fallback to simple concatenation if sentence counts don't match
            return " ".join(adjusted_sentences)
            
        # Create mapping of original sentences to their positions in text
        sentence_positions = []
        pos = 0
        
        for sentence in original_sentences:
            # Find sentence in original text (handling potential whitespace differences)
            sentence_stripped = sentence.strip()
            while pos < len(original_text):
                if original_text[pos:].strip().startswith(sentence_stripped):
                    start = pos
                    end = pos + len(sentence)
                    sentence_positions.append((start, end))
                    pos = end
                    break
                pos += 1
        
        # Reconstruct text by replacing sentences while preserving structure
        if len(sentence_positions) == len(adjusted_sentences):
            result = list(original_text)  # Convert to list for char-by-char replacement
            
            # Replace sentences in reverse order to avoid index shifting
            for (start, end), new_sentence in zip(reversed(sentence_positions), reversed(adjusted_sentences)):
                # Replace while preserving any whitespace at the end
                old_sentence = original_text[start:end]
                trailing_whitespace = ""
                for char in reversed(old_sentence):
                    if char.isspace():
                        trailing_whitespace = char + trailing_whitespace
                    else:
                        break
                        
                replacement = new_sentence + trailing_whitespace
                result[start:end] = replacement
                
            return "".join(result)
        else:
            # Fallback to simple concatenation if mapping failed
            return " ".join(adjusted_sentences)

            
    def _add_clarifying_context(self, text, issues):
        """Apply strategy: Add clarifying context to address issues"""
        # This would add explanatory context based on the issues
        # For now, return a simple example
        
        # Create disclaimer based on issue types
        disclaimers = set()
        
        for issue in issues:
            rule_id = issue.get('rule_id', '')
            
            if 'pii' in rule_id.lower():
                disclaimers.add("Please note that any personal identifiers mentioned are examples only and should be replaced with appropriate data in real usage.")
            elif 'health' in rule_id.lower() or 'medical' in rule_id.lower():
                disclaimers.add("Note: This information is not medical advice. Consult with healthcare professionals for specific guidance.")
            elif 'financial' in rule_id.lower():
                disclaimers.add("Disclaimer: This information is not financial advice. Consult with financial professionals for specific guidance.")
        
        if disclaimers:
            # Add disclaimers at the end
            modified_text = text
            for disclaimer in disclaimers:
                if disclaimer not in modified_text:
                    modified_text += f"\n\n{disclaimer}"
            
            return {
                'text': modified_text,
                'confidence': 0.7,
                'type': 'clarifying_context',
                'description': f"Added {len(disclaimers)} clarifying disclaimer(s)"
            }
        
        return None

    def _restructure_sentences(self, text, issues):
        """Apply strategy: Restructure sentences to address issues"""
        # Sentence restructuring requires advanced NLP
        # This is a placeholder for a more sophisticated implementation
        
        # Return placeholder suggestion
        return {
            'text': "[Restructured version would be generated here - requires NLP model]",
            'confidence': 0.4,
            'type': 'sentence_restructuring',
            'description': "Suggested restructuring sentences to address compliance issues"
        }
        
    def _replace_pii_with_generic_terms(self, text, affected_segments=None):
        """Replace specific PII with generic terms"""
        # If we have specific affected segments, focus on those
        if affected_segments and len(affected_segments) > 0:
            modified_text = text
            # Process segments in reverse order to avoid index shifting
            for segment in sorted(affected_segments, key=lambda x: x.get("start", 0), reverse=True):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                pii_type = segment.get("type", "").lower()
                
                if start < end and end <= len(modified_text):
                    replacement = self._get_generic_term_for_pii(pii_type)
                    modified_text = modified_text[:start] + replacement + modified_text[end:]
            
            return modified_text
        
        # Without specific segments, use pattern-based replacement
        modified_text = text
        
        # Replace email addresses
        modified_text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "[EMAIL ADDRESS]",
            modified_text
        )
        
        # Replace phone numbers
        modified_text = re.sub(
            r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "[PHONE NUMBER]",
            modified_text
        )
        
        # Replace dates of birth
        modified_text = re.sub(
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            "[DATE]",
            modified_text
        )
        
        # Replace addresses (basic pattern)
        modified_text = re.sub(
            r'\b\d+\s+[A-Za-z0-9\s,]+(St|Street|Rd|Road|Ave|Avenue|Blvd|Boulevard)[,\s]+[A-Za-z\s]+(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)(?:\s+\d{5}(?:-\d{4})?)?\b',
            "[ADDRESS]",
            modified_text,
            flags=re.IGNORECASE
        )
        
        # Replace SSNs
        modified_text = re.sub(
            r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
            "[SOCIAL SECURITY NUMBER]",
            modified_text
        )
        
        return modified_text

    def _get_generic_term_for_pii(self, pii_type):
        """Get appropriate generic term based on PII type"""
        generic_terms = {
            "email": "[EMAIL ADDRESS]",
            "phone": "[PHONE NUMBER]",
            "address": "[ADDRESS]",
            "ssn": "[SOCIAL SECURITY NUMBER]",
            "dob": "[DATE OF BIRTH]",
            "date_of_birth": "[DATE OF BIRTH]",
            "credit_card": "[PAYMENT INFORMATION]",
            "name": "[NAME]",
            "person": "[PERSON]",
            "location": "[LOCATION]",
            "age": "[AGE]",
            "id": "[ID]",
            "identification": "[ID NUMBER]",
            "account": "[ACCOUNT NUMBER]",
            "license": "[LICENSE NUMBER]",
            "passport": "[PASSPORT NUMBER]",
            "patient": "[PATIENT]",
            "medical": "[MEDICAL INFORMATION]",
            "health": "[HEALTH INFORMATION]",
            "financial": "[FINANCIAL INFORMATION]",
            "salary": "[INCOME INFORMATION]",
            "income": "[INCOME INFORMATION]",
            # Default catch-all
            "default": "[PERSONAL DATA]"
        }
        
        return generic_terms.get(pii_type, generic_terms["default"])

    def _abstract_details(self, text, affected_segments=None):
        """Abstract specific details to higher-level categories"""
        # If we have specific affected segments, focus on those
        if affected_segments and len(affected_segments) > 0:
            modified_text = text
            # Process segments in reverse order to avoid index shifting
            for segment in sorted(affected_segments, key=lambda x: x.get("start", 0), reverse=True):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                pii_type = segment.get("type", "").lower()
                original_text = text[start:end] if (start < end and end <= len(text)) else ""
                
                if original_text:
                    abstraction = self._get_abstraction_for_detail(original_text, pii_type)
                    modified_text = modified_text[:start] + abstraction + modified_text[end:]
            
            return modified_text
        
        # Without specific segments, use heuristic-based abstraction
        # This implementation would use more advanced NLP techniques in production
        modified_text = text
        
        # Abstract names to roles where contextually appropriate
        name_patterns = [
            (r'\b(Dr\.|Mr\.|Mrs\.|Ms\.) [A-Z][a-z]+ [A-Z][a-z]+\b', "the individual"),
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', "the person")
        ]
        
        for pattern, replacement in name_patterns:
            modified_text = re.sub(pattern, replacement, modified_text)
        
        # Abstract specific ages to age ranges
        modified_text = re.sub(r'\b(\d{1,2}) years old\b', "in their age group", modified_text)
        modified_text = re.sub(r'\bage (\d{1,2})\b', "in their age group", modified_text)
        
        # Abstract specific locations to regional areas
        # (A more comprehensive implementation would use gazetteer lookups)
        city_state_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)?, [A-Z]{2}\b'
        modified_text = re.sub(city_state_pattern, "in the region", modified_text)
        
        return modified_text
    
    def _get_abstraction_for_detail(self, detail_text, detail_type=None):
        """Get appropriate abstraction based on the type of detail"""
        # Detect type if not provided
        if not detail_type:
            detail_type = self._infer_detail_type(detail_text)
        
        # Apply appropriate abstraction based on type
        if detail_type in ["name", "person"]:
            return "the individual"
        elif detail_type in ["address", "location"]:
            return "the location"
        elif detail_type in ["dob", "date_of_birth", "age"]:
            return "their age"
        elif detail_type in ["email"]:
            return "their contact information"
        elif detail_type in ["phone"]:
            return "their contact information"
        elif detail_type in ["medical", "health"]:
            return "their health status"
        elif detail_type in ["financial", "salary", "income"]:
            return "their financial information"
        else:
            return "the information"

    def _infer_detail_type(self, text):
        """Infer the type of detail from text"""
        # Simple rule-based inference
        text_lower = text.lower()
        
        if re.search(r'@', text):
            return "email"
        elif re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
            return "phone"
        elif re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text):
            return "date"
        elif re.search(r'\b\d+ .*(St|Road|Avenue|Blvd)\b', text, re.IGNORECASE):
            return "address"
        elif re.search(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text):
            return "ssn"
        elif re.search(r'\b\d{1,2}\b', text) and ("age" in text_lower or "year" in text_lower):
            return "age"
        else:
            return "default"
        
        if re.search(r'@', text):
            return "email"
        elif re.search(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text):
            return "phone"
        elif re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text):
            return "date"
        elif re.search(r'\b\d+ .*(St|Road|Avenue|Blvd)\b', text, re.IGNORECASE):
            return "address"
        elif re.search(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', text):
            return "ssn"
        elif re.search(r'\b\d{1,2}\b', text) and ("age" in text_lower or "year" in text_lower):
            return "age"
        else:
            return "default"


    def _remove_nonessential_info(self, text, affected_segments=None):
        """Remove non-essential information to minimize data"""
        if affected_segments and len(affected_segments) > 0:
            modified_text = text
            # Process segments in reverse order to avoid index shifting
            for segment in sorted(affected_segments, key=lambda x: x.get("start", 0), reverse=True):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                
                if start < end and end <= len(modified_text):
                    # For highly sensitive information, complete removal might be appropriate
                    modified_text = modified_text[:start] + modified_text[end:]
            
            return modified_text
        
        # Without specific segments, use sentence-level analysis
        # In a production system, this would use NLP to identify non-essential sentences
        sentences = text.split(". ")
        essential_sentences = []
        
        for sentence in sentences:
            # Skip sentences with obvious PII patterns
            if (re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', sentence) or  # email
                re.search(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', sentence) or  # SSN
                re.search(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', sentence) or  # phone
                re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', sentence)):  # date
                continue
            
            essential_sentences.append(sentence)
        
        return ". ".join(essential_sentences) + ("." if essential_sentences else "")


    def _use_reference_placeholders(self, text, affected_segments=None):
        """Use reference placeholders instead of specific details"""
        if affected_segments and len(affected_segments) > 0:
            modified_text = text
            placeholder_map = {}
            
            # Process segments in reverse order to avoid index shifting
            for i, segment in enumerate(sorted(affected_segments, key=lambda x: x.get("start", 0), reverse=True)):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                pii_type = segment.get("type", "").lower()
                
                if start < end and end <= len(modified_text):
                    # Create reference placeholder
                    placeholder_id = f"{pii_type.upper()}_{i+1}" if pii_type else f"REF_{i+1}"
                    placeholder = f"[{placeholder_id}]"
                    
                    # Store original for reference
                    placeholder_map[placeholder] = text[start:end]
                    
                    # Replace in text
                    modified_text = modified_text[:start] + placeholder + modified_text[end:]
            
            return modified_text
        
        # Without specific segments, use pattern-based placeholders
        modified_text = text
        placeholder_count = 1
        
        # Replace emails
        email_matches = list(re.finditer(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', modified_text))
        for match in reversed(email_matches):
            placeholder = f"[EMAIL_{placeholder_count}]"
            modified_text = modified_text[:match.start()] + placeholder + modified_text[match.end():]
            placeholder_count += 1
        
        # Replace phone numbers
        phone_matches = list(re.finditer(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', modified_text))
        for match in reversed(phone_matches):
            placeholder = f"[PHONE_{placeholder_count}]"
            modified_text = modified_text[:match.start()] + placeholder + modified_text[match.end():]
            placeholder_count += 1
        
        # Additional patterns would follow the same approach
        
        return modified_text

    def _generate_disclaimer_alternatives(self, text, violation, context=None):
        """
        Generate alternative text that includes appropriate disclaimers.
        
        Args:
            text: Original text requiring disclaimers
            violation: The compliance violation
            context: Optional context information
            
        Returns:
            List of alternatives with appropriate disclaimers
        """
        alternatives = []
        violation_type = violation.get("type", "")
        violation_metadata = violation.get("metadata", {})
        severity = violation.get("severity", "medium")
        framework = violation_metadata.get("framework", "")
        
        # Determine disclaimer type based on violation
        disclaimer_types = []
        
        if "financial" in violation_type.lower():
            disclaimer_types.append("financial")
        if "health" in violation_type.lower() or "medical" in violation_type.lower():
            disclaimer_types.append("health")
        if "personal_data" in violation_type.lower() or "privacy" in violation_type.lower():
            disclaimer_types.append("privacy")
        if "security" in violation_type.lower():
            disclaimer_types.append("security")
        if "legal" in violation_type.lower() or "advice" in violation_type.lower():
            disclaimer_types.append("legal")
        
        # If we couldn't infer specific types, use a general disclaimer
        if not disclaimer_types:
            disclaimer_types.append("general")
        
        # Generate alternatives with different disclaimer placements
        
        # Option 1: Disclaimer at the beginning
        prefix_disclaimer = self._get_disclaimer_text(disclaimer_types, framework, placement="prefix")
        prefix_alternative = prefix_disclaimer + "\n\n" + text
        alternatives.append({
            "text": prefix_alternative,
            "strategy": "prefix_disclaimer",
            "confidence": 0.9,
            "description": "Added disclaimer at the beginning of the text"
        })
        
        # Option 2: Disclaimer at the end
        suffix_disclaimer = self._get_disclaimer_text(disclaimer_types, framework, placement="suffix")
        suffix_alternative = text + "\n\n" + suffix_disclaimer
        alternatives.append({
            "text": suffix_alternative,
            "strategy": "suffix_disclaimer",
            "confidence": 0.85,
            "description": "Added disclaimer at the end of the text"
        })
        
        # Option 3: Integrated disclaimer (for shorter content)
        if len(text.split()) < 100:  # Only for relatively short text
            integrated_disclaimer = self._get_disclaimer_text(disclaimer_types, framework, placement="integrated")
            
            # Try to add after the first paragraph
            paragraphs = text.split("\n\n")
            if len(paragraphs) > 1:
                paragraphs.insert(1, integrated_disclaimer)
                integrated_alternative = "\n\n".join(paragraphs)
            else:
                sentences = text.split(". ")
                if len(sentences) > 1:
                    # Insert after first sentence
                    sentences[0] = sentences[0] + ". " + integrated_disclaimer
                    integrated_alternative = ". ".join(sentences)
                else:
                    integrated_alternative = text + " " + integrated_disclaimer
            
            alternatives.append({
                "text": integrated_alternative,
                "strategy": "integrated_disclaimer",
                "confidence": 0.75,
                "description": "Integrated disclaimer within the text content"
            })
        
        # Option 4: Framed disclaimer (before and after) for high severity violations
        if severity == "high":
            framed_disclaimer_prefix = self._get_disclaimer_text(disclaimer_types, framework, placement="framed_prefix")
            framed_disclaimer_suffix = self._get_disclaimer_text(disclaimer_types, framework, placement="framed_suffix")
            framed_alternative = framed_disclaimer_prefix + "\n\n" + text + "\n\n" + framed_disclaimer_suffix
            
            alternatives.append({
                "text": framed_alternative,
                "strategy": "framed_disclaimer",
                "confidence": 0.95,
                "description": "Added disclaimers both before and after the content"
            })
        
        return alternatives


    def _get_disclaimer_text(self, disclaimer_types, framework=None, placement="suffix"):
        """Get appropriate disclaimer text based on types and placement"""
        # Base disclaimers by type
        disclaimer_templates = {
            "general": {
                "prefix": "DISCLAIMER: The following content is provided for informational purposes only.",
                "suffix": "This information is provided as-is without any warranties or guarantees of accuracy.",
                "integrated": "Please note that this information is provided for informational purposes only.",
                "framed_prefix": "⚠️ IMPORTANT DISCLAIMER: The following content is provided for informational purposes only and should not be considered as professional advice.",
                "framed_suffix": "Remember: This information is provided as-is without any warranties. Always consult with appropriate professionals before making decisions."
            },
            "financial": {
                "prefix": "FINANCIAL DISCLAIMER: The following content does not constitute financial advice.",
                "suffix": "This information should not be considered financial advice. Consult with a financial professional before making investment decisions.",
                "integrated": "Note that this information does not constitute financial advice.",
                "framed_prefix": "⚠️ FINANCIAL DISCLAIMER: The following content is for informational purposes only and does not constitute financial advice.",
                "framed_suffix": "IMPORTANT: Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions."
            },
            "health": {
                "prefix": "HEALTH DISCLAIMER: The following content is not medical advice.",
                "suffix": "This information is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider.",
                "integrated": "Please note that this information is not medical advice.",
                "framed_prefix": "⚠️ HEALTH DISCLAIMER: The following information is not medical advice and should not be used to diagnose or treat any health condition.",
                "framed_suffix": "IMPORTANT: Always consult with a qualified healthcare provider regarding any medical conditions or treatments."
            },
            "privacy": {
                "prefix": "PRIVACY NOTICE: The following content discusses personal information handling.",
                "suffix": "Always ensure appropriate data protection measures when handling personal information in accordance with applicable laws.",
                "integrated": "Note that handling of personal data should comply with applicable privacy regulations.",
                "framed_prefix": "⚠️ PRIVACY NOTICE: The following content discusses handling of personal information, which is subject to privacy regulations.",
                "framed_suffix": "IMPORTANT: Ensure all personal data processing complies with applicable privacy laws such as GDPR, CCPA, or other relevant regulations."
            },
            "security": {
                "prefix": "SECURITY NOTICE: The following content discusses security concepts.",
                "suffix": "Ensure proper security measures are implemented to protect systems and data.",
                "integrated": "Note that proper security measures should always be implemented.",
                "framed_prefix": "⚠️ SECURITY NOTICE: The following content discusses security concepts that should be implemented with care.",
                "framed_suffix": "IMPORTANT: Always follow security best practices and consult with security professionals when implementing security measures."
            },
            "legal": {
                "prefix": "LEGAL DISCLAIMER: The following content is not legal advice.",
                "suffix": "This information is not a substitute for professional legal advice. Consult with a qualified legal professional for advice on specific situations.",
                "integrated": "Please note that this information is not legal advice.",
                "framed_prefix": "⚠️ LEGAL DISCLAIMER: The following content is for informational purposes only and does not constitute legal advice.",
                "framed_suffix": "IMPORTANT: Laws and regulations vary by jurisdiction. Always consult with a qualified legal professional regarding your specific circumstances."
            }
        }
        
        # Framework-specific modifications
        framework_additions = {
            "GDPR": {
                "privacy": {
                    "suffix": "Ensure all personal data processing complies with GDPR principles, including lawfulness, fairness, and transparency."
                }
            },
            "HIPAA": {
                "health": {
                    "suffix": "Ensure all protected health information (PHI) is handled in accordance with HIPAA regulations."
                }
            },
            "FINREG": {
                "financial": {
                    "suffix": "Ensure compliance with applicable financial regulations and reporting requirements."
                }
            }
        }
        
        # Start with the most relevant disclaimer type
        primary_type = disclaimer_types[0] if disclaimer_types else "general"
        disclaimer_text = disclaimer_templates.get(primary_type, disclaimer_templates["general"]).get(placement, "")
        
        # Add framework-specific text if applicable
        if framework and framework in framework_additions:
            if primary_type in framework_additions[framework]:
                framework_text = framework_additions[framework][primary_type].get(placement)
                if framework_text and placement == "suffix":
                    disclaimer_text += " " + framework_text
        
        # For multiple disclaimer types, add additional context
        if len(disclaimer_types) > 1:
            additional_types = [t for t in disclaimer_types[1:] if t != primary_type]
            for add_type in additional_types:
                add_text = disclaimer_templates.get(add_type, {}).get("integrated", "")
                if add_text and "integrated" not in placement:
                    disclaimer_text += " " + add_text
        
        return disclaimer_text

    def _generate_disclaimer_alternatives(self, text, issues, context=None):
        """
        Generate alternative disclaimers for content based on compliance issues.
        
        Args:
            text: Original text content
            issues: Compliance issues detected
            context: Optional context information
                
        Returns:
            List of disclaimer alternatives with varying levels of detail
        """
        if not issues:
            return []
            
        alternatives = []
        
        # Group issues by category/type
        issue_categories = self._categorize_issues(issues)
        
        # Generate different disclaimer types (brief, standard, detailed)
        for detail_level in ["brief", "standard", "detailed"]:
            disclaimer = self._create_disclaimer(issue_categories, detail_level, context)
            if disclaimer:
                alternatives.append({
                    "text": disclaimer,
                    "type": "disclaimer",
                    "detail_level": detail_level,
                    "placement": "beginning",  # Default placement
                    "confidence": self._calculate_disclaimer_confidence(issues, detail_level)
                })
                
        # Generate framework-specific disclaimers if applicable
        framework_disclaimers = self._generate_framework_specific_disclaimers(issues, context)
        if framework_disclaimers:
            alternatives.extend(framework_disclaimers)
            
        # Generate placement variations (beginning, end, inline)
        placement_variations = self._generate_placement_variations(alternatives[0] if alternatives else None)
        if placement_variations:
            alternatives.extend(placement_variations)
            
        return alternatives
        
    def _categorize_issues(self, issues):
        """Categorize compliance issues for disclaimer generation"""
        categories = {
            "privacy": [],
            "financial": [],
            "health": [],
            "security": [],
            "legal": [],
            "ethical": [],
            "general": []
        }
        
        # Map issues to categories based on rule ID or metadata
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            description = issue.get("description", "")
            severity = issue.get("severity", "medium")
            
            # Determine category from rule ID prefixes or keywords in description
            category = "general"
            
            if any(p in rule_id.lower() for p in ["pii", "gdpr", "ccpa", "privacy", "data_"]):
                category = "privacy"
            elif any(p in rule_id.lower() for p in ["hipaa", "phi", "health", "medical"]):
                category = "health"
            elif any(p in rule_id.lower() for p in ["finreg", "financial", "payment"]):
                category = "financial"
            elif any(p in rule_id.lower() for p in ["security", "secure", "auth"]):
                category = "security"
            elif any(p in rule_id.lower() for p in ["legal", "compliance", "regulatory"]):
                category = "legal"
            elif any(p in rule_id.lower() for p in ["bias", "fairness", "ethical"]):
                category = "ethical"
                
            # Also check description keywords
            if category == "general":
                lower_desc = description.lower()
                if any(k in lower_desc for k in ["personal data", "privacy", "consent"]):
                    category = "privacy"
                elif any(k in lower_desc for k in ["health", "medical", "patient"]):
                    category = "health"
                elif any(k in lower_desc for k in ["financial", "payment", "transaction"]):
                    category = "financial"
                elif any(k in lower_desc for k in ["security", "secure", "protection"]):
                    category = "security"
                elif any(k in lower_desc for k in ["legal", "law", "regulation"]):
                    category = "legal"
                elif any(k in lower_desc for k in ["bias", "fairness", "ethical"]):
                    category = "ethical"
                    
            # Add to appropriate category with severity
            categories[category].append({
                "issue": issue,
                "severity": severity
            })
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
        
    def _create_disclaimer(self, issue_categories, detail_level, context=None):
        """Create a disclaimer based on issue categories and detail level"""
        if not issue_categories:
            return None
            
        # Get domain/industry from context if available
        domain = "general"
        if context and "domain" in context:
            domain = context["domain"]
            
        # Templates for different detail levels
        templates = {
            "brief": {
                "privacy": "This content contains information related to personal data. {additional_note}",
                "financial": "This content contains financial information. {additional_note}",
                "health": "This content contains health-related information. {additional_note}",
                "security": "This content contains security-related information. {additional_note}",
                "legal": "This content contains legal information. {additional_note}",
                "ethical": "This content may contain subjective viewpoints. {additional_note}",
                "general": "Disclaimer: This content may require regulatory compliance consideration. {additional_note}"
            },
            "standard": {
                "privacy": "PRIVACY NOTICE: This content contains references to personal data that may be subject to privacy regulations. Please ensure all data handling complies with applicable laws including {laws}. {additional_note}",
                "financial": "FINANCIAL NOTICE: This content contains financial information that may be subject to regulatory requirements. This is not financial advice. {additional_note}",
                "health": "HEALTH INFORMATION NOTICE: This content contains health-related information. This is not medical advice. Please consult qualified healthcare professionals for medical decisions. {additional_note}",
                "security": "SECURITY NOTICE: This content contains security-related information. Implementation of security measures should be verified by qualified professionals. {additional_note}",
                "legal": "LEGAL NOTICE: This content contains legal information. This is not legal advice. Please consult qualified legal professionals for legal decisions. {additional_note}",
                "ethical": "NOTICE: This content may contain subjective viewpoints and should be evaluated critically. Ensure fair and unbiased implementation in all contexts. {additional_note}",
                "general": "REGULATORY NOTICE: This content may be subject to regulatory requirements. Please verify compliance with all applicable regulations before use. {additional_note}"
            },
            "detailed": {
                "privacy": "IMPORTANT PRIVACY NOTICE: This content references personal data that is subject to privacy regulations including {laws}. Organizations processing such data must: (1) establish a legal basis for processing, (2) implement appropriate security measures, (3) honor data subject rights, and (4) maintain required documentation. Failure to comply may result in significant penalties. {specific_requirements}",
                "financial": "IMPORTANT FINANCIAL REGULATORY NOTICE: This content contains financial information subject to regulatory oversight. This is not financial advice and should not be relied upon for financial decisions. Organizations handling such information must adhere to applicable financial regulations including reporting requirements, disclosure obligations, and security standards. {specific_requirements}",
                "health": "IMPORTANT HEALTH INFORMATION NOTICE: This content contains health-related information protected under health privacy laws including {laws}. This is not medical advice. Any use, disclosure, or transmission of this information must comply with applicable health information privacy and security requirements. Specifically: {specific_requirements}",
                "security": "IMPORTANT SECURITY NOTICE: This content contains security-related information. Implementation of security measures should follow industry best practices and be verified by qualified professionals. Organizations should: (1) regularly assess security risks, (2) implement appropriate controls, (3) test security measures, and (4) maintain security documentation. {specific_requirements}",
                "legal": "IMPORTANT LEGAL NOTICE: This content contains legal information, not legal advice. The applicability of laws varies by jurisdiction and circumstance. Organizations should: (1) consult qualified legal professionals, (2) verify requirements in relevant jurisdictions, (3) maintain required documentation, and (4) implement appropriate compliance measures. {specific_requirements}",
                "ethical": "IMPORTANT ETHICAL CONSIDERATION NOTICE: This content may contain subjective viewpoints that should be evaluated critically to ensure fair and unbiased implementation. Organizations should: (1) review for potential bias, (2) consider diverse perspectives, (3) implement oversight mechanisms, and (4) regularly evaluate outcomes for fairness. {specific_requirements}",
                "general": "IMPORTANT REGULATORY NOTICE: This content may be subject to regulatory requirements. Organizations should: (1) identify applicable regulations, (2) implement required compliance measures, (3) maintain appropriate documentation, and (4) regularly review compliance status. Please verify compliance with all applicable regulations before use. {specific_requirements}"
            }
        }
        
        # Get the highest priority category (privacy, health, financial, etc.)
        category_priorities = {
            "health": 10, 
            "privacy": 9, 
            "financial": 8, 
            "security": 7, 
            "legal": 6, 
            "ethical": 5, 
            "general": 1
        }
        
        # Get applicable regulations based on categories
        regulations = set()
        specific_requirements = []
        high_severity_issues = False
        
        for category, issues in issue_categories.items():
            # Add category-specific regulations
            if category == "privacy":
                regulations.update(["GDPR", "CCPA", "CPRA"])
            elif category == "health":
                regulations.update(["HIPAA", "HITECH"])
            elif category == "financial":
                regulations.update(["GLBA", "PCI DSS"])
                
            # Check for high severity issues
            for issue_info in issues:
                if issue_info["severity"] in ["high", "critical"]:
                    high_severity_issues = True
                    
                # Add specific requirements based on issue
                issue = issue_info["issue"]
                if "description" in issue:
                    requirement = issue["description"].replace("Detected ", "").replace("Found ", "")
                    specific_requirements.append(requirement)
        
        # Limit specific requirements to top 3
        specific_requirements = specific_requirements[:3]
        
        # Select primary category based on priority
        primary_category = max(issue_categories.keys(), key=lambda k: category_priorities.get(k, 0))
        
        # Get template for primary category and detail level
        template = templates.get(detail_level, {}).get(primary_category, templates[detail_level]["general"])
        
        # Fill in template placeholders
        laws_text = ", ".join(regulations) if regulations else "applicable regulations"
        specific_requirements_text = ""
        
        if specific_requirements and detail_level == "detailed":
            specific_requirements_text = " Specific considerations include: " + "; ".join(specific_requirements) + "."
            
        additional_note = ""
        if high_severity_issues:
            additional_note = "Special attention required due to critical compliance considerations."
            
        disclaimer = template.format(
            laws=laws_text,
            specific_requirements=specific_requirements_text,
            additional_note=additional_note
        )
        
        return disclaimer
        
    def _generate_framework_specific_disclaimers(self, issues, context):
        """Generate framework-specific disclaimers for major regulatory frameworks"""
        framework_disclaimers = []
        
        # Detect frameworks from issues
        frameworks = set()
        for issue in issues:
            rule_id = issue.get("rule_id", "").lower()
            
            if "gdpr" in rule_id:
                frameworks.add("GDPR")
            elif "hipaa" in rule_id:
                frameworks.add("HIPAA")
            elif "ccpa" in rule_id or "cpra" in rule_id:
                frameworks.add("CCPA/CPRA")
            elif any(f in rule_id for f in ["glba", "pci", "finreg"]):
                frameworks.add("Financial")
                
        # Framework-specific templates
        for framework in frameworks:
            if framework == "GDPR":
                disclaimer = (
                    "GDPR COMPLIANCE NOTICE: This content involves personal data processing subject to the "
                    "General Data Protection Regulation. Processing requires a legal basis such as consent, "
                    "contract, legal obligation, vital interests, public task, or legitimate interests. "
                    "Data subjects have rights to access, rectification, erasure, restriction, portability, "
                    "and objection. Appropriate technical and organizational measures must be implemented to "
                    "ensure data security."
                )
                framework_disclaimers.append({
                    "text": disclaimer,
                    "type": "disclaimer",
                    "detail_level": "framework-specific",
                    "framework": "GDPR",
                    "placement": "beginning",
                    "confidence": 0.9
                })
                
            elif framework == "HIPAA":
                disclaimer = (
                    "HIPAA COMPLIANCE NOTICE: This content contains protected health information (PHI) "
                    "subject to the Health Insurance Portability and Accountability Act. Use and disclosure "
                    "of this information is restricted to permitted purposes. Appropriate safeguards must be "
                    "implemented to protect confidentiality, integrity, and availability of PHI. Unauthorized "
                    "use or disclosure may result in civil and criminal penalties."
                )
                framework_disclaimers.append({
                    "text": disclaimer,
                    "type": "disclaimer",
                    "detail_level": "framework-specific",
                    "framework": "HIPAA",
                    "placement": "beginning",
                    "confidence": 0.9
                })
                
            elif framework == "CCPA/CPRA":
                disclaimer = (
                    "CALIFORNIA PRIVACY NOTICE: This content involves personal information subject to the "
                    "California Consumer Privacy Act (CCPA) and California Privacy Rights Act (CPRA). "
                    "Consumers have rights to know, delete, correct, and limit use and disclosure of their "
                    "personal information. Businesses must provide privacy notices and honor consumer rights "
                    "requests. Sale or sharing of personal information requires opt-out mechanisms."
                )
                framework_disclaimers.append({
                    "text": disclaimer,
                    "type": "disclaimer",
                    "detail_level": "framework-specific",
                    "framework": "CCPA/CPRA",
                    "placement": "beginning",
                    "confidence": 0.9
                })
                
            elif framework == "Financial":
                disclaimer = (
                    "FINANCIAL REGULATORY NOTICE: This content involves financial information subject to "
                    "regulations such as Gramm-Leach-Bliley Act (GLBA) and Payment Card Industry Data "
                    "Security Standard (PCI DSS). Financial institutions must provide privacy notices, "
                    "implement information security programs, and protect nonpublic personal information. "
                    "This is not financial advice; consult qualified professionals for financial decisions."
                )
                framework_disclaimers.append({
                    "text": disclaimer,
                    "type": "disclaimer",
                    "detail_level": "framework-specific",
                    "framework": "Financial",
                    "placement": "beginning",
                    "confidence": 0.9
                })
        
        return framework_disclaimers
        
    def _generate_placement_variations(self, base_disclaimer):
        """Generate placement variations of a disclaimer"""
        if not base_disclaimer:
            return []
            
        variations = []
        
        # End placement
        end_variation = base_disclaimer.copy()
        end_variation["placement"] = "end"
        variations.append(end_variation)
        
        # Inline/boxed placement
        boxed_variation = base_disclaimer.copy()
        boxed_variation["placement"] = "boxed"
        boxed_variation["text"] = f"===== NOTICE =====\n{base_disclaimer['text']}\n==================="
        variations.append(boxed_variation)
        
        return variations
        
    def _calculate_disclaimer_confidence(self, issues, detail_level):
        """Calculate confidence score for a disclaimer"""
        # Base confidence by detail level
        detail_confidence = {
            "brief": 0.7,
            "standard": 0.8,
            "detailed": 0.9,
            "framework-specific": 0.95
        }
        
        # Adjust based on issue severity
        severity_count = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0
        }
        
        for issue in issues:
            severity = issue.get("severity", "medium")
            severity_count[severity] = severity_count.get(severity, 0) + 1
        
        # Calculate severity adjustment
        severity_factor = (
            severity_count["critical"] * 0.1 + 
            severity_count["high"] * 0.05 + 
            severity_count["medium"] * 0.02
        )
        
        # Cap at reasonable range
        final_confidence = min(0.99, detail_confidence.get(detail_level, 0.8) + severity_factor)
        
        return final_confidence


    def _generate_bias_correction_alternatives(self, text, issues, context=None):
        """
        Generate alternative text suggestions to correct potential bias.
        
        Args:
            text: Original text content
            issues: Compliance issues detected
            context: Optional context information
                
        Returns:
            List of bias correction alternatives
        """
        if not issues or not text:
            return []
            
        # Find bias-related issues
        bias_issues = [issue for issue in issues if self._is_bias_related(issue)]
        
        if not bias_issues:
            return []
            
        alternatives = []
        
        # Extract potentially biased segments
        biased_segments = self._extract_biased_segments(text, bias_issues)
        
        # Generate alternatives for each biased segment
        for segment in biased_segments:
            segment_alternatives = self._generate_segment_alternatives(segment, context)
            
            for alt in segment_alternatives:
                # Create modified text with this alternative
                modified_text = text.replace(segment["text"], alt["replacement"])
                
                alternatives.append({
                    "text": modified_text,
                    "type": "bias_correction",
                    "original_segment": segment["text"],
                    "replacement": alt["replacement"],
                    "bias_type": alt["bias_type"],
                    "explanation": alt["explanation"],
                    "confidence": alt["confidence"],
                    "segment_position": segment.get("position")
                })
        
        # If alternatives were generated, add a "minimal changes" option
        if alternatives and len(alternatives) > 1:
            # Find alternative with highest confidence
            best_alt = max(alternatives, key=lambda x: x["confidence"])
            
            # Create minimal changes version that only addresses the most confident fix
            alternatives.append({
                "text": best_alt["text"],
                "type": "bias_correction_minimal",
                "original_segment": best_alt["original_segment"],
                "replacement": best_alt["replacement"],
                "bias_type": best_alt["bias_type"],
                "explanation": f"Minimal change addressing {best_alt['bias_type']} bias",
                "confidence": best_alt["confidence"] - 0.05  # Slightly lower confidence for minimal approach
            })
        
        return alternatives
        
    def _is_bias_related(self, issue):
        """Check if an issue is related to bias"""
        # Check rule ID for bias-related keywords
        rule_id = issue.get("rule_id", "").lower()
        if any(term in rule_id for term in ["bias", "fairness", "inclusive", "gender", "stereotype", "discriminat"]):
            return True
            
        # Check description for bias-related keywords
        description = issue.get("description", "").lower()
        bias_terms = [
            "bias", "gender", "racial", "stereotype", "discriminatory", "inclusive", "fairness", 
            "prejudice", "ageism", "sexism", "racism", "ethnic", "neutral", "offensive"
        ]
        
        if any(term in description for term in bias_terms):
            return True
            
        return False
        
    def _extract_biased_segments(self, text, bias_issues):
        """Extract potentially biased segments from text based on issues"""
        segments = []
        
        for issue in bias_issues:
            # If issue contains location information, use it
            metadata = issue.get("metadata", {})
            location = issue.get("location")
            
            if location and "start" in location and "end" in location:
                start = location["start"]
                end = location["end"]
                if 0 <= start < end and end <= len(text):
                    segments.append({
                        "text": text[start:end],
                        "position": {"start": start, "end": end},
                        "issue": issue
                    })
                    continue
                    
            # If no location, try to find using rule-specific patterns
            rule_id = issue.get("rule_id", "")
            description = issue.get("description", "")
            
            # Extract potential problematic terms from description
            terms = self._extract_terms_from_description(description)
            
            for term in terms:
                if term in text:
                    # Find all occurrences
                    for match in re.finditer(re.escape(term), text):
                        start = match.start()
                        end = match.end()
                        
                        # Get context around the term (up to 100 chars)
                        context_start = max(0, start - 50)
                        context_end = min(len(text), end + 50)
                        context_text = text[context_start:context_end]
                        
                        segments.append({
                            "text": term,
                            "context": context_text,
                            "position": {"start": start, "end": end},
                            "issue": issue
                        })
        
        # Merge overlapping segments
        return self._merge_overlapping_segments(segments)
        
    def _extract_terms_from_description(self, description):
        """Extract potentially problematic terms from issue description"""
        terms = []
        
        # Check for quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', description)
        if quoted_terms:
            terms.extend(quoted_terms)
            
        # Check for specific patterns like "term X is biased"
        pattern_matches = re.findall(r'([\w\s-]+) (is|may be|could be) (biased|discriminatory|stereotyping)', description)
        if pattern_matches:
            terms.extend([match[0].strip() for match in pattern_matches])
            
        return terms
        
    def _merge_overlapping_segments(self, segments, text):
        """Merge overlapping text segments"""
        if not segments:
            return []
            
        # Sort by start position
        sorted_segments = sorted(segments, key=lambda s: s.get("position", {}).get("start", 0))
        
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            previous = merged[-1]
            
            # Check if segments overlap
            prev_end = previous.get("position", {}).get("end", 0)
            curr_start = current.get("position", {}).get("start", 0)
            
            if curr_start <= prev_end:
                # Merge segments
                prev_start = previous.get("position", {}).get("start", 0)
                curr_end = current.get("position", {}).get("end", 0)
                
                # Use the larger segment
                if curr_end > prev_end:
                    merged[-1]["position"] = {"start": prev_start, "end": curr_end}
                    merged[-1]["text"] = text[prev_start:curr_end]
                    
                    # Merge issues
                    if "issue" in current and "issue" in previous:
                        merged[-1]["issues"] = [previous["issue"], current["issue"]]
                    else:
                        merged[-1]["issues"] = [previous.get("issue"), current.get("issue")]
            else:
                # No overlap, add as new segment
                merged.append(current)
        
        return merged
        
    def _generate_segment_alternatives(self, segment, context):
        """Generate alternative text for a potentially biased segment"""
        original_text = segment["text"]
        bias_issue = segment.get("issue")
        
        alternatives = []
        
        # Identify bias type from issue
        bias_type = self._identify_bias_type(bias_issue)
        
        # Use appropriate correction strategy based on bias type
        if bias_type == "gender_bias":
            alternatives.extend(self._correct_gender_bias(original_text, bias_issue))
        elif bias_type == "racial_bias":
            alternatives.extend(self._correct_racial_bias(original_text, bias_issue))
        elif bias_type == "age_bias":
            alternatives.extend(self._correct_age_bias(original_text, bias_issue))
        elif bias_type == "disability_bias":
            alternatives.extend(self._correct_disability_bias(original_text, bias_issue))
        elif bias_type == "cultural_bias":
            alternatives.extend(self._correct_cultural_bias(original_text, bias_issue))
        else:
            # Generic bias correction for other types
            alternatives.extend(self._correct_generic_bias(original_text, bias_issue))
        
        return alternatives
        
    def _identify_bias_type(self, issue):
        """Identify the type of bias from an issue"""
        if not issue:
            return "general_bias"
            
        rule_id = issue.get("rule_id", "").lower()
        description = issue.get("description", "").lower()
        
        # Check for specific bias types
        if any(term in rule_id or term in description for term in ["gender", "sexist", "sexism", "man", "woman", "male", "female"]):
            return "gender_bias"
        elif any(term in rule_id or term in description for term in ["race", "racial", "ethnic", "racism"]):
            return "racial_bias"
        elif any(term in rule_id or term in description for term in ["age", "ageism", "elderly", "young", "old"]):
            return "age_bias"
        elif any(term in rule_id or term in description for term in ["disab", "handicap", "impair"]):
            return "disability_bias"
        elif any(term in rule_id or term in description for term in ["cultur", "religious", "nationality"]):
            return "cultural_bias"
        else:
            return "general_bias"
            
    def _correct_gender_bias(self, text, issue):
        """Generate alternatives to correct gender bias"""
        alternatives = []
        text_lower = text.lower()
        
        # Common gender-biased terms and alternatives
        gender_term_replacements = {
            "mankind": ["humanity", "humankind", "people"],
            "manpower": ["workforce", "staff", "personnel", "human resources"],
            "chairman": ["chairperson", "chair", "head"],
            "businessman": ["businessperson", "professional", "executive"],
            "policeman": ["police officer", "officer"],
            "stewardess": ["flight attendant"],
            "fireman": ["firefighter"],
            "mailman": ["mail carrier", "postal worker"],
            "salesman": ["salesperson", "sales representative"],
            "actress": ["actor"],
            "waitress": ["server", "wait staff"],
            "hostess": ["host"],
            "he or she": ["they"],
            "he/she": ["they"],
            "him or her": ["them"],
            "him/her": ["them"],
            "his or her": ["their"],
            "his/her": ["their"],
            "man": ["person", "individual"],
            "men": ["people", "individuals"],
            "manmade": ["artificial", "manufactured", "synthetic"],
            "guys": ["folks", "everyone", "people", "team"]
        }
        
        # Check for gender-specific terms
        for term, replacements in gender_term_replacements.items():
            if term in text_lower:
                for replacement in replacements:
                    # Attempt to preserve original capitalization
                    if text.isupper():
                        final_replacement = replacement.upper()
                    elif text[0].isupper():
                        final_replacement = replacement.capitalize()
                    else:
                        final_replacement = replacement
                    
                    alternatives.append({
                        "replacement": final_replacement,
                        "bias_type": "gender_bias",
                        "explanation": f"'{term}' may contain gender bias. '{replacement}' is gender-neutral.",
                        "confidence": 0.85
                    })
        
        # Check for gendered pronouns
        pronoun_patterns = [
            (r'\b(he|him|his)\b', "they/them/their", "The gendered pronoun may be replaced with a gender-neutral alternative."),
            (r'\b(she|her|hers)\b', "they/them/their", "The gendered pronoun may be replaced with a gender-neutral alternative.")
        ]
        
        for pattern, replacement, explanation in pronoun_patterns:
            if re.search(pattern, text_lower):
                # Replace all instances in the text
                replaced_text = re.sub(pattern, lambda m: "they" if m.group(0) in ["he", "she"] 
                                                else "them" if m.group(0) in ["him", "her"] 
                                                else "their", text, flags=re.IGNORECASE)
                
                alternatives.append({
                    "replacement": replaced_text,
                    "bias_type": "gender_bias",
                    "explanation": explanation,
                    "confidence": 0.8
                })
        
        # If no specific replacements were found but this is a gender bias issue
        if not alternatives:
            alternatives.append({
                "replacement": text,  # No automatic replacement available
                "bias_type": "gender_bias",
                "explanation": "This text may contain subtle gender bias that requires manual review.",
                "confidence": 0.5
            })
        
        return alternatives
        
    def _correct_racial_bias(self, text, issue):
        """Generate alternatives to correct racial bias"""
        alternatives = []
        text_lower = text.lower()
        
        # Check for problematic terms and phrases
        racial_term_replacements = {
            "colored people": ["people of color", "people", "individuals"],
            "oriental": ["Asian", "East Asian"],
            "indian": ["Native American", "Indigenous person"],
            "illegal alien": ["undocumented immigrant", "unauthorized immigrant"],
            "illegal immigrant": ["undocumented immigrant", "unauthorized immigrant"],
            "tribe": ["community", "group", "people"],
            "primitive": ["traditional", "indigenous", "early"],
            "third world": ["developing regions", "low-income countries"],
            "ghetto": ["underserved neighborhood", "economically disadvantaged area"],
            "ethnic food": ["[specific cuisine] food", "regional cuisine"],
            "urban": ["metropolitan", "city", "densely populated"]
        }
        
        # Check for potentially biased terms
        for term, replacements in racial_term_replacements.items():
            if term in text_lower:
                for replacement in replacements:
                    if replacement.startswith("[specific"):
                        # This requires manual intervention
                        alternatives.append({
                            "replacement": text,  # No automatic replacement
                            "bias_type": "racial_bias",
                            "explanation": f"'{term}' may contain racial bias. Consider specifying the actual cuisine instead of using generic terms.",
                            "confidence": 0.7
                        })
                    else:
                        alternatives.append({
                            "replacement": text.replace(term, replacement),
                            "bias_type": "racial_bias",
                            "explanation": f"'{term}' may contain racial bias. '{replacement}' is a more neutral alternative.",
                            "confidence": 0.85
                        })
        
        # Check for stereotypical characterizations
        stereotype_patterns = [
            (r'all (asian|african|hispanic|latino|black|white) people', "Some {0} people", "Avoid generalizations about racial or ethnic groups"),
            (r'(asians|africans|hispanics|latinos|blacks|whites) are', "{0} may have diverse characteristics", "Avoid generalizations about racial or ethnic groups")
        ]
        
        for pattern, replacement_template, explanation in stereotype_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                group = match.group(1)
                replacement = replacement_template.format(group.capitalize())
                alternatives.append({
                    "replacement": text.replace(match.group(0), replacement),
                    "bias_type": "racial_bias",
                    "explanation": explanation,
                    "confidence": 0.9
                })
        
        # If no specific replacements were found but this is a racial bias issue
        if not alternatives:
            alternatives.append({
                "replacement": text,  # No automatic replacement available
                "bias_type": "racial_bias",
                "explanation": "This text may contain subtle racial bias that requires manual review.",
                "confidence": 0.5
            })
        
        return alternatives
        
    def _correct_age_bias(self, text, issue):
        """Generate alternatives to correct age bias"""
        alternatives = []
        text_lower = text.lower()
        
        # Common age-biased terms and alternatives
        age_term_replacements = {
            "senior citizen": ["older adult", "older person"],
            "elderly": ["older adult", "older person"],
            "young person": ["person", "individual"],
            "old person": ["older adult", "older individual"],
            "geriatric": ["older", "senior"],
            "senile": ["having cognitive impairment", "experiencing memory issues"],
            "millennial": ["young adult", "person in their 20s/30s"],
            "boomer": ["older adult", "person born between 1946-1964"],
            "gen z": ["young person", "person born after 1996"],
            "old-fashioned": ["traditional", "classic"],
            "too old": ["experienced", "seasoned"],
            "too young": ["early in their career", "developing"]
        }
        
        # Check for potentially biased terms
        for term, replacements in age_term_replacements.items():
            if term in text_lower:
                for replacement in replacements:
                    alternatives.append({
                        "replacement": text.replace(term, replacement),
                        "bias_type": "age_bias",
                        "explanation": f"'{term}' may contain age bias. '{replacement}' is a more neutral alternative.",
                        "confidence": 0.8
                    })
        
        # Check for age-based stereotypes
        stereotype_patterns = [
            (r'(old|older) people (can\'t|cannot|don\'t|do not|are unable to)', "Some older individuals may have difficulty with", "Avoid generalizations about capabilities based on age"),
            (r'(young|younger) people (are all|all|always)', "Some younger individuals may", "Avoid generalizations about behavior based on age")
        ]
        
        for pattern, replacement_prefix, explanation in stereotype_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                action = text[match.end():].strip().split(" ")[0]  # Get the action being stereotyped
                replacement = f"{replacement_prefix} {action}"
                alternatives.append({
                    "replacement": text.replace(match.group(0), replacement),
                    "bias_type": "age_bias",
                    "explanation": explanation,
                    "confidence": 0.85
                })
        
        # If no specific replacements were found but this is an age bias issue
        if not alternatives:
            alternatives.append({
                "replacement": text,  # No automatic replacement available
                "bias_type": "age_bias",
                "explanation": "This text may contain subtle age bias that requires manual review.",
                "confidence": 0.5
            })
        
        return alternatives
        
    def _correct_disability_bias(self, text, issue):
        """Generate alternatives to correct disability bias"""
        alternatives = []
        text_lower = text.lower()
        
        # Common disability-biased terms and alternatives
        disability_term_replacements = {
            "handicapped": ["person with a disability", "disabled person"],
            "disabled": ["person with a disability", "person with disabilities"],
            "crippled": ["person with mobility impairment", "person with physical disability"],
            "wheelchair-bound": ["wheelchair user", "person who uses a wheelchair"],
            "retarded": ["person with cognitive disability", "person with intellectual disability"],
            "mentally ill": ["person with mental health condition", "person with psychiatric disability"],
            "crazy": ["person with mental health condition"],
            "insane": ["person with mental health condition"],
            "deaf and dumb": ["deaf", "person who is deaf"],
            "blind": ["person who is blind", "person with visual impairment"],
            "suffers from": ["has", "lives with"],
            "confined to a wheelchair": ["wheelchair user", "person who uses a wheelchair"],
            "special needs": ["specific needs", "support needs", "disability-related needs"]
        }
        
        # Check for potentially biased terms
        for term, replacements in disability_term_replacements.items():
            if term in text_lower:
                for replacement in replacements:
                    alternatives.append({
                        "replacement": text.replace(term, replacement),
                        "bias_type": "disability_bias",
                        "explanation": f"'{term}' may contain disability bias. '{replacement}' uses person-first or identity-first language that respects dignity.",
                        "confidence": 0.9
                    })
        
        # Check for inspiration-based objectification ("inspiration porn")
        inspiration_patterns = [
            (r'(inspire|inspiring|inspiration|brave|courage|courageous|despite) (disability|handicap|wheelchair|condition)', 
            "living with a disability", 
            "Avoid portraying people with disabilities as inspirational merely for living with a disability")
        ]
        
        for pattern, replacement, explanation in inspiration_patterns:
            if re.search(pattern, text_lower):
                alternatives.append({
                    "replacement": text,  # This requires careful rewording
                    "bias_type": "disability_bias",
                    "explanation": explanation,
                    "confidence": 0.7
                })
        
        # If no specific replacements were found but this is a disability bias issue
        if not alternatives:
            alternatives.append({
                "replacement": text,  # No automatic replacement available
                "bias_type": "disability_bias",
                "explanation": "This text may contain subtle disability bias that requires manual review.",
                "confidence": 0.5
            })
        
        return alternatives
        
    def _correct_cultural_bias(self, text, issue):
        """Generate alternatives to correct cultural or religious bias"""
        alternatives = []
        text_lower = text.lower()
        
        # Common cultural bias terms and alternatives
        cultural_term_replacements = {
            "third world country": ["developing nation", "low-income country"],
            "illegal alien": ["undocumented immigrant", "unauthorized immigrant"],
            "exotic": ["unique", "distinctive", "different"],
            "backwards culture": ["different cultural practices", "cultural tradition"],
            "primitive culture": ["indigenous culture", "traditional society"],
            "savage": ["indigenous person", "traditional"],
            "uncivilized": ["non-Western", "traditional"],
            "shaman": ["spiritual leader", "traditional healer", "indigenous spiritual practitioner"],
            "normal food": ["typical Western food", "common American/European food"],
            "normal dress": ["typical Western dress", "common American/European clothing"]
        }
        
        # Check for potentially biased terms
        for term, replacements in cultural_term_replacements.items():
            if term in text_lower:
                for replacement in replacements:
                    alternatives.append({
                        "replacement": text.replace(term, replacement),
                        "bias_type": "cultural_bias",
                        "explanation": f"'{term}' may contain cultural bias. '{replacement}' is a more culturally sensitive alternative.",
                        "confidence": 0.85
                    })
        
        # Check for religious generalizations
        religion_patterns = [
            (r'all (muslims|christians|jews|hindus|buddhists|atheists) (are|believe)', 
            "Many {0} may", 
            "Avoid generalizations about religious groups; respect diversity within traditions"),
            (r'(islam|christianity|judaism|hinduism|buddhism|atheism) teaches that', 
            "Some interpretations of {0} suggest that", 
            "Acknowledge diversity of interpretation within religious traditions")
        ]
        
        for pattern, replacement_template, explanation in religion_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                group = match.group(1)
                replacement = replacement_template.format(group.capitalize())
                alternatives.append({
                    "replacement": text.replace(match.group(0), replacement),
                    "bias_type": "cultural_bias",
                    "explanation": explanation,
                    "confidence": 0.85
                })
        
        # If no specific replacements were found but this is a cultural bias issue
        if not alternatives:
            alternatives.append({
                "replacement": text,  # No automatic replacement available
                "bias_type": "cultural_bias",
                "explanation": "This text may contain subtle cultural bias that requires manual review.",
                "confidence": 0.5
            })
        
        return alternatives
        
    def _correct_generic_bias(self, text, issue):
        """Generate alternatives for general bias issues"""
        alternatives = []
        
        # Extract potentially problematic phrases from issue description
        description = issue.get("description", "")
        terms = self._extract_terms_from_description(description)
        
        if terms:
            for term in terms:
                if term in text:
                    # Suggest removing or revising the term
                    alternatives.append({
                        "replacement": text.replace(term, "[revised text]"),
                        "bias_type": "general_bias",
                        "explanation": f"The phrase '{term}' may contain bias and should be reviewed or replaced.",
                        "confidence": 0.6
                    })
        
        # If no specific terms identified, suggest general review
        if not alternatives:
            alternatives.append({
                "replacement": text,  # No automatic replacement
                "bias_type": "general_bias",
                "explanation": "This text may contain subtle bias that requires manual review.",
                "confidence": 0.5
            })
        
        return alternatives
    def _generate_bias_correction_alternatives(self, text, violation, context=None):
        """
        Generate alternative text with reduced bias.
        
        Args:
            text: Original text containing potential bias
            violation: The compliance violation
            context: Optional context information
            
        Returns:
            List of alternative text options with reduced bias
        """
        alternatives = []
        violation_metadata = violation.get("metadata", {})
        bias_type = violation_metadata.get("bias_type", "")
        severity = violation.get("severity", "medium")
        affected_segments = violation_metadata.get("affected_segments", [])
        
        # Identify bias type if not explicitly provided
        if not bias_type:
            bias_type = self._detect_bias_type(text, affected_segments)
        
        # Gender bias correction
        if bias_type == "gender" or not bias_type:
            gender_neutral = self._apply_gender_neutral_language(text, affected_segments)
            if gender_neutral != text:
                alternatives.append({
                    "text": gender_neutral,
                    "strategy": "gender_neutral_language",
                    "confidence": 0.85,
                    "description": "Applied gender-neutral language to reduce gender bias"
                })
        
        # Age bias correction
        if bias_type == "age" or not bias_type:
            age_neutral = self._apply_age_neutral_language(text, affected_segments)
            if age_neutral != text:
                alternatives.append({
                    "text": age_neutral,
                    "strategy": "age_neutral_language",
                    "confidence": 0.8,
                    "description": "Applied age-neutral language to reduce age bias"
                })
        
        # Cultural/ethnic bias correction
        if bias_type == "cultural" or bias_type == "ethnic" or not bias_type:
            culturally_neutral = self._apply_culturally_neutral_language(text, affected_segments)
            if culturally_neutral != text:
                alternatives.append({
                    "text": culturally_neutral,
                    "strategy": "culturally_neutral_language",
                    "confidence": 0.85,
                    "description": "Applied culturally neutral language to reduce cultural bias"
                })
        
        # Socioeconomic bias correction
        if bias_type == "socioeconomic" or not bias_type:
            socioeconomic_neutral = self._apply_socioeconomic_neutral_language(text, affected_segments)
            if socioeconomic_neutral != text:
                alternatives.append({
                    "text": socioeconomic_neutral,
                    "strategy": "socioeconomic_neutral_language",
                    "confidence": 0.8,
                    "description": "Applied socioeconomic neutral language to reduce class bias"
                })
        
        # Ability/disability bias correction
        if bias_type == "ability" or not bias_type:
            ability_neutral = self._apply_ability_neutral_language(text, affected_segments)
            if ability_neutral != text:
                alternatives.append({
                    "text": ability_neutral,
                    "strategy": "ability_neutral_language",
                    "confidence": 0.85,
                    "description": "Applied ability-neutral language to reduce ableism"
                })
        
        # For high severity cases, apply comprehensive rewriting
        if severity == "high" and affected_segments:
            comprehensive_rewrite = self._comprehensive_bias_reduction(text, affected_segments, bias_type)
            if comprehensive_rewrite != text:
                alternatives.append({
                    "text": comprehensive_rewrite,
                    "strategy": "comprehensive_rewrite",
                    "confidence": 0.9,
                    "description": "Comprehensively rewrote content to address multiple forms of bias"
                })
        
        return alternatives


    def _detect_bias_type(self, text, affected_segments=None):
        """Detect the type of bias present in text"""
        text_lower = text.lower()
        
        # If we have affected segments, focus analysis there
        if affected_segments:
            segment_texts = []
            for segment in affected_segments:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                if start < end and end <= len(text):
                    segment_texts.append(text[start:end].lower())
            
            # Analyze the segments if available
            if segment_texts:
                text_to_analyze = " ".join(segment_texts)
            else:
                text_to_analyze = text_lower
        else:
            text_to_analyze = text_lower
        
        # Simple rule-based bias detection
        # Gender bias indicators
        gender_terms = ['he', 'she', 'his', 'her', 'him', 'hers', 'himself', 'herself', 
                    'man', 'woman', 'men', 'women', 'male', 'female',
                    'chairman', 'chairwoman', 'policeman', 'policewoman',
                    'businessman', 'businesswoman', 'salesman', 'saleswoman',
                    'steward', 'stewardess', 'waiter', 'waitress']
        
        gender_bias_count = sum(text_to_analyze.count(' ' + term + ' ') for term in gender_terms)
        
        # Age bias indicators
        age_terms = ['old', 'young', 'elderly', 'senior', 'junior', 'millennial', 'boomer',
                    'generation', 'retirement', 'retiree', 'aged', 'youthful', 'geriatric']
        
        age_bias_count = sum(text_to_analyze.count(' ' + term + ' ') for term in age_terms)
        
        # Cultural/ethnic bias indicators - simplified approach
        cultural_terms = ['foreign', 'immigrant', 'ethnic', 'culture', 'cultural', 
                        'western', 'eastern', 'oriental', 'exotic', 'traditional',
                        'developing', 'developed', 'third world', 'first world']
        
        cultural_bias_count = sum(text_to_analyze.count(' ' + term + ' ') for term in cultural_terms)
        
        # Socioeconomic bias indicators
        socioeconomic_terms = ['poor', 'rich', 'wealthy', 'low-income', 'high-income',
                            'privileged', 'underprivileged', 'disadvantaged', 'affluent',
                            'poverty', 'wealth', 'class', 'elite', 'ghetto', 'blue-collar',
                            'white-collar', 'welfare']
        
        socioeconomic_bias_count = sum(text_to_analyze.count(' ' + term + ' ') for term in socioeconomic_terms)
        
        # Ability/disability bias indicators
        ability_terms = ['disabled', 'handicapped', 'special needs', 'differently abled',
                        'blind', 'deaf', 'dumb', 'crippled', 'wheelchair', 'mental illness',
                        'crazy', 'insane', 'retarded', 'normal', 'abnormal', 'lame']
        
        ability_bias_count = sum(text_to_analyze.count(' ' + term + ' ') for term in ability_terms)
        
        # Determine the most prevalent bias type
        bias_counts = {
            "gender": gender_bias_count,
            "age": age_bias_count,
            "cultural": cultural_bias_count,
            "socioeconomic": socioeconomic_bias_count,
            "ability": ability_bias_count
        }
        
        # Return the bias type with the highest count, or None if all counts are 0
        max_bias_type = max(bias_counts.items(), key=lambda x: x[1])
        if max_bias_type[1] > 0:
            return max_bias_type[0]
        else:
            return None
