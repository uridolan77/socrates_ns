from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import datetime
import json

class ConflictLevel(Enum):
    """Levels of regulatory conflicts"""
    MINOR = "minor"           # Small inconsistencies, easily reconcilable
    MODERATE = "moderate"     # Significant differences requiring attention
    MAJOR = "major"           # Fundamental conflicts requiring careful resolution
    CRITICAL = "critical"     # Irreconcilable conflicts requiring legal guidance


@dataclass
class ExplainedResolution:
    """Detailed explanation of a conflict resolution"""
    winning_framework: str
    losing_frameworks: List[str]
    resolution_method: str
    conflict_level: ConflictLevel
    justification: str
    legal_references: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    recommendation: str = ""
    human_review_needed: bool = False


class EnhancedConflictResolver:
    """
    Enhanced resolver for regulatory conflicts with detailed explanations
    and natural language justifications for decisions.
    """
    
    def __init__(self, 
                framework_data: Dict[str, Dict],
                precedence_rules: Dict[str, Dict] = None,
                explainer: Any = None):
        """
        Initialize the enhanced conflict resolver
        
        Args:
            framework_data: Data about regulatory frameworks
            precedence_rules: Rules for precedence between frameworks
            explainer: Component for generating natural language explanations
        """
        self.framework_data = framework_data
        self.precedence_rules = precedence_rules or {}
        self.explainer = explainer
        
    def resolve_conflict(self, 
                        frameworks: List[str], 
                        context: Dict[str, Any]) -> ExplainedResolution:
        """
        Resolve conflicts between regulatory frameworks with detailed explanation
        
        Args:
            frameworks: List of conflicting frameworks
            context: Context information for resolution
            
        Returns:
            Detailed explanation of the resolution
        """
        if len(frameworks) < 2:
            return ExplainedResolution(
                winning_framework=frameworks[0] if frameworks else "none",
                losing_frameworks=[],
                resolution_method="no_conflict",
                conflict_level=ConflictLevel.MINOR,
                justification="No conflict to resolve with single framework",
                human_review_needed=False
            )
            
        # Find the level of conflict between frameworks
        conflict_level = self._determine_conflict_level(frameworks, context)
        
        # For critical conflicts, recommend human review
        if conflict_level == ConflictLevel.CRITICAL:
            return self._generate_critical_conflict_resolution(frameworks, context)
            
        # For other conflicts, determine winning framework
        winning_framework, method, justification = self._determine_winner(frameworks, context)
        losing_frameworks = [f for f in frameworks if f != winning_framework]
        
        # Generate legal references
        legal_references = self._gather_legal_references(winning_framework, losing_frameworks, method)
        
        # Identify edge cases where this resolution might not apply
        edge_cases = self._identify_edge_cases(winning_framework, losing_frameworks, context)
        
        # Generate a recommendation
        recommendation = self._generate_recommendation(
            winning_framework, losing_frameworks, method, conflict_level
        )
        
        # Determine if human review is needed
        human_review_needed = conflict_level in [ConflictLevel.MAJOR, ConflictLevel.CRITICAL]
        
        return ExplainedResolution(
            winning_framework=winning_framework,
            losing_frameworks=losing_frameworks,
            resolution_method=method,
            conflict_level=conflict_level,
            justification=justification,
            legal_references=legal_references,
            edge_cases=edge_cases,
            recommendation=recommendation,
            human_review_needed=human_review_needed
        )
    
    def _determine_conflict_level(self, 
                                frameworks: List[str], 
                                context: Dict[str, Any]) -> ConflictLevel:
        """Determine the level of conflict between frameworks"""
        # Count conflicting requirements
        conflicting_requirements = self._count_conflicting_requirements(frameworks)
        
        # Check for fundamentally opposing principles
        has_opposing_principles = self._has_opposing_principles(frameworks)
        
        # Check for jurisdictional conflicts
        has_jurisdictional_conflict = self._has_jurisdictional_conflict(frameworks, context)
        
        # Determine level based on conflicts found
        if has_opposing_principles or conflicting_requirements > 5:
            return ConflictLevel.CRITICAL
        elif has_jurisdictional_conflict or conflicting_requirements > 3:
            return ConflictLevel.MAJOR
        elif conflicting_requirements > 1:
            return ConflictLevel.MODERATE
        else:
            return ConflictLevel.MINOR
    
    def _count_conflicting_requirements(self, frameworks: List[str]) -> int:
        """Count requirements that conflict between frameworks"""
        # This would be a more sophisticated implementation in practice
        # For now, just return a sample count
        if len(frameworks) > 2:
            return 3
        else:
            return 1
    
    def _has_opposing_principles(self, frameworks: List[str]) -> bool:
        """Check if frameworks have fundamentally opposing principles"""
        # This would check for deep incompatibilities between frameworks
        # For example, one requiring disclosure and another prohibiting it
        return False  # Placeholder implementation
    
    def _has_jurisdictional_conflict(self, 
                                   frameworks: List[str],
                                   context: Dict[str, Any]) -> bool:
        """Check if there's a jurisdictional conflict between frameworks"""
        jurisdictions = set()
        
        for framework in frameworks:
            if framework in self.framework_data:
                framework_jurisdictions = self.framework_data[framework].get("jurisdictions", [])
                jurisdictions.update(framework_jurisdictions)
                
        # If there's more than one jurisdiction, there might be a conflict
        return len(jurisdictions) > 1
    
    def _determine_winner(self, 
                        frameworks: List[str], 
                        context: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Determine the winning framework and explain why
        
        Returns:
            Tuple of (winning_framework, method, justification)
        """
        # Check for explicit precedence rules
        for i, framework_a in enumerate(frameworks):
            for framework_b in frameworks[i+1:]:
                if framework_a in self.precedence_rules and framework_b in self.precedence_rules[framework_a]:
                    rule = self.precedence_rules[framework_a][framework_b]
                    return framework_a, "explicit_precedence", f"{framework_a} takes precedence over {framework_b} based on explicit precedence rule: {rule}"
                elif framework_b in self.precedence_rules and framework_a in self.precedence_rules[framework_b]:
                    rule = self.precedence_rules[framework_b][framework_a]
                    return framework_b, "explicit_precedence", f"{framework_b} takes precedence over {framework_a} based on explicit precedence rule: {rule}"
                    
        # Check for jurisdictional precedence
        if "jurisdiction" in context:
            current_jurisdiction = context["jurisdiction"]
            for framework in frameworks:
                if framework in self.framework_data:
                    framework_jurisdictions = self.framework_data[framework].get("jurisdictions", [])
                    if current_jurisdiction in framework_jurisdictions:
                        return framework, "jurisdictional", f"{framework} applies in the current jurisdiction ({current_jurisdiction})"
        
        # Check for domain-specific frameworks that match the context
        if "domain" in context:
            current_domain = context["domain"]
            for framework in frameworks:
                if framework in self.framework_data:
                    framework_domains = self.framework_data[framework].get("domains", [])
                    if current_domain in framework_domains:
                        return framework, "domain_specific", f"{framework} is specifically designed for the {current_domain} domain"
        
        # Check for strictest framework
        strictest_framework = self._find_strictest_framework(frameworks)
        if strictest_framework:
            return strictest_framework, "strictest_rules", f"{strictest_framework} has the strictest requirements and takes precedence for safety"
            
        # Default to most recent framework
        newest_framework = self._find_newest_framework(frameworks)
        if newest_framework:
            return newest_framework, "most_recent", f"{newest_framework} is the most recently updated framework"
            
        # If all else fails, pick the first one
        return frameworks[0], "default", f"No specific resolution rule applied, defaulting to {frameworks[0]}"
    
    def _find_strictest_framework(self, frameworks: List[str]) -> Optional[str]:
        """Find the framework with the strictest requirements"""
        strictness_scores = {}
        
        for framework in frameworks:
            if framework in self.framework_data:
                strictness_scores[framework] = self.framework_data[framework].get("strictness_score", 0)
                
        if strictness_scores:
            return max(strictness_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _find_newest_framework(self, frameworks: List[str]) -> Optional[str]:
        """Find the most recently updated framework"""
        dates = {}
        
        for framework in frameworks:
            if framework in self.framework_data:
                last_updated = self.framework_data[framework].get("last_updated")
                if last_updated:
                    dates[framework] = datetime.datetime.fromisoformat(last_updated)
                    
        if dates:
            return max(dates.items(), key=lambda x: x[1])[0]
        return None
    
    def _gather_legal_references(self, 
                               winning_framework: str, 
                               losing_frameworks: List[str],
                               method: str) -> List[str]:
        """Gather relevant legal references for the resolution"""
        references = []
        
        # Add references for the winning framework
        if winning_framework in self.framework_data:
            framework_refs = self.framework_data[winning_framework].get("legal_references", [])
            references.extend(framework_refs)
            
        # Add precedence-related references
        if method == "explicit_precedence":
            for loser in losing_frameworks:
                if (winning_framework in self.precedence_rules and 
                    loser in self.precedence_rules[winning_framework]):
                    rule_refs = self.precedence_rules[winning_framework][loser].get("references", [])
                    references.extend(rule_refs)
                    
        return references
    
    def _identify_edge_cases(self, 
                           winning_framework: str, 
                           losing_frameworks: List[str],
                           context: Dict[str, Any]) -> List[str]:
        """Identify edge cases where this resolution might not apply"""
        edge_cases = []
        
        # Add framework-specific edge cases
        if winning_framework in self.framework_data:
            framework_edge_cases = self.framework_data[winning_framework].get("edge_cases", [])
            edge_cases.extend(framework_edge_cases)
            
        # Add general edge cases
        edge_cases.append("Different jurisdiction might change precedence")
        edge_cases.append("Future regulatory updates might change this resolution")
        
        return edge_cases
    
    def _generate_recommendation(self, 
                               winning_framework: str, 
                               losing_frameworks: List[str],
                               method: str,
                               conflict_level: ConflictLevel) -> str:
        """Generate a recommendation based on the resolution"""
        if conflict_level == ConflictLevel.CRITICAL:
            return "Seek legal counsel for definitive guidance on resolving this critical regulatory conflict"
        elif conflict_level == ConflictLevel.MAJOR:
            return f"Apply {winning_framework} requirements but document compliance approach for all frameworks"
        elif conflict_level == ConflictLevel.MODERATE:
            return f"Follow {winning_framework} requirements and ensure they don't explicitly violate other frameworks"
        else:
            return f"Apply {winning_framework} requirements with minimal concern for conflicts"
    
    def _generate_critical_conflict_resolution(self, 
                                            frameworks: List[str], 
                                            context: Dict[str, Any]) -> ExplainedResolution:
        """Generate a resolution for critical conflicts"""
        # For critical conflicts, we recommend human review and provide detailed analysis
        
        # Analyze the nature of the conflict
        conflict_analysis = self._analyze_critical_conflict(frameworks, context)
        
        # Generate a justification that explains the conflict
        justification = (
            f"Critical conflict detected between {', '.join(frameworks)}. "
            f"{conflict_analysis} This type of conflict requires legal expertise to resolve properly."
        )
        
        # Gather all potentially relevant legal references
        legal_references = []
        for framework in frameworks:
            if framework in self.framework_data:
                framework_refs = self.framework_data[framework].get("legal_references", [])
                legal_references.extend(framework_refs)
                
        # Identify strict requirements from each framework that might be in conflict
        edge_cases = []
        for framework in frameworks:
            if framework in self.framework_data:
                requirements = self.framework_data[framework].get("strict_requirements", [])
                for req in requirements:
                    edge_cases.append(f"{framework}: {req}")
                    
        return ExplainedResolution(
            winning_framework="REQUIRES_LEGAL_GUIDANCE",
            losing_frameworks=[],
            resolution_method="escalate_to_legal",
            conflict_level=ConflictLevel.CRITICAL,
            justification=justification,
            legal_references=legal_references,
            edge_cases=edge_cases,
            recommendation="Escalate to legal team for resolution. Document decision and rationale.",
            human_review_needed=True
        )
    
    def _analyze_critical_conflict(self, 
                                 frameworks: List[str], 
                                 context: Dict[str, Any]) -> str:
        """Analyze the nature of a critical conflict"""
        # This would be more sophisticated in a real implementation
        framework_names = ", ".join(frameworks)
        return f"These frameworks have fundamentally different approaches to compliance in this context. "


# Example usage of the enhanced conflict resolver
def demonstrate_conflict_resolution():
    # Sample framework data
    framework_data = {
        "GDPR": {
            "name": "General Data Protection Regulation",
            "description": "EU regulation on data protection and privacy",
            "jurisdictions": ["EU", "EEA"],
            "domains": ["data_privacy", "digital_services"],
            "strictness_score": 8,
            "last_updated": "2018-05-25T00:00:00",
            "legal_references": [
                "GDPR Article 5 - Principles relating to processing of personal data",
                "GDPR Article 6 - Lawfulness of processing"
            ],
            "strict_requirements": [
                "Explicit consent required for processing personal data",
                "Right to be forgotten must be honored"
            ],
            "edge_cases": [
                "Legal obligation may override certain GDPR requirements"
            ]
        },
        "HIPAA": {
            "name": "Health Insurance Portability and Accountability Act",
            "description": "US regulation for medical information privacy",
            "jurisdictions": ["US"],
            "domains": ["healthcare", "insurance"],
            "strictness_score": 7,
            "last_updated": "2013-01-25T00:00:00",
            "legal_references": [
                "45 CFR Part 160 - General Administrative Requirements",
                "45 CFR Part 164 - Security and Privacy"
            ],
            "strict_requirements": [
                "Patient authorization required for disclosure of PHI",
                "Minimum necessary standard must be applied"
            ],
            "edge_cases": [
                "Emergency situations may permit certain disclosures"
            ]
        },
        "CCPA": {
            "name": "California Consumer Privacy Act",
            "description": "California law on consu mer data privacy",
            "jurisdictions": ["US-CA"],
            "domains": ["data_privacy", "consumer_protection"],
            "strictness_score": 6,
            "last_updated": "2020-01-01T00:00:00",
            "legal_references": [
                "California Civil Code § 1798.100-1798.199"
            ],
            "strict_requirements": [
                "Right to opt out of sale of personal information",
                "Businesses must disclose categories of personal information collected"
            ],
            "edge_cases": [
                "Certain business relationships exempt from 'sale' provisions"
            ]
        }
    }
    
    # Sample precedence rules
    precedence_rules = {
        "HIPAA": {
            "GDPR": {
                "rule": "domain_specific_supersedes",
                "references": [
                    "HHS Guidance on HIPAA & GDPR",
                    "Legal Opinion 2019-05"
                ]
            }
        }
    }
    
    # Create resolver
    resolver = EnhancedConflictResolver(framework_data, precedence_rules)
    
    # Resolve conflict for a healthcare scenario in the US
    healthcare_context = {
        "jurisdiction": "US",
        "domain": "healthcare",
        "data_types": ["medical_records", "patient_contact_info"]
    }
    
    resolution = resolver.resolve_conflict(
        frameworks=["GDPR", "HIPAA", "CCPA"],
        context=healthcare_context
    )
    
    return resolution

# Example output:
# resolution = demonstrate_conflict_resolution()
# print(f"Winning framework: {resolution.winning_framework}")
# print(f"Justification: {resolution.justification}")
# print(f"Recommendation: {resolution.recommendation}")
# print(f"Human review needed: {resolution.human_review_needed}")