from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass


class Framework(Enum):
    """Enum representing supported regulatory frameworks."""
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    CCPA = "CCPA"
    # Add more frameworks as they're implemented


class ConceptType(Enum):
    """Types of regulatory concepts."""
    PRINCIPLE = "principle"
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    RIGHT = "right"
    ENTITY = "entity"
    PROCESS = "process"
    DEFINITION = "definition"


@dataclass
class RegulatoryReference:
    """Reference to a specific part of a regulation."""
    framework: Framework
    reference_id: str  # e.g., "art5" for GDPR Article 5
    title: str  # e.g., "Principles relating to processing of personal data"
    text: str
    parent_id: Optional[str] = None  # Reference to parent (e.g., chapter or section)


@dataclass
class RegulatoryConcept:
    """Representation of a regulatory concept."""
    id: str
    name: str
    description: str
    concept_type: ConceptType
    references: List[RegulatoryReference]  # References that define/mention this concept
    related_concepts: List[str]  # IDs of related concepts
    examples: List[str] = None  # Example applications/violations
    metadata: Dict[str, Any] = None


@dataclass
class ComplianceRequirement:
    """A specific compliance requirement."""
    id: str
    reference: RegulatoryReference
    description: str
    applicability_conditions: Dict[str, Any]  # Conditions under which this requirement applies
    severity: str  # e.g., "high", "medium", "low"
    validation_criteria: List[Dict[str, Any]]  # Criteria to validate compliance
    related_concepts: List[str]  # IDs of related concepts


@dataclass
class ComplianceConstraint:
    """A specific constraint for ensuring compliance."""
    id: str
    requirement_id: str  # The requirement this constraint helps satisfy
    description: str
    constraint_type: str  # e.g., "prohibition", "limitation", "obligation"
    implementation: Dict[str, Any]  # Technical details for implementing the constraint
    severity: str  # Impact of violating this constraint


class RegulatoryKnowledgeBase(ABC):
    """Interface for a regulatory knowledge base."""

    @abstractmethod
    def get_frameworks(self) -> List[Framework]:
        """Get all supported regulatory frameworks."""
        pass

    @abstractmethod
    def get_regulatory_references(self, framework: Framework, parent_id: Optional[str] = None) -> List[RegulatoryReference]:
        """
        Get regulatory references for a framework.
        
        Args:
            framework: The regulatory framework to query
            parent_id: Optional parent ID to get child references
            
        Returns:
            List of regulatory references
        """
        pass

    @abstractmethod
    def get_reference_by_id(self, framework: Framework, reference_id: str) -> Optional[RegulatoryReference]:
        """Get a specific regulatory reference by ID."""
        pass

    @abstractmethod
    def get_concepts(self, framework: Optional[Framework] = None, concept_type: Optional[ConceptType] = None) -> List[RegulatoryConcept]:
        """
        Get regulatory concepts, optionally filtered by framework and/or type.
        
        Args:
            framework: Optional framework to filter by
            concept_type: Optional concept type to filter by
            
        Returns:
            List of matching regulatory concepts
        """
        pass

    @abstractmethod
    def get_concept_by_id(self, concept_id: str) -> Optional[RegulatoryConcept]:
        """Get a specific concept by ID."""
        pass

    @abstractmethod
    def search_concepts(self, query: str, framework: Optional[Framework] = None) -> List[RegulatoryConcept]:
        """
        Search for concepts matching a query string.
        
        Args:
            query: Search string
            framework: Optional framework to limit search
            
        Returns:
            List of matching concepts
        """
        pass

    @abstractmethod
    def get_related_concepts(self, concept_id: str, max_distance: int = 1) -> List[RegulatoryConcept]:
        """
        Get concepts related to the specified concept.
        
        Args:
            concept_id: ID of the concept to find relations for
            max_distance: Maximum relation distance (1 = direct relations only)
            
        Returns:
            List of related concepts
        """
        pass

    @abstractmethod
    def get_requirements(self, framework: Framework, context: Optional[Dict[str, Any]] = None) -> List[ComplianceRequirement]:
        """
        Get compliance requirements for a framework, optionally filtered by context.
        
        Args:
            framework: The regulatory framework
            context: Optional context information (e.g., data types, processing purposes)
            
        Returns:
            List of applicable compliance requirements
        """
        pass

    @abstractmethod
    def get_requirements_for_concept(self, concept_id: str) -> List[ComplianceRequirement]:
        """
        Get requirements related to a specific concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            List of related requirements
        """
        pass

    @abstractmethod
    def get_constraints(self, requirement_ids: List[str]) -> List[ComplianceConstraint]:
        """
        Get compliance constraints for specified requirements.
        
        Args:
            requirement_ids: List of requirement IDs
            
        Returns:
            List of constraints for implementing those requirements
        """
        pass

    @abstractmethod
    def resolve_conflicts(self, constraints: List[ComplianceConstraint]) -> List[ComplianceConstraint]:
        """
        Resolve conflicts between compliance constraints.
        
        Args:
            constraints: List of potentially conflicting constraints
            
        Returns:
            List of resolved constraints
        """
        pass

    @abstractmethod
    def get_applicable_frameworks(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Framework]:
        """
        Determine which frameworks apply to the given text and context.
        
        Args:
            text: Text to analyze
            context: Optional context information
            
        Returns:
            List of applicable frameworks
        """
        pass

    @abstractmethod
    def explain_requirement(self, requirement_id: str) -> Dict[str, Any]:
        """
        Generate human-readable explanation for a requirement.
        
        Args:
            requirement_id: ID of the requirement
            
        Returns:
            Explanation including references, examples, and guidance
        """
        pass


class GDPRKnowledgeBase(RegulatoryKnowledgeBase):
    """Implementation of knowledge base focused on GDPR."""
    
    def __init__(self, data_source: Optional[str] = None):
        """
        Initialize GDPR knowledge base.
        
        Args:
            data_source: Optional path to data source (file or API endpoint)
        """
        self.data_source = data_source
        self._load_gdpr_data()
        
    def _load_gdpr_data(self):
        """Load GDPR data from the source."""
        # Implementation will load data from file or API
        # This would populate:
        # - self.references: Dict mapping reference IDs to RegulatoryReference objects
        # - self.concepts: Dict mapping concept IDs to RegulatoryConcept objects
        # - self.requirements: Dict mapping requirement IDs to ComplianceRequirement objects
        # - self.constraints: Dict mapping constraint IDs to ComplianceConstraint objects
        pass
        
    def get_frameworks(self) -> List[Framework]:
        """Get all supported regulatory frameworks."""
        # For now, only GDPR is supported
        return [Framework.GDPR]
        
    def get_regulatory_references(self, framework: Framework, parent_id: Optional[str] = None) -> List[RegulatoryReference]:
        """Get regulatory references for GDPR."""
        if framework != Framework.GDPR:
            return []
            
        # Filter references by parent_id if provided
        if parent_id:
            return [ref for ref in self.references.values() if ref.parent_id == parent_id]
        else:
            return list(self.references.values())
            
    def get_reference_by_id(self, framework: Framework, reference_id: str) -> Optional[RegulatoryReference]:
        """Get a specific GDPR reference by ID."""
        if framework != Framework.GDPR:
            return None
            
        return self.references.get(reference_id)
        
    def get_concepts(self, framework: Optional[Framework] = None, concept_type: Optional[ConceptType] = None) -> List[RegulatoryConcept]:
        """Get GDPR concepts, optionally filtered by type."""
        if framework is not None and framework != Framework.GDPR:
            return []
            
        concepts = list(self.concepts.values())
        
        # Filter by concept_type if provided
        if concept_type:
            concepts = [c for c in concepts if c.concept_type == concept_type]
            
        return concepts
        
    def get_concept_by_id(self, concept_id: str) -> Optional[RegulatoryConcept]:
        """Get a specific concept by ID."""
        return self.concepts.get(concept_id)
        
    def search_concepts(self, query: str, framework: Optional[Framework] = None) -> List[RegulatoryConcept]:
        """Search for GDPR concepts matching a query string."""
        if framework is not None and framework != Framework.GDPR:
            return []
            
        # Simple search implementation - would be more sophisticated in real implementation
        query = query.lower()
        results = []
        
        for concept in self.concepts.values():
            if (query in concept.name.lower() or 
                query in concept.description.lower()):
                results.append(concept)
                
        return results
        
    def get_related_concepts(self, concept_id: str, max_distance: int = 1) -> List[RegulatoryConcept]:
        """Get concepts related to the specified concept."""
        if concept_id not in self.concepts:
            return []
            
        # For now, only return direct relations (distance = 1)
        related_ids = self.concepts[concept_id].related_concepts
        return [self.concepts[rid] for rid in related_ids if rid in self.concepts]
        
    def get_requirements(self, framework: Framework, context: Optional[Dict[str, Any]] = None) -> List[ComplianceRequirement]:
        """Get GDPR compliance requirements, optionally filtered by context."""
        if framework != Framework.GDPR:
            return []
            
        requirements = list(self.requirements.values())
        
        # Filter by context if provided
        if context:
            requirements = self._filter_requirements_by_context(requirements, context)
            
        return requirements
        
    def _filter_requirements_by_context(self, requirements: List[ComplianceRequirement], context: Dict[str, Any]) -> List[ComplianceRequirement]:
        """Filter requirements based on context."""
        # This would implement contextual filtering logic
        # Examples:
        # - If "data_types" includes "health_data", include special category requirements
        # - If "data_subjects" includes "children", include child-specific requirements
        # Placeholder implementation returns all requirements
        return requirements
        
    def get_requirements_for_concept(self, concept_id: str) -> List[ComplianceRequirement]:
        """Get requirements related to a specific concept."""
        if concept_id not in self.concepts:
            return []
            
        # Find requirements that reference this concept
        return [req for req in self.requirements.values() 
                if concept_id in req.related_concepts]
        
    def get_constraints(self, requirement_ids: List[str]) -> List[ComplianceConstraint]:
        """Get compliance constraints for specified requirements."""
        constraints = []
        
        for req_id in requirement_ids:
            # Get constraints for this requirement
            req_constraints = [c for c in self.constraints.values() 
                              if c.requirement_id == req_id]
            constraints.extend(req_constraints)
            
        return constraints
        
    def resolve_conflicts(self, constraints: List[ComplianceConstraint]) -> List[ComplianceConstraint]:
        """Resolve conflicts between GDPR compliance constraints."""
        # This would implement conflict resolution logic
        # Examples:
        # - If two constraints affect the same data but one is more restrictive, use the more restrictive one
        # - If constraints are from different requirements but affect same process, determine precedence
        # Placeholder implementation returns all constraints unchanged
        return constraints
        
    def get_applicable_frameworks(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Framework]:
        """Determine if GDPR applies to the given text and context."""
        # For now, always return GDPR as the only framework
        return [Framework.GDPR]
        
    def explain_requirement(self, requirement_id: str) -> Dict[str, Any]:
        """Generate human-readable explanation for a GDPR requirement."""
        if requirement_id not in self.requirements:
            return {"error": f"Requirement {requirement_id} not found"}
            
        requirement = self.requirements[requirement_id]
        reference = requirement.reference
        
        # Create explanation
        explanation = {
            "requirement_id": requirement_id,
            "description": requirement.description,
            "reference": {
                "id": reference.reference_id,
                "title": reference.title,
                "text": reference.text
            },
            "severity": requirement.severity,
            "related_concepts": []
        }
        
        # Add related concept information
        for concept_id in requirement.related_concepts:
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
                explanation["related_concepts"].append({
                    "id": concept_id,
                    "name": concept.name,
                    "description": concept.description
                })
                
        return explanation