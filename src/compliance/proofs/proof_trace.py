import json
from typing import List, Optional, Dict, Any
from enum import Enum  # Add this import for defining enums like ConflictResolutionStrategy
from enum import Enum  # Add this import for defining ProofNodeStatus
# Define ComplianceProofNode or import it if it exists in another module
class ComplianceProofNode:
    """Class representing a compliance proof node."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data  # Store node data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary."""
        return self.data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceProofNode':
        """Create a ComplianceProofNode from a dictionary."""
        return cls(data)

class ComplianceProofTrace:
    """Class representing a compliance proof trace."""
    
    def __init__(self, nodes: List['ComplianceProofNode']):
        self.nodes = nodes  # List of ComplianceProofNode objects
    
    def to_json(self) -> str:
        """Convert the proof trace to JSON."""
        return json.dumps([node.to_dict() for node in self.nodes], indent=4)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ComplianceProofTrace':
        """Create a ComplianceProofTrace from JSON string."""
        data = json.loads(json_str)
        nodes = [ComplianceProofNode.from_dict(node_data) for node_data in data]
        return cls(nodes)