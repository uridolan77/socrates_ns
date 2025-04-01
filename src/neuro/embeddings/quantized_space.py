import numpy as np
from sklearn.random_projection import sparse_random_matrix
import torch
from torch import nn

class QuantizedRegulatoryEmbeddingSpace:
    """Memory-efficient quantized embedding space for regulatory concepts"""
    
    def __init__(self, original_embeddings, quantization_bits=8, use_mixed_precision=True):
        self.quantization_bits = quantization_bits
        self.regulatory_dim = original_embeddings.regulatory_dim
        self.use_mixed_precision = use_mixed_precision
        
        # Quantize framework vectors
        self.framework_vectors = self._quantize_embeddings(
            original_embeddings.framework_vectors,
            bits=quantization_bits
        )
        
        # Quantize concept embeddings, potentially with mixed precision
        if use_mixed_precision:
            # Use higher precision for critical concepts
            self.concept_embeddings = {}
            for concept, embedding in original_embeddings.concept_embeddings.items():
                if concept in original_embeddings.critical_concepts:
                    # Use 16-bit for critical concepts
                    self.concept_embeddings[concept] = self._quantize_embeddings(
                        {concept: embedding}, bits=16
                    )[concept]
                else:
                    # Use standard bits for other concepts
                    self.concept_embeddings[concept] = self._quantize_embeddings(
                        {concept: embedding}, bits=quantization_bits
                    )[concept]
        else:
            # Use uniform quantization
            self.concept_embeddings = self._quantize_embeddings(
                original_embeddings.concept_embeddings,
                bits=quantization_bits
            )
        
        # Calculate memory savings
        self.memory_reduction = self._calculate_memory_reduction(original_embeddings)
        
    def _quantize_embeddings(self, embeddings_dict, bits):
        """Quantize embeddings to specified bit precision"""
        quantized_dict = {}
        
        for key, embedding in embeddings_dict.items():
            # Calculate scaling factor
            max_val = torch.max(torch.abs(embedding)).item()
            scale = (2**(bits-1) - 1) / max_val if max_val > 0 else 1.0
            
            # Quantize
            if bits <= 8:
                # Use int8
                quantized = torch.round(embedding * scale).to(torch.int8)
                quantized_dict[key] = {
                    'quantized': quantized,
                    'scale': scale,
                    'bits': bits
                }
            elif bits <= 16:
                # Use int16
                quantized = torch.round(embedding * scale).to(torch.int16)
                quantized_dict[key] = {
                    'quantized': quantized,
                    'scale': scale,
                    'bits': bits
                }
            else:
                # No quantization needed
                quantized_dict[key] = embedding
                
        return quantized_dict

        
    def project_to_regulatory_space(self, embedding):
        """
        Project a language model embedding to quantized regulatory space
        
        Args:
            embedding: Original embedding vector from language model
            
        Returns:
            Quantized embedding in regulatory space
        """
        # Ensure embedding is numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        # Apply domain projections
        domain_projections = {}
        for domain, matrix in self.domain_projection_matrices.items():
            domain_projections[domain] = embedding @ matrix
        
        # Concatenate domain projections
        concatenated = np.concatenate(list(domain_projections.values()))
        
        # Apply random projection for dimension reduction
        if len(concatenated) > self.regulatory_dim:
            projected = concatenated @ self.projection_matrix
        else:
            projected = concatenated
            
        # Quantize the embedding
        quantized = self._quantize_embedding(projected)
        
        return quantized
        
    def compute_compliance_scores(self, quantized_embedding, frameworks):
        """
        Compute compliance scores against regulatory frameworks
        
        Args:
            quantized_embedding: Quantized embedding in regulatory space
            frameworks: List of regulatory frameworks to check
            
        Returns:
            Dict of compliance scores by framework
        """
        compliance_scores = {}
        
        # Dequantize for computation
        dequantized = self._dequantize_embedding(quantized_embedding)
        
        for framework in frameworks:
            framework_id = framework.id
            
            if framework_id in self.framework_embeddings:
                # Get framework embedding (already dequantized for efficiency)
                framework_embedding = self.framework_embeddings[framework_id]
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(dequantized, framework_embedding)
                
                # Convert to compliance score (higher similarity = higher compliance)
                compliance_score = self._convert_to_compliance_score(similarity)
                compliance_scores[framework_id] = compliance_score
                
        return compliance_scores
        
    def find_nearest_concepts(self, quantized_embedding, top_k=5):
        """
        Find regulatory concepts closest to the embedding
        
        Args:
            quantized_embedding: Quantized embedding in regulatory space
            top_k: Number of concepts to return
            
        Returns:
            List of (concept_id, similarity) tuples
        """
        # Dequantize for computation
        dequantized = self._dequantize_embedding(quantized_embedding)
        
        # Calculate similarities to all concepts
        similarities = []
        for concept_id, concept_embedding in self.concept_embeddings.items():
            similarity = self._cosine_similarity(dequantized, concept_embedding)
            similarities.append((concept_id, similarity))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return similarities[:top_k]
        
    def detect_compliance_violations(self, quantized_embedding, frameworks):
        """
        Detect potential compliance violations
        
        Args:
            quantized_embedding: Quantized embedding in regulatory space
            frameworks: List of regulatory frameworks to check
            
        Returns:
            List of potential violations
        """
        violations = []
        dequantized = self._dequantize_embedding(quantized_embedding)
        
        for framework in frameworks:
            framework_id = framework.id
            
            if framework_id in self.compliance_boundaries:
                boundaries = self.compliance_boundaries[framework_id]
                
                for boundary_id, boundary in boundaries.items():
                    # Check if embedding violates boundary
                    distance = self._signed_distance_to_boundary(dequantized, boundary)
                    
                    if distance < 0:  # Negative distance = violation
                        violations.append({
                            "framework_id": framework_id,
                            "boundary_id": boundary_id,
                            "distance": abs(distance),
                            "severity": self._calculate_violation_severity(distance)
                        })
                        
        return violations
        
    def _initialize_projection_matrices(self):
        """Initialize domain-specific projection matrices"""
        matrices = {}
        
        # Get regulatory domains
        domains = self.regulatory_kb.get_domains()
        domain_dim = self.embedding_dim // max(len(domains), 1)
        
        for domain in domains:
            # Create semi-orthogonal random projection matrix
            random_matrix = np.random.randn(self.embedding_dim, domain_dim)
            q, _ = np.linalg.qr(random_matrix)
            matrices[domain] = q
            
        return matrices
        
    def _initialize_random_projection(self):
        """Initialize random projection matrix for dimension reduction"""
        # Johnson-Lindenstrauss projection
        total_dim = sum(m.shape[1] for m in self.domain_projection_matrices.values())
        matrix = np.random.randn(total_dim, self.regulatory_dim) / np.sqrt(self.regulatory_dim)
        return matrix
        
    def _initialize_framework_embeddings(self):
        """Initialize framework embeddings"""
        embeddings = {}
        
        # Get all frameworks
        frameworks = self.regulatory_kb.get_all_frameworks()
        
        for framework in frameworks:
            # Generate framework embedding (placeholder - would use real embeddings)
            framework_id = framework.id
            raw_embedding = np.random.randn(self.regulatory_dim)
            normalized = raw_embedding / np.linalg.norm(raw_embedding)
            
            # Store normalized but not quantized for computation efficiency
            embeddings[framework_id] = normalized
            
        return embeddings
        
    def _initialize_concept_embeddings(self):
        """Initialize regulatory concept embeddings"""
        embeddings = {}
        
        # Get regulatory concepts
        concepts = self.regulatory_kb.get_all_concepts()
        
        for concept in concepts:
            # Generate concept embedding (placeholder - would use real embeddings)
            concept_id = concept.id
            raw_embedding = np.random.randn(self.regulatory_dim)
            normalized = raw_embedding / np.linalg.norm(raw_embedding)
            
            # Store normalized but not quantized
            embeddings[concept_id] = normalized
            
        return embeddings
        
    def _initialize_compliance_boundaries(self):
        """Initialize compliance boundaries for frameworks"""
        boundaries = {}
        
        # Get all frameworks
        frameworks = self.regulatory_kb.get_all_frameworks()
        
        for framework in frameworks:
            framework_id = framework.id
            framework_boundaries = {}
            
            # Get compliance requirements
            requirements = framework.get_requirements()
            
            for i, req in enumerate(requirements):
                # Create boundary hyperplane for each requirement
                boundary_id = f"{framework_id}_boundary_{i}"
                
                # Random hyperplane (placeholder - would be learned from data)
                normal = np.random.randn(self.regulatory_dim)
                normal = normal / np.linalg.norm(normal)
                offset = np.random.uniform(0.1, 0.5)
                
                framework_boundaries[boundary_id] = {
                    "normal": normal,
                    "offset": offset,
                    "requirement": req
                }
                
            boundaries[framework_id] = framework_boundaries
            
        return boundaries
        
    def _calibrate_quantization(self):
        """Calibrate quantization parameters based on sample embeddings"""
        # In a real system, this would analyze a dataset of embeddings
        # For now, use reasonable defaults for normalized embeddings
        self.embedding_stats = {
            "mean": 0.0,
            "std": 0.25,
            "min": -1.0,
            "max": 1.0
        }
        
        # Calculate scale factor for quantization
        self.scale_factor = self.quant_range / (self.embedding_stats["max"] - self.embedding_stats["min"])
        
    def _quantize_embedding(self, embedding):
        """Quantize an embedding vector to reduced precision"""
        # Scale to quantization range
        scaled = (embedding - self.embedding_stats["min"]) * self.scale_factor + self.quant_min
        
        # Clamp to valid range
        clamped = np.clip(scaled, self.quant_min, self.quant_max)
        
        # Convert to integers
        quantized = np.round(clamped).astype(np.int8)
        
        return quantized
        
    def _dequantize_embedding(self, quantized):
        """Dequantize an embedding back to floating point"""
        # Convert back to original range
        dequantized = (quantized - self.quant_min) / self.scale_factor + self.embedding_stats["min"]
        return dequantized
        
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
        
    def _convert_to_compliance_score(self, similarity):
        """Convert similarity to compliance score"""
        # Sigmoid transformation to emphasize differences near decision boundary
        return 1.0 / (1.0 + np.exp(-10 * (similarity - 0.5)))
        
    def _signed_distance_to_boundary(self, embedding, boundary):
        """Calculate signed distance to boundary hyperplane"""
        # Positive = compliant side, negative = non-compliant
        return np.dot(embedding, boundary["normal"]) - boundary["offset"]
        
    def _calculate_violation_severity(self, distance):
        """Calculate violation severity based on distance"""
        distance = abs(distance)
        if distance > 0.3:
            return "high"
        elif distance > 0.1:
            return "medium"
        else:
            return "low"