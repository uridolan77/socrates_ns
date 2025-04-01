import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from src.ner.classifiers import ComplianceBoundaryClassifier
from src.utils.math.similarity import CosineSimilarityCalculator

class RegulatoryEmbeddingSpace:
    """
    Projects language model embeddings into a specialized regulatory space
    optimized for compliance verification.
    """
    def __init__(self, embedding_dim, regulatory_knowledge_base):
        self.embedding_dim = embedding_dim
        self.regulatory_kb = regulatory_knowledge_base
        
        # Initialize projection matrices
        self.projection_matrices = self._initialize_projection_matrices()
        
        # Initialize framework comparison vectors
        self.framework_vectors = self._initialize_framework_vectors()
        
        # Initialize concept embedding cache
        self.concept_embeddings = {}
        
        # Dimension reduction for efficiency
        self.regulatory_dim = min(embedding_dim, 128)  # Regulatory space dimension
        self.pca = self._initialize_dimensionality_reduction()
        
        # Initialize semantic similarity calculator
        self.similarity_calculator = CosineSimilarityCalculator()
        
        # Clustering for detecting compliance regions
        self.compliance_clusters = self._initialize_compliance_regions()
        
        # Compliance boundary classifier
        self.boundary_classifier = ComplianceBoundaryClassifier(
            regulatory_dim=self.regulatory_dim,
            num_frameworks=len(self.framework_vectors)
        )
        
    def project_to_regulatory_space(self, embedding):
        """
        Project a language model embedding to regulatory space.
        
        Args:
            embedding: Original language model embedding
            
        Returns:
            Projected embedding in regulatory space
        """
        # Ensure embedding is a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        # Apply domain-specific projections
        domain_projections = {}
        for domain, matrix in self.projection_matrices.items():
            domain_projections[domain] = embedding @ matrix
            
        # Combine domain projections
        combined = np.concatenate(list(domain_projections.values()))
        
        # Apply dimensionality reduction if combined vector is too large
        if len(combined) > self.regulatory_dim:
            regulatory_embedding = self.pca.transform(combined.reshape(1, -1))[0]
        else:
            # Pad if necessary
            if len(combined) < self.regulatory_dim:
                padding = np.zeros(self.regulatory_dim - len(combined))
                regulatory_embedding = np.concatenate([combined, padding])
            else:
                regulatory_embedding = combined
                
        # Normalize embedding
        regulatory_embedding = regulatory_embedding / np.linalg.norm(regulatory_embedding)
        
        return regulatory_embedding
    
    def compute_compliance_scores(self, regulatory_embedding, frameworks):
        """
        Compute compliance scores against regulatory frameworks.
        
        Args:
            regulatory_embedding: Embedding in regulatory space
            frameworks: List of regulatory frameworks to check against
            
        Returns:
            Dict of compliance scores per framework
        """
        compliance_scores = {}
        
        # Calculate similarity to framework vectors
        for framework in frameworks:
            framework_id = framework.id
            
            if framework_id in self.framework_vectors:
                framework_vector = self.framework_vectors[framework_id]
                
                # Calculate similarity score
                similarity = self.similarity_calculator.calculate(
                    regulatory_embedding, framework_vector
                )
                
                # Convert similarity to compliance score (higher similarity = higher compliance)
                compliance_score = self._convert_similarity_to_compliance(similarity)
                
                compliance_scores[framework_id] = compliance_score
        
        return compliance_scores
    
    def detect_potential_violations(self, regulatory_embedding, frameworks):
        """
        Detect potential violations by identifying closest non-compliant regions.
        
        Args:
            regulatory_embedding: Embedding in regulatory space
            frameworks: List of regulatory frameworks
            
        Returns:
            List of potential violations with distances
        """
        potential_violations = []
        
        # Check boundary classifier for each framework
        for framework in frameworks:
            framework_id = framework.id
            is_compliant, distance, violation_type = self.boundary_classifier.check_compliance(
                regulatory_embedding, framework_id
            )
            
            if not is_compliant:
                potential_violations.append({
                    "framework_id": framework_id,
                    "violation_type": violation_type,
                    "distance": distance,
                    "severity": self._calculate_violation_severity(distance)
                })
                
        # Sort by severity (most severe first)
        potential_violations.sort(key=lambda v: {"high": 0, "medium": 1, "low": 2}[v["severity"]])
        
        return potential_violations
    
    def get_concept_compliance(self, regulatory_embedding, concepts):
        """
        Check compliance against specific regulatory concepts.
        
        Args:
            regulatory_embedding: Embedding in regulatory space
            concepts: List of regulatory concepts to check against
            
        Returns:
            Dict of concept compliance scores
        """
        concept_scores = {}
        
        for concept in concepts:
            # Get or compute concept embedding
            concept_embedding = self._get_concept_embedding(concept)
            
            if concept_embedding is not None:
                # Calculate similarity to concept vector
                similarity = self.similarity_calculator.calculate(
                    regulatory_embedding, concept_embedding
                )
                
                # Convert to concept compliance score
                concept_scores[concept] = similarity
                
        return concept_scores
    
    def _initialize_projection_matrices(self):
        """Initialize domain-specific projection matrices"""
        matrices = {}
        
        # Get regulatory domains from knowledge base
        domains = self.regulatory_kb.get_domains()
        
        for domain in domains:
            # Create domain-specific projection matrix
            # (In practice, these would be learned from domain-specific data)
            domain_dim = min(self.embedding_dim // len(domains), 64)
            
            # Initialize with random orthogonal matrix for now
            # In a real system, these would be trained on domain-specific data
            random_matrix = np.random.randn(self.embedding_dim, domain_dim)
            q_matrix, _ = np.linalg.qr(random_matrix)
            
            matrices[domain] = q_matrix
            
        return matrices
    
    def _initialize_framework_vectors(self):
        """Initialize vectors representing regulatory frameworks"""
        framework_vectors = {}
        
        # Get all frameworks from knowledge base
        frameworks = self.regulatory_kb.get_all_frameworks()
        
        for framework in frameworks:
            # Get framework text representation
            framework_text = self._get_framework_representation(framework)
            
            # Get base embedding for framework text
            # (This is a placeholder - in a real system, you'd use the language model)
            base_embedding = np.random.randn(self.embedding_dim)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            
            # Project to regulatory space
            framework_vector = self.project_to_regulatory_space(base_embedding)
            
            framework_vectors[framework.id] = framework_vector
            
        return framework_vectors
    
    def _initialize_dimensionality_reduction(self):
        """Initialize dimensionality reduction component"""
        # Simple PCA implementation for dimensionality reduction
        # In a real system, you might use more sophisticated methods
        class SimplePCA:
            def __init__(self, n_components):
                self.n_components = n_components
                self.components = None
                
            def fit(self, X):
                # Center the data
                X_centered = X - np.mean(X, axis=0)
                
                # SVD decomposition
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                
                # Get top components
                self.components = Vt[:self.n_components]
                
                return self
                
            def transform(self, X):
                # Project data onto principal components
                return X @ self.components.T
                
        # Create and return PCA instance
        pca = SimplePCA(n_components=self.regulatory_dim)
        
        # In a real system, you would fit this on a dataset of regulatory texts
        # For now, we'll return the uninitialized PCA
        return pca
    
    def _initialize_compliance_regions(self):
        """Initialize compliance region clusters"""
        # In a real system, these would be learned from labeled data
        # Simple placeholder implementation
        clusters = {
            "compliant": np.random.randn(5, self.regulatory_dim),
            "non_compliant": np.random.randn(5, self.regulatory_dim)
        }
        
        # Normalize cluster centroids
        for category, centroids in clusters.items():
            for i in range(len(centroids)):
                clusters[category][i] = centroids[i] / np.linalg.norm(centroids[i])
                
        return clusters
    
    def _get_framework_representation(self, framework):
        """Get text representation of framework (placeholder)"""
        # In a real system, this would extract key information from the framework
        return f"Regulatory framework {framework.id} with key provisions"
    
    def _get_concept_embedding(self, concept):
        """Get or compute embedding for a regulatory concept"""
        if concept in self.concept_embeddings:
            return self.concept_embeddings[concept]
            
        # Get concept text from knowledge base
        concept_text = self.regulatory_kb.get_concept_description(concept)
        
        if not concept_text:
            return None
            
        # Get base embedding for concept text
        # (This is a placeholder - in a real system, you'd use the language model)
        base_embedding = np.random.randn(self.embedding_dim)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        # Project to regulatory space
        concept_embedding = self.project_to_regulatory_space(base_embedding)
        
        # Cache for future use
        self.concept_embeddings[concept] = concept_embedding
        
        return concept_embedding
    
    def _convert_similarity_to_compliance(self, similarity):
        """Convert similarity score to compliance score"""
        # Adjust similarity to compliance scale
        # Higher similarity should indicate higher compliance
        # Scale from 0-1 with non-linear transformation
        
        # Simple sigmoid-based transformation
        return 1.0 / (1.0 + np.exp(-10 * (similarity - 0.5)))
    
    def _calculate_violation_severity(self, distance):
        """Calculate violation severity based on distance to compliance boundary"""
        if distance > 0.5:
            return "high"
        elif distance > 0.2:
            return "medium"
        else:
            return "low"