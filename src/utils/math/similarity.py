import numpy as np
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