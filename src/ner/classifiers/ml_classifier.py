
from src.ner.classifiers.rule_based_classifier import RuleBasedEntityClassifier
import logging

class MLEntityClassifier:
    """Machine learning based classifier for entity types"""
    
    def __init__(self, config):
        self.config = config
        self.model = self._initialize_model()
        self.fallback_classifier = RuleBasedEntityClassifier(config)
        
    def _initialize_model(self):
        """Initialize ML model for classification"""
        model_path = self.config.get('entity_classifier_model_path')
        if model_path:
            try:
                # Example using a basic ML approach
                from sklearn.ensemble import RandomForestClassifier
                import pickle
                
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except (ImportError, FileNotFoundError):
                logging.warning(f"Could not load ML entity classifier from {model_path}")
                return None
        return None
    
    def classify(self, entity, text, embeddings=None):
        """Classify entity using ML model or fallback"""
        if self.model is None:
            return self.fallback_classifier.classify(entity, text, embeddings)
            
        # Extract features
        features = self._extract_features(entity, text, embeddings)
        
        # Make prediction
        try:
            # Get prediction probabilities
            proba = self.model.predict_proba([features])[0]
            # Get class with highest probability
            class_idx = proba.argmax()
            entity_type = self.model.classes_[class_idx]
            confidence = proba[class_idx]
            
            return entity_type, confidence
        except:
            # Fallback to rule-based on error
            return self.fallback_classifier.classify(entity, text, embeddings)
    
    def _extract_features(self, entity, text, embeddings):
        """Extract features for ML classification"""
        # This is a simplified implementation
        # A real implementation would extract various features
        features = []
        
        # Feature: entity text length
        features.append(len(entity['text']))
        
        # Feature: capitalization ratio
        caps_ratio = sum(1 for c in entity['text'] if c.isupper()) / max(1, len(entity['text']))
        features.append(caps_ratio)
        
        # Feature: digit ratio
        digit_ratio = sum(1 for c in entity['text'] if c.isdigit()) / max(1, len(entity['text']))
        features.append(digit_ratio)
        
        # Feature: Special character ratio
        special_ratio = sum(1 for c in entity['text'] if not c.isalnum()) / max(1, len(entity['text']))
        features.append(special_ratio)
        
        # Feature: entity position in text ratio
        position_ratio = entity['start'] / max(1, len(text))
        features.append(position_ratio)
        
        # Add embeddings features if available
        if embeddings is not None:
            # Use entity span embedding or extract entity embedding
            # This is a placeholder - real implementation would use actual embeddings
            features.extend([0.0] * 10)  # Placeholder for embeddings
        
        return features
