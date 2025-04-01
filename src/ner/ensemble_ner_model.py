import numpy as np
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

class EnsembleNERModel:
    """Combine predictions from multiple NER models using ensemble techniques"""
    
    ENSEMBLE_METHODS = ["voting", "weighted", "confidence", "priority"]
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.ensemble_method = config.get("ensemble_method", "weighted")
        self.model_weights = config.get("model_weights", {})
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.model_priorities = config.get("model_priorities", [])
        
    def add_model(self, model, weight=1.0, priority=None):
        """Add a model to the ensemble"""
        model_info = {
            "model": model,
            "weight": weight,
            "priority": priority if priority is not None else len(self.models)
        }
        self.models.append(model_info)
        
        # Update model weights dictionary
        if hasattr(model, "__class__") and hasattr(model, "__name__"):
            model_name = model.__class__.__name__
            self.model_weights[model_name] = weight
            
    def extract_entities(self, text):
        """Extract entities using ensemble of models"""
        if not self.models:
            return []
        
        # Collect predictions from all models
        all_predictions = []
        for model_info in self.models:
            model = model_info["model"]
            try:
                entities = model.extract_entities(text)
                all_predictions.append({
                    "entities": entities,
                    "weight": model_info["weight"],
                    "priority": model_info["priority"]
                })
            except Exception as e:
                logging.error(f"Error getting predictions from model: {e}")
        
        # Apply ensemble method
        if self.ensemble_method == "voting":
            return self._apply_voting_ensemble(all_predictions, text)
        elif self.ensemble_method == "weighted":
            return self._apply_weighted_ensemble(all_predictions, text)
        elif self.ensemble_method == "confidence":
            return self._apply_confidence_ensemble(all_predictions, text)
        elif self.ensemble_method == "priority":
            return self._apply_priority_ensemble(all_predictions, text)
        else:
            # Default to weighted ensemble
            return self._apply_weighted_ensemble(all_predictions, text)
    
    def _apply_voting_ensemble(self, all_predictions, text):
        """Apply voting ensemble method"""
        # Group overlapping entity predictions
        entity_groups = self._group_overlapping_entities(all_predictions, text)
        
        # Apply voting to each group
        final_entities = []
        for i, group in enumerate(entity_groups):
            if not group:
                continue
                
            # Count votes for each entity type in this group
            type_votes = defaultdict(int)
            for entity in group:
                type_votes[entity.get('ner_type', 'UNKNOWN')] += 1
            
            # Get most voted entity type
            most_voted_type = max(type_votes.items(), key=lambda x: x[1])[0]
            
            # Find entity with this type and highest confidence
            best_entity = None
            best_score = -1
            for entity in group:
                if entity.get('ner_type') == most_voted_type:
                    score = entity.get('score', 0)
                    if score > best_score:
                        best_score = score
                        best_entity = entity
            
            if best_entity:
                # Create a new entity with ensemble information
                ensemble_entity = best_entity.copy()
                ensemble_entity['id'] = f"e{i}"
                ensemble_entity['detection_method'] = f"ensemble_voting_{ensemble_entity.get('detection_method', 'unknown')}"
                ensemble_entity['votes'] = type_votes[most_voted_type]
                ensemble_entity['total_votes'] = sum(type_votes.values())
                final_entities.append(ensemble_entity)
        
        return final_entities
    
    def _apply_weighted_ensemble(self, all_predictions, text):
        """Apply weighted ensemble method"""
        # Group overlapping entity predictions
        entity_groups = self._group_overlapping_entities(all_predictions, text)
        
        # Apply weighted voting to each group
        final_entities = []
        for i, group in enumerate(entity_groups):
            if not group:
                continue
                
            # Calculate weighted votes for each entity type
            type_weighted_votes = defaultdict(float)
            for entity in group:
                # Find the model weight for this entity
                model_weight = 1.0
                for pred in all_predictions:
                    if entity in pred["entities"]:
                        model_weight = pred["weight"]
                        break
                
                # Apply weight to vote
                entity_type = entity.get('ner_type', 'UNKNOWN')
                # Consider entity confidence if available
                entity_weight = entity.get('score', 1.0) * model_weight
                type_weighted_votes[entity_type] += entity_weight
            
            # Get entity type with highest weighted votes
            best_type = max(type_weighted_votes.items(), key=lambda x: x[1])[0]
            
            # Find entity with this type and highest confidence
            best_entity = None
            best_score = -1
            for entity in group:
                if entity.get('ner_type') == best_type:
                    score = entity.get('score', 0)
                    if score > best_score:
                        best_score = score
                        best_entity = entity
            
            if best_entity:
                # Create a new entity with ensemble information
                ensemble_entity = best_entity.copy()
                ensemble_entity['id'] = f"e{i}"
                ensemble_entity['detection_method'] = f"ensemble_weighted_{ensemble_entity.get('detection_method', 'unknown')}"
                ensemble_entity['weighted_score'] = type_weighted_votes[best_type]
                final_entities.append(ensemble_entity)
        
        return final_entities
    
    def _apply_confidence_ensemble(self, all_predictions, text):
        """Apply confidence-based ensemble method"""
        # Group overlapping entity predictions
        entity_groups = self._group_overlapping_entities(all_predictions, text)
        
        # Take entity with highest confidence in each group
        final_entities = []
        for i, group in enumerate(entity_groups):
            if not group:
                continue
                
            # Find entity with highest confidence score
            best_entity = None
            best_score = -1
            for entity in group:
                score = entity.get('score', 0)
                if score > best_score:
                    best_score = score
                    best_entity = entity
            
            # Only include entities with confidence above threshold
            if best_entity and best_score >= self.confidence_threshold:
                ensemble_entity = best_entity.copy()
                ensemble_entity['id'] = f"e{i}"
                ensemble_entity['detection_method'] = f"ensemble_confidence_{ensemble_entity.get('detection_method', 'unknown')}"
                final_entities.append(ensemble_entity)
        
        return final_entities
    
    def _apply_priority_ensemble(self, all_predictions, text):
        """Apply priority-based ensemble method"""
        # Sort predictions by priority
        sorted_predictions = sorted(all_predictions, key=lambda x: x["priority"])
        
        # Start with highest priority model's predictions
        final_entities = []
        covered_spans = set()
        
        for pred in sorted_predictions:
            for entity in pred["entities"]:
                # Create a span key
                span = (entity.get('start', 0), entity.get('end', 0))
                
                # Check if this span overlaps with any already covered span
                overlaps = False
                for covered_span in covered_spans:
                    if self._spans_overlap(span, covered_span):
                        overlaps = True
                        break
                
                # If no overlap, add this entity
                if not overlaps:
                    final_entities.append(entity.copy())
                    covered_spans.add(span)
        
        # Re-assign IDs
        for i, entity in enumerate(final_entities):
            entity['id'] = f"e{i}"
            entity['detection_method'] = f"ensemble_priority_{entity.get('detection_method', 'unknown')}"
        
        return final_entities
    
    def _group_overlapping_entities(self, all_predictions, text):
        """Group overlapping entity predictions"""
        # Create a list of all entities
        all_entities = []
        for pred in all_predictions:
            all_entities.extend(pred["entities"])
        
        # Sort entities by start position
        all_entities.sort(key=lambda x: x.get('start', 0))
        
        # Group overlapping entities
        entity_groups = []
        current_group = []
        current_end = -1
        
        for entity in all_entities:
            start = entity.get('start', 0)
            end = entity.get('end', 0)
            
            # If this entity overlaps with current group, add to group
            if start <= current_end:
                current_group.append(entity)
                # Update current end position
                current_end = max(current_end, end)
            else:
                # Start a new group
                if current_group:
                    entity_groups.append(current_group)
                current_group = [entity]
                current_end = end
        
        # Add the last group
        if current_group:
            entity_groups.append(current_group)
        
        return entity_groups
    
    def _spans_overlap(self, span1, span2):
        """Check if two spans overlap"""
        return max(0, min(span1[1], span2[1]) - max(span1[0], span2[0])) > 0
    
    def set_ensemble_method(self, method):
        """Set the ensemble method"""
        if method in self.ENSEMBLE_METHODS:
            self.ensemble_method = method
            return True
        return False
    
    def set_model_weights(self, weights):
        """Set weights for models"""
        self.model_weights = weights
        
        # Update weights in model info
        for model_info in self.models:
            model = model_info["model"]
            if hasattr(model, "__class__") and hasattr(model, "__name__"):
                model_name = model.__class__.__name__
                if model_name in weights:
                    model_info["weight"] = weights[model_name]