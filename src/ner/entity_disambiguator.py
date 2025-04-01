import numpy as np
import re
import logging
from collections import defaultdict
import string
import difflib

class EntityDisambiguator:
    """Improves disambiguation of entities with similar surface forms"""
    
    def __init__(self, config):
        self.config = config
        self.context_window_size = config.get("context_window_size", 50)
        self.min_similarity_threshold = config.get("min_similarity_threshold", 0.7)
        self.use_embeddings = config.get("use_embeddings_for_disambiguation", True)
        self.embedding_model = None
        self.kb_linker = None
        self.context_weight = config.get("context_weight", 0.6)
        self.text_weight = config.get("text_weight", 0.4)
        
        # Minimum frequency to consider entity as ambiguous
        self.min_frequency = config.get("min_disambiguation_frequency", 2)
        
        # Cache for disambiguated entities
        self.entity_cache = {}
    
    def set_embedding_model(self, model):
        """Set embedding model for context comparison"""
        self.embedding_model = model
    
    def set_kb_linker(self, linker):
        """Set knowledge base linker for entity lookup"""
        self.kb_linker = linker
    
    def disambiguate_entities(self, text, entities):
        """
        Disambiguate entities with similar surface forms
        
        Args:
            text: Original text
            entities: Extracted entities to disambiguate
            
        Returns:
            Disambiguated entities with additional information
        """
        if not entities:
            return entities
            
        # Group entities by similar text
        entity_groups = self._group_similar_entities(entities)
        
        # Skip if no ambiguous entities found
# Skip if no ambiguous entities found
        if all(len(group) < self.min_frequency for group in entity_groups.values()):
            return entities
        
        # Process each group of similar entities
        disambiguated_entities = []
        
        for entity_text, group in entity_groups.items():
            if len(group) < self.min_frequency:
                # No disambiguation needed
                disambiguated_entities.extend(group)
                continue
            
            # For ambiguous entities, apply disambiguation
            disambiguated_group = self._disambiguate_entity_group(text, entity_text, group)
            disambiguated_entities.extend(disambiguated_group)
        
        return disambiguated_entities
    
    def _group_similar_entities(self, entities):
        """Group entities by similar text"""
        entity_groups = defaultdict(list)
        
        for entity in entities:
            entity_text = entity.get('text', '').lower()
            
            # Skip empty text
            if not entity_text:
                continue
            
            # Check cache first
            cache_key = f"{entity_text}_{entity.get('ner_type', '')}"
            if cache_key in self.entity_cache:
                canonicalized_text = self.entity_cache[cache_key]
                entity_groups[canonicalized_text].append(entity)
                continue
            
            # Find best match in existing groups
            best_match = None
            best_score = 0
            
            for group_text in entity_groups.keys():
                # Skip if different entity types in group
                group_entity_type = entity_groups[group_text][0].get('ner_type') or entity_groups[group_text][0].get('type', 'UNKNOWN')
                entity_type = entity.get('ner_type') or entity.get('type', 'UNKNOWN')
                
                if entity_type != group_entity_type:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_text_similarity(entity_text, group_text)
                
                if similarity > best_score and similarity >= self.min_similarity_threshold:
                    best_score = similarity
                    best_match = group_text
            
            if best_match:
                # Add to existing group
                entity_groups[best_match].append(entity)
                self.entity_cache[cache_key] = best_match
            else:
                # Create new group
                entity_groups[entity_text].append(entity)
                self.entity_cache[cache_key] = entity_text
        
        return entity_groups
    
    def _disambiguate_entity_group(self, text, entity_text, group):
        """Disambiguate a group of similar entities"""
        # If only one type in group, no disambiguation needed
        entity_types = set(entity.get('ner_type') or entity.get('type', 'UNKNOWN') for entity in group)
        
        if len(entity_types) == 1:
            # Just mark as disambiguated
            for entity in group:
                entity['disambiguated'] = True
                entity['canonical_form'] = entity_text
            return group
        
        # Extract context for each entity
        for entity in group:
            context = self._extract_entity_context(text, entity)
            entity['context'] = context
        
        # Group by entity type
        type_groups = defaultdict(list)
        
        for entity in group:
            entity_type = entity.get('ner_type') or entity.get('type', 'UNKNOWN')
            type_groups[entity_type].append(entity)
        
        # For each entity, compare context with others to find best type
        for entity in group:
            entity_type = entity.get('ner_type') or entity.get('type', 'UNKNOWN')
            entity_context = entity.get('context', '')
            
            # Calculate context similarity scores for each type
            type_scores = {}
            
            for candidate_type, candidates in type_groups.items():
                # Skip if same type
                if candidate_type == entity_type:
                    continue
                
                # Calculate average context similarity with this type
                similarities = []
                for candidate in candidates:
                    candidate_context = candidate.get('context', '')
                    similarity = self._calculate_context_similarity(entity_context, candidate_context)
                    similarities.append(similarity)
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    type_scores[candidate_type] = avg_similarity
            
            # Find best matching type
            current_score = 0.5  # Default score for current type
            best_type = entity_type
            best_score = current_score
            
            for candidate_type, score in type_scores.items():
                if score > best_score:
                    best_score = score
                    best_type = candidate_type
            
            # Update entity with disambiguation info
            entity['disambiguated'] = True
            entity['disambiguation_confidence'] = best_score
            
            if best_type != entity_type:
                entity['original_type'] = entity_type
                entity['ner_type'] = best_type
                entity['disambiguation_note'] = f"Type changed from {entity_type} to {best_type} based on context"
            
            entity['canonical_form'] = entity_text
        
        # Link to knowledge base if linker available
        if self.kb_linker:
            self._link_disambiguated_entities(text, group)
        
        return group
    
    def _extract_entity_context(self, text, entity):
        """Extract context window around entity"""
        start = entity.get('start', 0)
        end = entity.get('end', 0)
        
        # Get surrounding context
        context_start = max(0, start - self.context_window_size)
        context_end = min(len(text), end + self.context_window_size)
        
        left_context = text[context_start:start]
        right_context = text[end:context_end]
        
        return (left_context, right_context)
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two entity texts"""
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Check exact match
        if text1 == text2:
            return 1.0
        
        # Check if one is contained in the other
        if text1 in text2 or text2 in text1:
            shorter = text1 if len(text1) < len(text2) else text2
            longer = text2 if len(text1) < len(text2) else text1
            return len(shorter) / len(longer) * 0.9  # Slightly discount partial containment
        
        # Calculate Levenshtein ratio
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_context_similarity(self, context1, context2):
        """Calculate similarity between entity contexts"""
        # Unpack contexts
        left1, right1 = context1
        left2, right2 = context2
        
        if self.use_embeddings and self.embedding_model:
            # Use embeddings for semantic similarity
            try:
                # Combine contexts
                full_context1 = left1 + " " + right1
                full_context2 = left2 + " " + right2
                
                # Get embeddings
                emb1 = self.embedding_model.encode(full_context1)
                emb2 = self.embedding_model.encode(full_context2)
                
                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                return float(similarity)
            except Exception as e:
                logging.error(f"Error calculating embedding similarity: {e}")
                # Fall back to text-based similarity
        
        # Text-based similarity
        # Compare left and right contexts separately
        left_sim = self._calculate_text_similarity(left1, left2)
        right_sim = self._calculate_text_similarity(right1, right2)
        
        # Combine similarities
        return (left_sim + right_sim) / 2
    
    def _link_disambiguated_entities(self, text, entities):
        """Link disambiguated entities to knowledge base"""
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('ner_type') or entity.get('type', 'UNKNOWN')
            
            try:
                # Link with context
                link_result = self.kb_linker.link(entity, text)
                
                if link_result:
                    # Add link information
                    entity.update(link_result)
                    
                    # Check if linked entity type differs from current type
                    kb_type = link_result.get('kb_type')
                    
                    if kb_type and kb_type != entity_type:
                        # Knowledge base suggests different type
                        kb_confidence = link_result.get('kb_confidence', 0.0)
                        disambig_confidence = entity.get('disambiguation_confidence', 0.0)
                        
                        # Use KB type if more confident
                        if kb_confidence > disambig_confidence:
                            entity['original_type'] = entity_type
                            entity['ner_type'] = kb_type
                            entity['disambiguation_note'] = f"Type changed to {kb_type} based on knowledge base"
                            entity['disambiguation_confidence'] = kb_confidence
            except Exception as e:
                logging.error(f"Error linking disambiguated entity: {e}")