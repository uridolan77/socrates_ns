import logging
import re
import numpy as np
from collections import defaultdict

class CoreferenceResolver:
    """Resolves coreferences to track entities across different mentions"""
    
    def __init__(self, config):
        self.config = config
        self.model_type = config.get("coref_model_type", "rule_based")
        self.neural_model = None
        self.confidence_threshold = config.get("coref_confidence_threshold", 0.5)
        self.max_mention_distance = config.get("max_mention_distance", 5)  # Number of sentences
        self.pronoun_map = self._initialize_pronoun_map()
        
        # Initialize neural model if specified
        if self.model_type != "rule_based":
            self._initialize_neural_model()
    
    def _initialize_pronoun_map(self):
        """Initialize pronoun-to-entity type mapping"""
        return {
            "he": "PERSON",
            "him": "PERSON",
            "his": "PERSON",
            "himself": "PERSON",
            "she": "PERSON",
            "her": "PERSON",
            "hers": "PERSON",
            "herself": "PERSON",
            "they": "PERSON",
            "them": "PERSON",
            "their": "PERSON",
            "themselves": "PERSON",
            "it": "THING",
            "its": "THING",
            "itself": "THING",
            "this": "THING",
            "that": "THING",
            "these": "THING",
            "those": "THING",
            "the company": "ORGANIZATION",
            "the organization": "ORGANIZATION",
            "the institution": "ORGANIZATION",
            "the agency": "ORGANIZATION",
            "the place": "LOCATION",
            "the area": "LOCATION",
            "the region": "LOCATION",
            "the country": "LOCATION"
        }
    
    def _initialize_neural_model(self):
        """Initialize neural coreference resolution model"""
        try:
            from transformers import pipeline
            self.neural_model = pipeline(
                "text2text-generation",
                model="valhalla/t5-base-e2e-coref"
            )
            logging.info(f"Initialized {self.model_type} coreference resolution model")
        except ImportError as e:
            logging.warning(f"Failed to load {self.model_type} model: {e}. Falling back to rule-based.")
            self.model_type = "rule_based"
            self.neural_model = None
        except Exception as e:
            logging.error(f"Error initializing {self.model_type} model: {e}. Falling back to rule-based.")
            self.model_type = "rule_based"
            self.neural_model = None
    
    def resolve_coreferences(self, text, entities):
        """Resolve coreferences in text and link entities"""
        if not entities:
            return entities, []
            
        # Choose resolution method based on model type
        if self.model_type == "rule_based":
            return self._rule_based_resolution(text, entities)
        elif self.model_type == "neuralcoref" and self.neural_model:
            return self._neuralcoref_resolution(text, entities)
        elif self.model_type == "allennlp" and self.neural_model:
            return self._allennlp_resolution(text, entities)
        elif self.model_type == "huggingface" and self.neural_model:
            return self._huggingface_resolution(text, entities)
        else:
            # Fall back to rule-based if neural model not available
            return self._rule_based_resolution(text, entities)
    
    def _rule_based_resolution(self, text, entities):
        """Rule-based coreference resolution"""
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda e: e.get('start', 0))
        
        # Create entity clusters
        entity_clusters = []
        mention_to_cluster = {}
        resolved_entities = []
        coreference_links = []
        
        # First pass: group entities by exact name match
        for entity in sorted_entities:
            entity_text = entity.get('text', '').lower()
            entity_type = entity.get('ner_type') or entity.get('type', 'UNKNOWN')
            
            # Skip if this is likely a pronoun
            if entity_text.lower() in self.pronoun_map:
                continue
                
            # Check if this entity matches any existing clusters
            found_cluster = False
            for cluster_id, cluster in enumerate(entity_clusters):
                # Check if any entity in the cluster matches
                for cluster_entity in cluster['entities']:
                    cluster_text = cluster_entity.get('text', '').lower()
                    cluster_type = cluster_entity.get('ner_type') or cluster_entity.get('type', 'UNKNOWN')
                    
                    # Exact match or contained match with same type
                    if ((entity_text == cluster_text or 
                         entity_text in cluster_text or 
                         cluster_text in entity_text) and
                        entity_type == cluster_type):
                        # Add to cluster
                        cluster['entities'].append(entity)
                        mention_to_cluster[entity['id']] = cluster_id
                        found_cluster = True
                        break
                        
                if found_cluster:
                    break
            
            if not found_cluster:
                # Create new cluster
                cluster_id = len(entity_clusters)
                entity_clusters.append({
                    'id': cluster_id,
                    'entities': [entity],
                    'canonical_entity': entity  # First entity is canonical for now
                })
                mention_to_cluster[entity['id']] = cluster_id
        
        # Second pass: resolve pronouns and nominal mentions
        sentences = self._split_into_sentences(text)
        
        # Locate each entity in sentences
        entity_to_sentence = {}
        for entity in sorted_entities:
            entity_start = entity.get('start', 0)
            
            # Find which sentence contains this entity
            sent_start = 0
            for sent_idx, sentence in enumerate(sentences):
                sent_end = sent_start + len(sentence)
                if sent_start <= entity_start < sent_end:
                    entity_to_sentence[entity['id']] = sent_idx
                    break
                sent_start = sent_end + 1
        
        # Resolve pronouns and nominal mentions
        for entity in sorted_entities:
            entity_text = entity.get('text', '').lower()
            entity_id = entity.get('id')
            
            # Check if this is a pronoun or nominal mention
            if entity_text in self.pronoun_map:
                # This is a pronoun, find the closest matching entity
                entity_sent_idx = entity_to_sentence.get(entity_id, 0)
                entity_start = entity.get('start', 0)
                
                # Find candidate antecedents (entities before this pronoun)
                candidates = []
                for other_entity in sorted_entities:
                    if other_entity['id'] == entity_id:
                        continue
                        
                    other_sent_idx = entity_to_sentence.get(other_entity['id'], 0)
                    other_start = other_entity.get('start', 0)
                    other_type = other_entity.get('ner_type') or other_entity.get('type', 'UNKNOWN')
                    
                    # Only consider entities before this pronoun and within reasonable distance
                    if (other_start < entity_start and 
                        entity_sent_idx - other_sent_idx <= self.max_mention_distance):
                        
                        # Check if pronoun matches entity type
                        pronoun_type = self.pronoun_map.get(entity_text)
                        if pronoun_type == "PERSON" and "PER" in other_type:
                            candidates.append((other_entity, entity_sent_idx - other_sent_idx))
                        elif pronoun_type == "ORGANIZATION" and "ORG" in other_type:
                            candidates.append((other_entity, entity_sent_idx - other_sent_idx))
                        elif pronoun_type == "LOCATION" and ("LOC" in other_type or "GPE" in other_type):
                            candidates.append((other_entity, entity_sent_idx - other_sent_idx))
                        elif pronoun_type == "THING" and other_type not in ["PERSON", "ORGANIZATION", "LOCATION"]:
                            candidates.append((other_entity, entity_sent_idx - other_sent_idx))
                
                # Sort candidates by distance (closest first)
                candidates.sort(key=lambda x: x[1])
                
                if candidates:
                    # Link with closest matching entity
                    antecedent = candidates[0][0]
                    antecedent_id = antecedent['id']
                    
                    # Find cluster for antecedent
                    if antecedent_id in mention_to_cluster:
                        cluster_id = mention_to_cluster[antecedent_id]
                        cluster = entity_clusters[cluster_id]
                        
                        # Add pronoun to cluster
                        cluster['entities'].append(entity)
                        mention_to_cluster[entity_id] = cluster_id
                        
                        # Create coreference link
                        coreference_links.append({
                            'source_id': entity_id,
                            'target_id': antecedent_id,
                            'source_text': entity_text,
                            'target_text': antecedent.get('text', ''),
                            'confidence': 0.7,
                            'relation_type': 'coreference'
                        })
            
            # Handle nominal mentions (e.g., "the company" referring to "Microsoft")
            elif entity_text.startswith("the ") and len(entity_text.split()) <= 3:
                nominal_type = None
                for key, value in self.pronoun_map.items():
                    if key == entity_text and value in ["ORGANIZATION", "LOCATION"]:
                        nominal_type = value
                        break
                
                if nominal_type:
                    # Similar logic as pronoun resolution
                    entity_sent_idx = entity_to_sentence.get(entity_id, 0)
                    entity_start = entity.get('start', 0)
                    
                    candidates = []
                    for other_entity in sorted_entities:
                        if other_entity['id'] == entity_id:
                            continue
                            
                        other_sent_idx = entity_to_sentence.get(other_entity['id'], 0)
                        other_start = other_entity.get('start', 0)
                        other_type = other_entity.get('ner_type') or other_entity.get('type', 'UNKNOWN')
                        
                        if (other_start < entity_start and 
                            entity_sent_idx - other_sent_idx <= self.max_mention_distance):
                            
                            if (nominal_type == "ORGANIZATION" and "ORG" in other_type) or \
                               (nominal_type == "LOCATION" and ("LOC" in other_type or "GPE" in other_type)):
                                candidates.append((other_entity, entity_sent_idx - other_sent_idx))
                    
                    candidates.sort(key=lambda x: x[1])
                    
                    if candidates:
                        antecedent = candidates[0][0]
                        antecedent_id = antecedent['id']
                        
                        if antecedent_id in mention_to_cluster:
                            cluster_id = mention_to_cluster[antecedent_id]
                            cluster = entity_clusters[cluster_id]
                            
                            cluster['entities'].append(entity)
                            mention_to_cluster[entity_id] = cluster_id
                            
                            coreference_links.append({
                                'source_id': entity_id,
                                'target_id': antecedent_id,
                                'source_text': entity_text,
                                'target_text': antecedent.get('text', ''),
                                'confidence': 0.7,
                                'relation_type': 'coreference'
                            })
        
        # Update entities with coreference information
        for entity in sorted_entities:
            entity_id = entity.get('id')
            if entity_id in mention_to_cluster:
                cluster_id = mention_to_cluster[entity_id]
                cluster = entity_clusters[cluster_id]
                canonical_entity = cluster['canonical_entity']
                
                # Create resolved entity
                resolved_entity = entity.copy()
                
                # Add coreference information
                resolved_entity['cluster_id'] = cluster_id
                resolved_entity['canonical_id'] = canonical_entity['id']
                
                # If this isn't the canonical entity, add link to canonical
                if entity_id != canonical_entity['id']:
                    resolved_entity['refers_to'] = canonical_entity['id']
                    resolved_entity['canonical_text'] = canonical_entity.get('text')
                
                resolved_entities.append(resolved_entity)
            else:
                # Entity not in any cluster
                resolved_entities.append(entity.copy())
        
        return resolved_entities, coreference_links
    
    def _neuralcoref_resolution(self, text, entities):
        """Coreference resolution using neuralcoref library"""
        try:
            # Process text with neuralcoref
            doc = self.neural_model(text)
            
            # Extract clusters
            clusters = list(doc._.coref_clusters)
            
            # Map spans to entity IDs
            span_to_entity = {}
            for entity in entities:
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                span_to_entity[(start, end)] = entity['id']
            
            # Create entity clusters
            entity_clusters = []
            mention_to_cluster = {}
            coreference_links = []
            
            # Process each neuralcoref cluster
            for cluster_id, cluster in enumerate(clusters):
                entity_cluster = {
                    'id': cluster_id,
                    'entities': [],
                    'canonical_entity': None
                }
                
                # Find canonical mention (typically the first one)
                canonical_mention = cluster.main
                canonical_entity = None
                
                # Find entities in this cluster
                for mention in cluster.mentions:
                    mention_start = mention.start_char
                    mention_end = mention.end_char
                    
                    # Find entity matching this mention
                    matched_entity = None
                    for entity in entities:
                        entity_start = entity.get('start', 0)
                        entity_end = entity.get('end', 0)
                        
                        # Check for overlap
                        if self._spans_overlap((entity_start, entity_end), (mention_start, mention_end)):
                            matched_entity = entity
                            break
                    
                    if matched_entity:
                        entity_cluster['entities'].append(matched_entity)
                        mention_to_cluster[matched_entity['id']] = cluster_id
                        
                        # Check if this is the canonical mention
                        if str(mention) == str(canonical_mention):
                            canonical_entity = matched_entity
                
                # Set canonical entity (or first entity if no match)
                if canonical_entity:
                    entity_cluster['canonical_entity'] = canonical_entity
                elif entity_cluster['entities']:
                    entity_cluster['canonical_entity'] = entity_cluster['entities'][0]
                
                # Add to clusters if it has entities
                if entity_cluster['entities']:
                    entity_clusters.append(entity_cluster)
            
            # Create coreference links
            for cluster in entity_clusters:
                canonical_id = cluster['canonical_entity']['id']
                
                for entity in cluster['entities']:
                    if entity['id'] != canonical_id:
                        coreference_links.append({
                            'source_id': entity['id'],
                            'target_id': canonical_id,
                            'source_text': entity.get('text', ''),
                            'target_text': cluster['canonical_entity'].get('text', ''),
                            'confidence': 0.9,  # Higher confidence for neural model
                            'relation_type': 'coreference'
                        })
            
            # Update entities with coreference information
            resolved_entities = []
            for entity in entities:
                entity_id = entity.get('id')
                
                if entity_id in mention_to_cluster:
                    cluster_id = mention_to_cluster[entity_id]
                    cluster = entity_clusters[cluster_id]
                    canonical_entity = cluster['canonical_entity']
                    
                    # Create resolved entity
                    resolved_entity = entity.copy()
                    
                    # Add coreference information
                    resolved_entity['cluster_id'] = cluster_id
                    resolved_entity['canonical_id'] = canonical_entity['id']
                    
                    # If this isn't the canonical entity, add link to canonical
                    if entity_id != canonical_entity['id']:
                        resolved_entity['refers_to'] = canonical_entity['id']
                        resolved_entity['canonical_text'] = canonical_entity.get('text')
                    
                    resolved_entities.append(resolved_entity)
                else:
                    # Entity not in any cluster
                    resolved_entities.append(entity.copy())
            
            return resolved_entities, coreference_links
            
        except Exception as e:
            logging.error(f"Error in neuralcoref resolution: {e}")
            # Fall back to rule-based resolution
            return self._rule_based_resolution(text, entities)
    
    def _allennlp_resolution(self, text, entities):
        """Coreference resolution using AllenNLP"""
        try:
            # Get coreference predictions
            result = self.neural_model.predict(document=text)
            
            # Extract clusters
            clusters = result.get('clusters', [])
            
            # Get document tokens
            tokens = result.get('document', [])
            token_spans = self._get_token_spans(text, tokens)
            
            # Map token indices to text spans
            cluster_spans = []
            for cluster in clusters:
                spans = []
                for mention in cluster:
                    start_idx, end_idx = mention
                    if 0 <= start_idx < len(token_spans) and 0 <= end_idx < len(token_spans):
                        start_char = token_spans[start_idx][0]
                        end_char = token_spans[end_idx][1]
                        spans.append((start_char, end_char))
                cluster_spans.append(spans)
            
            # Map spans to entities
            entity_clusters = []
            mention_to_cluster = {}
            coreference_links = []
            
            # Process each cluster
            for cluster_id, spans in enumerate(cluster_spans):
                entity_cluster = {
                    'id': cluster_id,
                    'entities': [],
                    'canonical_entity': None
                }
                
                # Find entities in this cluster
                for span_start, span_end in spans:
                    matched_entity = None
                    for entity in entities:
                        entity_start = entity.get('start', 0)
                        entity_end = entity.get('end', 0)
                        
                        # Check for overlap
                        if self._spans_overlap((entity_start, entity_end), (span_start, span_end)):
                            matched_entity = entity
                            break
                    
                    if matched_entity and matched_entity['id'] not in mention_to_cluster:
                        entity_cluster['entities'].append(matched_entity)
                        mention_to_cluster[matched_entity['id']] = cluster_id
                
                # Set canonical entity (first mention is typically most representative)
                if entity_cluster['entities']:
                    entity_cluster['canonical_entity'] = entity_cluster['entities'][0]
                    entity_clusters.append(entity_cluster)
            
            # Create coreference links
            for cluster in entity_clusters:
                if not cluster['canonical_entity']:
                    continue
                    
                canonical_id = cluster['canonical_entity']['id']
                
                for entity in cluster['entities']:
                    if entity['id'] != canonical_id:
                        coreference_links.append({
                            'source_id': entity['id'],
                            'target_id': canonical_id,
                            'source_text': entity.get('text', ''),
                            'target_text': cluster['canonical_entity'].get('text', ''),
                            'confidence': 0.9,
                            'relation_type': 'coreference'
                        })
            
            # Update entities with coreference information
            resolved_entities = []
            for entity in entities:
                entity_id = entity.get('id')
                
                if entity_id in mention_to_cluster:
                    cluster_id = mention_to_cluster[entity_id]
                    cluster = entity_clusters[cluster_id]
                    canonical_entity = cluster['canonical_entity']
                    
                    # Create resolved entity
                    resolved_entity = entity.copy()
                    
                    # Add coreference information
                    resolved_entity['cluster_id'] = cluster_id
                    resolved_entity['canonical_id'] = canonical_entity['id']
                    
                    # If this isn't the canonical entity, add link to canonical
                    if entity_id != canonical_entity['id']:
                        resolved_entity['refers_to'] = canonical_entity['id']
                        resolved_entity['canonical_text'] = canonical_entity.get('text')
                    
                    resolved_entities.append(resolved_entity)
                else:
                    # Entity not in any cluster
                    resolved_entities.append(entity.copy())
            
            return resolved_entities, coreference_links
            
        except Exception as e:
            logging.error(f"Error in AllenNLP resolution: {e}")
            # Fall back to rule-based resolution
            return self._rule_based_resolution(text, entities)
    
    def _huggingface_resolution(self, text, entities):
        """Coreference resolution using Hugging Face model"""
        try:
            # Generate coreference resolved text
            result = self.neural_model(text)
            resolved_text = result[0]['generated_text']
            
            # Find coreference links by comparing original and resolved texts
            # This is a simplistic approach - we'd need a more sophisticated 
            # algorithm to extract the actual clusters from the resolved text
            
            # For now, fall back to rule-based as Hugging Face implementation
            # would require more complex parsing
            return self._rule_based_resolution(text, entities)
            
        except Exception as e:
            logging.error(f"Error in Hugging Face resolution: {e}")
            # Fall back to rule-based resolution
            return self._rule_based_resolution(text, entities)
    
    def _get_token_spans(self, text, tokens):
        """Get character spans for tokens"""
        spans = []
        offset = 0
        
        for token in tokens:
            # Find token in text starting from offset
            token_idx = text.find(token, offset)
            if token_idx == -1:
                # Token not found, use approximate position
                spans.append((offset, offset + len(token)))
            else:
                spans.append((token_idx, token_idx + len(token)))
                offset = token_idx + len(token)
        
        return spans
    
    def _split_into_sentences(self, text):
        """Split text into sentences"""
        # Simple sentence splitting using regex
        # More sophisticated splitting could be used
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences
    
    def _spans_overlap(self, span1, span2):
        """Check if two spans overlap"""
        return max(0, min(span1[1], span2[1]) - max(span1[0], span2[0])) > 0