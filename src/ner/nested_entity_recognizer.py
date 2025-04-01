import logging
import numpy as np
import re
from collections import defaultdict

class NestedEntityRecognizer:
    """Recognizes nested entities within larger entity spans"""
    
    def __init__(self, config):
        self.config = config
        self.max_nesting_level = config.get("max_nesting_level", 3)
        self.min_nested_length = config.get("min_nested_length", 2)  # Minimum token length to consider
        self.min_confidence = config.get("min_nested_confidence", 0.5)
        self.reuse_model = config.get("reuse_ner_model", True)
        self.external_ner_model = None
        
        # Patterns for common nested entity types
        self.nested_patterns = self._initialize_nested_patterns()
    
    def _initialize_nested_patterns(self):
        """Initialize patterns for detecting potential nested entities"""
        patterns = {
            "PERSON_in_ORGANIZATION": [
                # Person name followed by organization indicator
                r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?)'s (Company|Corporation|Organization|Association|Agency|Department|Office)",
                # Person name with title in organization
                r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?) of (the [A-Z][a-zA-Z ]+ (Company|Corporation|Organization|Association|Agency|Department|Office))"
            ],
            "PERSON_in_LOCATION": [
                # Person from location
                r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?) from ([A-Z][a-zA-Z ]+)"
            ],
            "LOCATION_in_ORGANIZATION": [
                # Organization with location name
                r"\b([A-Z][a-zA-Z]+ (University|Hospital|Airport|Station)) in ([A-Z][a-zA-Z]+)",
                # Location-based organization name
                r"(The [A-Z][a-zA-Z]+ (Company|Corporation|Organization|Association)) of ([A-Z][a-zA-Z]+)"
            ],
            "LOCATION_in_LOCATION": [
                # City in state/country
                r"\b([A-Z][a-zA-Z]+), ([A-Z]{2}|[A-Z][a-zA-Z]+)",
                # Location in larger location
                r"\b([A-Z][a-zA-Z]+) in ([A-Z][a-zA-Z]+)",
                # District of city
                r"([A-Z][a-zA-Z]+ District) of ([A-Z][a-zA-Z]+)"
            ],
            "ORGANIZATION_in_ORGANIZATION": [
                # Department of organization
                r"([A-Z][a-zA-Z]+ Department) of (the [A-Z][a-zA-Z]+ (Company|Corporation|Organization|Association))",
                # Subsidiary relationship
                r"([A-Z][a-zA-Z]+), a subsidiary of ([A-Z][a-zA-Z]+)"
            ],
            "DATE_in_EVENT": [
                # Event with date
                r"(The [A-Z][a-zA-Z]+ (Conference|Meeting|Summit|Ceremony)) on ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?, \d{4})"
            ],
            "PRODUCT_in_ORGANIZATION": [
                # Product made by organization
                r"([A-Z][a-zA-Z]+(?: \d+)?), (made|developed|produced) by ([A-Z][a-zA-Z]+)"
            ]
        }
        return patterns
    
    def set_external_model(self, model):
        """Set external NER model to use for recursive extraction"""
        self.external_ner_model = model
    
    def recognize_nested_entities(self, text, entities, ner_model=None):
        """
        Recognize nested entities within larger entity spans
        
        Args:
            text: Original text
            entities: Initial entities from first-pass NER
            ner_model: Optional NER model for recursive extraction
            
        Returns:
            Expanded list with nested entities
        """
        # Use provided model, external model, or none
        model = ner_model or self.external_ner_model
        
        # Sort entities by length (longest first) to process containers before contained
        sorted_entities = sorted(entities, key=lambda e: e.get('end', 0) - e.get('start', 0), reverse=True)
        
        # Find nested entities
        nested_entities = []
        parent_to_children = defaultdict(list)
        entity_spans = [(e.get('start', 0), e.get('end', 0)) for e in sorted_entities]
        
        # First approach: Use pattern matching
        pattern_nested = self._detect_pattern_nested_entities(text, sorted_entities)
        
        # Second approach: Rule-based detection of potential nested entities
        rule_nested = self._detect_rule_based_nested_entities(text, sorted_entities)
        
        # Third approach: Recursive NER on entity texts (if model provided)
        recursive_nested = []
        if model and self.reuse_model:
            recursive_nested = self._detect_recursive_nested_entities(text, sorted_entities, model)
        
        # Combine nested entities from all approaches
        all_nested = pattern_nested + rule_nested + recursive_nested
        
        # Deduplicate nested entities
        unique_nested = self._deduplicate_nested_entities(all_nested)
        
        # Build parent-child relationships
        for nested in unique_nested:
            nested_start = nested.get('start', 0)
            nested_end = nested.get('end', 0)
            
            # Find parent entity (smallest container)
            best_parent = None
            best_size = float('inf')
            
            for entity in sorted_entities:
                entity_start = entity.get('start', 0)
                entity_end = entity.get('end', 0)
                
                # Check if this entity contains the nested entity
                if entity_start <= nested_start and entity_end >= nested_end:
                    # Calculate container size
                    size = entity_end - entity_start
                    
                    # Find smallest container
                    if size < best_size:
                        best_size = size
                        best_parent = entity
            
            if best_parent:
                # Set parent-child relationship
                parent_id = best_parent.get('id', '')
                nested['parent_id'] = parent_id
                nested['container_type'] = best_parent.get('ner_type') or best_parent.get('type', 'UNKNOWN')
                
                # Add to parent's children
                parent_to_children[parent_id].append(nested)
                
                # Add to nested entities list
                nested_entities.append(nested)
        
        # Update original entities with children information
        for entity in sorted_entities:
            entity_id = entity.get('id', '')
            if entity_id in parent_to_children:
                entity['contains_entities'] = True
                entity['nested_count'] = len(parent_to_children[entity_id])
                entity['nested_types'] = list(set(child.get('ner_type') or child.get('type', 'UNKNOWN') 
                                               for child in parent_to_children[entity_id]))
        
        # Combine original and nested entities
        all_entities = sorted_entities + nested_entities
        
        # Reassign IDs to ensure uniqueness
        for i, entity in enumerate(all_entities):
            # Keep original ID for original entities
            if 'parent_id' in entity:
                entity['id'] = f"n{i}"
        
        return all_entities
    
    def _detect_pattern_nested_entities(self, text, entities):
        """Detect nested entities using predefined patterns"""
        nested_entities = []
        entity_id_counter = 0
        
        # Apply each pattern to the text
        for relation_type, patterns in self.nested_patterns.items():
            entity_types = relation_type.split('_in_')
            inner_type, outer_type = entity_types[0], entity_types[1]
            
            for pattern in patterns:
                # Find all matches
                for match in re.finditer(pattern, text):
                    # Extract the groups (nested entities)
                    groups = match.groups()
                    
                    if len(groups) >= 2:  # Need at least two groups
                        # First group is typically inner entity
                        inner_text = groups[0]
                        inner_start = match.start(1)
                        inner_end = match.start(1) + len(inner_text)
                        
                        # Last group is typically outer entity
                        outer_text = groups[-1]
                        outer_start = match.start(len(groups))
                        outer_end = match.start(len(groups)) + len(outer_text)
                        
                        # Create nested entity
                        nested_entity = {
                            'id': f"pn{entity_id_counter}",
                            'text': inner_text,
                            'start': inner_start,
                            'end': inner_end,
                            'ner_type': inner_type,
                            'detection_method': 'nested_pattern',
                            'confidence': 0.8,
                            'nested': True,
                            'relation_to_container': relation_type
                        }
                        
                        entity_id_counter += 1
                        nested_entities.append(nested_entity)
                        
                        # If outer entity not already in entities, add it too
                        outer_found = False
                        for entity in entities:
                            entity_start = entity.get('start', 0)
                            entity_end = entity.get('end', 0)
                            
                            if self._spans_overlap((entity_start, entity_end), (outer_start, outer_end)):
                                outer_found = True
                                break
                        
                        if not outer_found:
                            outer_entity = {
                                'id': f"pn{entity_id_counter}",
                                'text': outer_text,
                                'start': outer_start,
                                'end': outer_end,
                                'ner_type': outer_type,
                                'detection_method': 'nested_pattern',
                                'confidence': 0.7,
                                'contains_entities': True,
                                'nested_count': 1,
                                'nested_types': [inner_type]
                            }
                            
                            entity_id_counter += 1
                            nested_entities.append(outer_entity)
        
        return nested_entities
    
    def _detect_rule_based_nested_entities(self, text, entities):
        """Detect nested entities using rule-based approaches"""
        nested_entities = []
        entity_id_counter = 0
        
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('ner_type') or entity.get('type', 'UNKNOWN')
            entity_start = entity.get('start', 0)
            
            # Skip short entities
            if len(entity_text.split()) < 2:
                continue
                
            # Check entity type for specific nested entity rules
            if entity_type == 'ORGANIZATION':
                # Extract person names from organization names
                # E.g., "Johnson & Johnson" -> "Johnson"
                name_pattern = r'\b([A-Z][a-z]+)\b(?:\s+&\s+\1\b)?'
                for match in re.finditer(name_pattern, entity_text):
                    if match.group(0) != entity_text:  # Not the whole entity
                        name = match.group(1)
                        name_start = entity_start + match.start(1)
                        name_end = name_start + len(name)
                        
                        nested_entities.append({
                            'id': f"rn{entity_id_counter}",
                            'text': name,
                            'start': name_start,
                            'end': name_end,
                            'ner_type': 'PERSON',
                            'detection_method': 'nested_rule',
                            'confidence': 0.7,
                            'nested': True,
                            'relation_to_container': 'PERSON_in_ORGANIZATION'
                        })
                        entity_id_counter += 1
                
                # Extract location names from organization names
                # E.g., "University of California" -> "California"
                loc_pattern = r'\b(?:University|College|Institute|Association|Bank) of ([A-Z][a-zA-Z\s]+)\b'
                for match in re.finditer(loc_pattern, entity_text):
                    location = match.group(1)
                    loc_start = entity_start + match.start(1)
                    loc_end = loc_start + len(location)
                    
                    nested_entities.append({
                        'id': f"rn{entity_id_counter}",
                        'text': location,
                        'start': loc_start,
                        'end': loc_end,
                        'ner_type': 'LOCATION',
                        'detection_method': 'nested_rule',
                        'confidence': 0.7,
                        'nested': True,
                        'relation_to_container': 'LOCATION_in_ORGANIZATION'
                    })
                    entity_id_counter += 1
            
            elif entity_type == 'LOCATION':
                # Extract nested locations
                # E.g., "New York City, New York" -> "New York City" and "New York"
                loc_pattern = r'\b([A-Z][a-zA-Z\s]+), ([A-Z][a-zA-Z\s]+)\b'
                for match in re.finditer(loc_pattern, entity_text):
                    city = match.group(1)
                    state = match.group(2)
                    
                    city_start = entity_start + match.start(1)
                    city_end = city_start + len(city)
                    
                    state_start = entity_start + match.start(2)
                    state_end = state_start + len(state)
                    
                    # Add city as nested entity
                    nested_entities.append({
                        'id': f"rn{entity_id_counter}",
                        'text': city,
                        'start': city_start,
                        'end': city_end,
                        'ner_type': 'LOCATION',
                        'detection_method': 'nested_rule',
                        'confidence': 0.8,
                        'nested': True,
                        'relation_to_container': 'LOCATION_in_LOCATION'
                    })
                    entity_id_counter += 1
                    
                    # Add state as nested entity
                    nested_entities.append({
                        'id': f"rn{entity_id_counter}",
                        'text': state,
                        'start': state_start,
                        'end': state_end,
                        'ner_type': 'LOCATION',
                        'detection_method': 'nested_rule',
                        'confidence': 0.8,
                        'nested': True,
                        'relation_to_container': 'LOCATION_in_LOCATION'
                    })
                    entity_id_counter += 1
            
            elif entity_type == 'PERSON':
                # Extract title and name from person entities
                # E.g., "Dr. Jane Smith" -> "Dr." and "Jane Smith"
                title_pattern = r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|President|CEO|Director)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
                for match in re.finditer(title_pattern, entity_text):
                    title = match.group(1)
                    name = match.group(2)
                    
                    title_start = entity_start + match.start(1)
                    title_end = title_start + len(title)
                    
                    name_start = entity_start + match.start(2)
                    name_end = name_start + len(name)
                    
                    # Add title as nested entity
                    nested_entities.append({
                        'id': f"rn{entity_id_counter}",
                        'text': title,
                        'start': title_start,
                        'end': title_end,
                        'ner_type': 'TITLE',
                        'detection_method': 'nested_rule',
                        'confidence': 0.8,
                        'nested': True,
                        'relation_to_container': 'TITLE_in_PERSON'
                    })
                    entity_id_counter += 1
                    
                    # Add name as nested entity if different from whole entity
                    if name != entity_text:
                        nested_entities.append({
                            'id': f"rn{entity_id_counter}",
                            'text': name,
                            'start': name_start,
                            'end': name_end,
                            'ner_type': 'PERSON',
                            'detection_method': 'nested_rule',
                            'confidence': 0.9,
                            'nested': True,
                            'relation_to_container': 'PERSON_in_PERSON'
                        })
                        entity_id_counter += 1
        
        return nested_entities
    
    def _detect_recursive_nested_entities(self, text, entities, model, level=0):
        """Recursively detect nested entities using NER model"""
        if level >= self.max_nesting_level:
            return []
            
        nested_entities = []
        entity_id_counter = 0
        
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_start = entity.get('start', 0)
            
            # Skip short entities
            if len(entity_text.split()) < self.min_nested_length:
                continue
                
            # Run NER on entity text
            try:
                inner_entities = model.extract_entities(entity_text)
                
                for inner_entity in inner_entities:
                    inner_text = inner_entity.get('text', '')
                    inner_start_relative = inner_entity.get('start', 0)
                    inner_end_relative = inner_entity.get('end', 0)
                    
                    # Calculate absolute position in original text
                    inner_start_absolute = entity_start + inner_start_relative
                    inner_end_absolute = entity_start + inner_end_relative
                    
                    # Skip if inner entity is the same as outer entity
                    if inner_text == entity_text:
                        continue
                    
                    # Create nested entity
                    inner_entity_copy = inner_entity.copy()
                    inner_entity_copy.update({
                        'id': f"en{entity_id_counter}",
                        'start': inner_start_absolute,
                        'end': inner_end_absolute,
                        'nested': True,
                        'detection_method': f"nested_recursive_level{level+1}"
                    })
                    
                    entity_id_counter += 1
                    nested_entities.append(inner_entity_copy)
                
                # Recursively find nested entities within nested entities
                if level < self.max_nesting_level - 1:
                    recursive_nested = self._detect_recursive_nested_entities(
                        entity_text, inner_entities, model, level + 1
                    )
                    
                    # Adjust positions to original text
                    for rec_entity in recursive_nested:
                        rec_entity['start'] = entity_start + rec_entity.get('start', 0)
                        rec_entity['end'] = entity_start + rec_entity.get('end', 0)
                        rec_entity['id'] = f"en{entity_id_counter}"
                        entity_id_counter += 1
                    
                    nested_entities.extend(recursive_nested)
                
            except Exception as e:
                logging.error(f"Error in recursive nested entity detection: {e}")
        
        return nested_entities
    
    def _deduplicate_nested_entities(self, nested_entities):
        """Deduplicate nested entities"""
        unique_entities = []
        seen_spans = set()
        
        for entity in nested_entities:
            entity_start = entity.get('start', 0)
            entity_end = entity.get('end', 0)
            entity_type = entity.get('ner_type') or entity.get('type', 'UNKNOWN')
            
            span_key = (entity_start, entity_end, entity_type)
            
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _spans_overlap(self, span1, span2):
        """Check if two spans overlap"""
        return max(0, min(span1[1], span2[1]) - max(span1[0], span2[0])) > 0