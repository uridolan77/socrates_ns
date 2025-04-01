import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np
import logging
from typing import List, Dict, Any, Optional

class TransformerNERModel:
    """Enhanced NER using modern transformer models like BERT, RoBERTa, DeBERTa"""
    
    def __init__(self, config):
        self.config = config
        self.model_name = config.get("transformer_model", "dslim/bert-base-NER")
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
        
    def initialize(self):
        """Initialize the transformer model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Create NER pipeline
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",  # Merge subwords into words
                device=0 if self.device == "cuda" else -1
            )
            return True
        except Exception as e:
            logging.error(f"Failed to initialize transformer model {self.model_name}: {e}")
            return False
    
    def extract_entities(self, text):
        """Extract entities using the transformer model"""
        if not self.ner_pipeline:
            if not self.initialize():
                return []
        
        try:
            # Process text using the NER pipeline
            entities = self.ner_pipeline(text)
            
            # Convert to standard format
            standardized_entities = []
            for i, entity in enumerate(entities):
                standardized_entities.append({
                    'id': f"t{i}",
                    'text': entity['word'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'ner_type': entity['entity_group'],
                    'score': entity['score'],
                    'detection_method': f"transformer_{self.model_name.split('/')[-1]}"
                })
            
            return standardized_entities
        except Exception as e:
            logging.error(f"Error in transformer NER extraction: {e}")
            return []
    
    def extract_entities_batch(self, texts, batch_size=8):
        """Extract entities from multiple texts in batches"""
        if not self.ner_pipeline:
            if not self.initialize():
                return [[] for _ in texts]
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                batch_entities = self.ner_pipeline(batch)
                
                # Handle single text vs batch results
                if len(batch) == 1:
                    batch_entities = [batch_entities]
                
                for text_entities in batch_entities:
                    standardized_entities = []
                    for j, entity in enumerate(text_entities):
                        standardized_entities.append({
                            'id': f"t{j}",
                            'text': entity['word'],
                            'start': entity['start'],
                            'end': entity['end'],
                            'ner_type': entity['entity_group'],
                            'score': entity['score'],
                            'detection_method': f"transformer_{self.model_name.split('/')[-1]}"
                        })
                    results.append(standardized_entities)
            except Exception as e:
                logging.error(f"Error in batch transformer NER extraction: {e}")
                results.extend([[] for _ in batch])
        
        return results