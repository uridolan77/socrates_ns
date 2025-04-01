import logging
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from transformers import logging as hf_logging

class DomainSpecificNERModel:
    """Domain-specific NER using specialized pre-trained transformer models"""
    
    # Default domain model mappings
    DOMAIN_MODELS = {
        "healthcare": "bvanaken/clinical-assertion-negation-bert",
        "biomedical": "drAbreu/bioBERT-NER-NCBI-disease",
        "finance": "ProsusAI/finbert",
        "legal": "nlpaueb/legal-bert-base-uncased",
        "scientific": "allenai/scibert_scivocab_uncased",
        "social_media": "vinai/bertweet-base"
    }
    
    # Domain-specific entity type mappings
    DOMAIN_ENTITY_TYPES = {
        "healthcare": {
            "DISEASE": "CONDITION",
            "DRUG": "MEDICATION",
            "PROCEDURE": "PROCEDURE",
            "TEST": "DIAGNOSTIC"
        },
        "finance": {
            "ORG": "ORGANIZATION",
            "MONEY": "CURRENCY",
            "PERCENT": "PERCENTAGE",
            "GPE": "LOCATION"
        }
    }
    
    def __init__(self, config):
        self.config = config
        self.domain = config.get("domain", "general")
        self.custom_model_path = config.get("domain_model_path")
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
    
    def initialize(self, domains=None):
        """Initialize domain-specific models"""
        if domains is None:
            # If domain is specified in config, use that
            if self.domain != "general":
                domains = [self.domain]
            else:
                # Otherwise initialize all available domains
                domains = list(self.DOMAIN_MODELS.keys())
        
        for domain in domains:
            try:
                # Check if custom model path is provided for this domain
                model_path = self.config.get(f"{domain}_model_path", self.DOMAIN_MODELS.get(domain))
                
                if not model_path:
                    logging.warning(f"No model specified for domain: {domain}")
                    continue
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                model.to(self.device)
                
                # Create NER pipeline for this domain
                ner_pipeline = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=0 if self.device == "cuda" else -1
                )
                
                self.models[domain] = {
                    "pipeline": ner_pipeline,
                    "entity_types": self.DOMAIN_ENTITY_TYPES.get(domain, {})
                }
                
                logging.info(f"Initialized domain-specific model for {domain}")
            except Exception as e:
                logging.error(f"Failed to initialize model for domain {domain}: {e}")
    
    def extract_entities(self, text, domain=None):
        """Extract entities using domain-specific model"""
        # Use specified domain or default from config
        domain = domain or self.domain
        
        # If general domain or domain not available, return empty list
        if domain == "general" or domain not in self.models:
            if domain not in self.models:
                # Try to initialize the requested domain
                self.initialize([domain])
                if domain not in self.models:
                    return []
        
        try:
            # Get domain-specific pipeline
            ner_pipeline = self.models[domain]["pipeline"]
            entity_type_mapping = self.models[domain]["entity_types"]
            
            # Extract entities
            entities = ner_pipeline(text)
            
            # Convert to standard format with domain-specific entity types
            standardized_entities = []
            for i, entity in enumerate(entities):
                # Map domain-specific entity types if defined
                ner_type = entity['entity_group']
                if ner_type in entity_type_mapping:
                    ner_type = entity_type_mapping[ner_type]
                
                standardized_entities.append({
                    'id': f"d{i}",
                    'text': entity['word'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'ner_type': ner_type,
                    'domain': domain,
                    'score': entity['score'],
                    'detection_method': f"domain_model_{domain}"
                })
            
            return standardized_entities
        except Exception as e:
            logging.error(f"Error in domain-specific NER extraction for {domain}: {e}")
            return []
    
    def get_available_domains(self):
        """Get list of available domain models"""
        return list(self.models.keys())
    
    def add_custom_domain(self, domain_name, model_path, entity_mapping=None):
        """Add a custom domain model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            model.to(self.device)
            
            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            
            self.models[domain_name] = {
                "pipeline": ner_pipeline,
                "entity_types": entity_mapping or {}
            }
            
            # Update the domain models dictionary
            self.DOMAIN_MODELS[domain_name] = model_path
            if entity_mapping:
                self.DOMAIN_ENTITY_TYPES[domain_name] = entity_mapping
                
            logging.info(f"Added custom domain model for {domain_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to add custom domain model {domain_name}: {e}")
            return False