import logging
import re
import numpy as np
import requests
import json

class PromptBasedNER:
    """Implements prompt-based NER using large language models"""
    
    # Example NER prompts for few-shot learning
    DEFAULT_PROMPTS = {
        "zero_shot": """
            Extract all named entities from the following text. 
            For each entity, provide the entity text, its type (PERSON, ORGANIZATION, LOCATION, DATE, etc.), and character positions.
            Text: {text}
        """,
        "few_shot": """
            Extract named entities from text, following these examples:
            
            Example 1:
            Text: "Apple is planning to open a new store in New York next month."
            Entities: 
            - "Apple": ORGANIZATION (0-5)
            - "New York": LOCATION (35-44)
            - "next month": DATE (45-55)
            
            Example 2:
            Text: "Tim Cook announced that the company's revenue increased by 5% in Q3."
            Entities:
            - "Tim Cook": PERSON (0-8)
            - "company": ORGANIZATION (24-31)
            - "5%": PERCENT (53-55)
            - "Q3": DATE (59-61)
            
            Now extract entities from this text:
            {text}
        """,
        "domain_specific": {
            "healthcare": """
                Extract medical entities from the following clinical text.
                Entity types include: DISEASE, MEDICATION, PROCEDURE, ANATOMICAL_SITE, LAB_TEST, DOSAGE.
                
                Example:
                Text: "Patient presents with acute sinusitis, prescribed Amoxicillin 500mg three times daily."
                Entities:
                - "acute sinusitis": DISEASE (24-39)
                - "Amoxicillin": MEDICATION (51-62)
                - "500mg": DOSAGE (63-68)
                
                Now extract medical entities from this text:
                {text}
            """,
            "finance": """
                Extract financial entities from the following text.
                Entity types include: COMPANY, AMOUNT, PERCENTAGE, INDEX, CURRENCY, FINANCIAL_METRIC.
                
                Example:
                Text: "Microsoft reported Q2 revenue of $52.7 billion, up 2% year-over-year."
                Entities:
                - "Microsoft": COMPANY (0-9)
                - "Q2": DATE (19-21)
                - "$52.7 billion": AMOUNT (33-45)
                - "2%": PERCENTAGE (50-52)
                
                Now extract financial entities from this text:
                {text}
            """
        },
        "custom": "{custom_prompt}"
    }
    
    def __init__(self, config):
        self.config = config
        self.llm_provider = config.get("llm_provider", "openai")
        self.api_key = config.get("llm_api_key")
        self.model = config.get("llm_model", "gpt-4")
        self.prompt_type = config.get("prompt_type", "few_shot")
        self.custom_prompt = config.get("custom_prompt", "")
        self.custom_examples = config.get("custom_examples", [])
        self.endpoint = config.get("llm_endpoint", self._get_default_endpoint())
        self.temperature = config.get("llm_temperature", 0.0)
        self.max_tokens = config.get("llm_max_tokens", 1000)
        
    def _get_default_endpoint(self):
        """Get default API endpoint based on provider"""
        if self.llm_provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        elif self.llm_provider == "anthropic":
            return "https://api.anthropic.com/v1/messages"
        # Add other providers as needed
        return None
        
    def extract_entities(self, text, domain=None, prompt_type=None):
        """Extract entities using prompt-based approach"""
        # Use specified prompt type or default from config
        prompt_type = prompt_type or self.prompt_type
        
        # Generate prompt based on type
        prompt = self._generate_prompt(text, domain, prompt_type)
        
        # Get LLM response
        response = self._get_llm_response(prompt)
        
        # Parse entities from response
        entities = self._parse_entities_from_response(response, text)
        
        return entities
    
    def _generate_prompt(self, text, domain=None, prompt_type=None):
        """Generate appropriate prompt based on type and domain"""
        # Handle domain-specific prompts
        if domain and prompt_type == "domain_specific":
            if domain in self.DEFAULT_PROMPTS["domain_specific"]:
                prompt_template = self.DEFAULT_PROMPTS["domain_specific"][domain]
            else:
                # Fall back to few-shot if domain not found
                prompt_template = self.DEFAULT_PROMPTS["few_shot"]
        elif prompt_type == "custom" and self.custom_prompt:
            # Use custom prompt
            prompt_template = self.DEFAULT_PROMPTS["custom"].format(custom_prompt=self.custom_prompt)
        else:
            # Use standard prompt types
            prompt_template = self.DEFAULT_PROMPTS.get(prompt_type, self.DEFAULT_PROMPTS["few_shot"])
        
        # Insert custom examples if available and not using custom prompt
        if self.custom_examples and prompt_type != "custom" and prompt_type != "zero_shot":
            examples_text = "\n\n".join([f"Example {i+3}:\n{example}" for i, example in enumerate(self.custom_examples)])
            # Insert before the "Now extract..." part
            parts = prompt_template.split("Now extract")
            if len(parts) > 1:
                prompt_template = f"{parts[0]}\n{examples_text}\n\nNow extract{parts[1]}"
        
        # Format prompt with text
        return prompt_template.format(text=text)
    
    def _get_llm_response(self, prompt):
        """Get response from LLM API"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Set up provider-specific headers and payload
            if self.llm_provider == "openai":
                headers["Authorization"] = f"Bearer {self.api_key}"
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            elif self.llm_provider == "anthropic":
                headers["x-api-key"] = self.api_key
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            else:
                logging.error(f"Unsupported LLM provider: {self.llm_provider}")
                return ""
            
            # Make API request
            response = requests.post(
                self.endpoint,
                headers=headers,
                data=json.dumps(data)
            )
            
            # Parse response based on provider
            if response.status_code == 200:
                response_json = response.json()
                if self.llm_provider == "openai":
                    return response_json["choices"][0]["message"]["content"]
                elif self.llm_provider == "anthropic":
                    return response_json["content"][0]["text"]
                else:
                    return ""
            else:
                logging.error(f"API request failed with status {response.status_code}: {response.text}")
                return ""
                
        except Exception as e:
            logging.error(f"Error getting LLM response: {e}")
            return ""
    
    def _parse_entities_from_response(self, response, original_text):
        """Parse entities from LLM response"""
        entities = []
        
        # Try different parsing approaches
        
        # Pattern 1: "- "Entity": TYPE (start-end)"
        pattern1 = r'-\s*"([^"]+)":\s*([A-Z_]+)\s*\((\d+)-(\d+)\)'
        matches1 = re.findall(pattern1, response)
        
        # Pattern 2: "- Entity: TYPE (start-end)"
        pattern2 = r'-\s*([^:]+):\s*([A-Z_]+)\s*\((\d+)-(\d+)\)'
        matches2 = re.findall(pattern2, response)
        
        # Combine matches
        all_matches = matches1 + [m for m in matches2 if m not in matches1]
        
        # Process matches
        for i, match in enumerate(all_matches):
            entity_text, entity_type, start_str, end_str = match
            
            try:
                start = int(start_str)
                end = int(end_str)
                
                # Validate entity boundaries
                if 0 <= start < end <= len(original_text):
                    # Verify entity text matches original text at positions
                    expected_text = original_text[start:end]
                    if entity_text.strip() == expected_text.strip() or entity_text in expected_text:
                        entities.append({
                            'id': f"llm{i}",
                            'text': entity_text,
                            'start': start,
                            'end': end,
                            'ner_type': entity_type,
                            'detection_method': 'prompt_based_llm',
                            'confidence': 0.9  # Default confidence for LLM-based extraction
                        })
                    else:
                        # Text doesn't match, try to find it
                        actual_start = original_text.find(entity_text)
                        if actual_start >= 0:
                            entities.append({
                                'id': f"llm{i}",
                                'text': entity_text,
                                'start': actual_start,
                                'end': actual_start + len(entity_text),
                                'ner_type': entity_type,
                                'detection_method': 'prompt_based_llm',
                                'confidence': 0.7  # Lower confidence for adjusted positions
                            })
            except ValueError:
                # Invalid position format
                continue
        
        return entities
    
    def add_custom_example(self, example_text):
        """Add a custom example for few-shot learning"""
        self.custom_examples.append(example_text)
        
    def set_custom_prompt(self, prompt):
        """Set a custom prompt template"""
        self.custom_prompt = prompt
        self.prompt_type = "custom"