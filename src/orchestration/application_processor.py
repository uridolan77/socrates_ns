import datetime
import logging
import uuid
import traceback
from dataclasses import dataclass, field
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import src.compliance.verification.verifier as ComplianceVerifier
from src.utils.text.token_tracker import TokenConfidenceTracker, SensitiveTokenDetector
from src.symbolic.rules import RulePerformanceTracker, SemanticRuleAdaptationEngine
from src.neuro.embeddings import RegulatoryEmbeddingSpace, TokenLevelComplianceGate
from src.compliance.verification.verifier import ComplianceVerifier
from src.compliance.verification.entity_checker import EntityComplianceChecker
import src.compliance.verification.semantic_compliance_checker as SemanticComplianceChecker
from src.utils.cache.lru_cache import LRUCache
from src.orchestration.regulatory_retrieval import OptimizedRegulatoryRetrieval
from src.compliance.proofs import ComplianceProofNode, ProofTraceNodeType, ProofNodeStatus
from src.compliance.utils.model_registry import ComplianceModelRegistry
from src.orchestration.parallel_prefilter import ParallelCompliancePrefilter
from src.interface.optimized_token_gate import OptimizedTokenLevelComplianceGate
from src.compliance.proofs.proof_tracer import ComplianceProofTracer
from src.orchestration.regulation_augmenter import EfficientRegulationAugmenter
from src.utils.performance.optimizer import ComplianceSystemPerformanceOptimizer
from src.compliance.models.regulatory_models import RegulatoryDocumentStore, RegulatoryFramework, RegulatoryConceptDefinition
from src.compliance.monitoring.optimized_semantic_compliance_monitor import SemanticComplianceMonitor
from src.neuro.embeddings.domain_specific_space import RegulatoryEmbeddingSpace
from src.neuro.embeddings.quantized_space import QuantizedRegulatoryEmbeddingSpace
from src.compliance.verification.violation_analyzer import ViolationAnalyzer
from src.compliance.verification.constraint_optimizer import ComplianceConstraintOptimizer
from src.interface.optimized_interface import OptimizedNeuralSymbolicInterface
from src.symbolic.rules.rule_performance_tracker import RulePerformanceTracker

class CompliantLanguageModelProcessor:
    """
    Compliant Language Model Processor (CLMP) that extends the Neural Processing Module
    to incorporate language models with integrated regulatory compliance enforcement.
    
    This component enforces compliance at three critical stages:
    1. Pre-generation: Analyzing prompts and determining applicable constraints
    2. In-generation: Filtering token probabilities during text generation
    3. Post-generation: Verifying the complete generated text
    """
    def __init__(self, language_model, neural_symbolic_interface, regulatory_knowledge_base, compliance_config):
        """
        Initialize the CLMP with required components.
        
        Args:
            language_model: The base language model
            neural_symbolic_interface: Interface for bidirectional translation
            regulatory_knowledge_base: Repository of regulatory frameworks
            compliance_config: Configuration parameters for compliance enforcement
        """
        self.language_model = language_model
        self.interface = neural_symbolic_interface
        self.regulatory_kb = regulatory_knowledge_base
        
        # Enhanced components with improved capabilities
        self.compliance_verifier = ComplianceVerifier(compliance_config)
        
        # Initialize specialized compliance components with advanced capabilities
        self.token_gate = TokenLevelComplianceGate(compliance_config)
        self.semantic_monitor = SemanticComplianceMonitor(compliance_config)
        self.regulatory_embedding = RegulatoryEmbeddingSpace(
            language_model.embedding_dim, 
            regulatory_knowledge_base
        )
        
        # Add performance monitoring and caching
        self.compliance_cache = LRUCache(maxsize=compliance_config.get("cache_size", 1000))
        self.performance_metrics = RulePerformanceTracker()
        
        # Add additional components for advanced functionality
        self.violation_analyzer = ViolationAnalyzer(compliance_config)
        self.constraint_optimizer = ComplianceConstraintOptimizer(compliance_config)

    def generate_compliant_text(self, prompt, context=None, max_tokens=100, compliance_mode='strict'):
        """
        Generate text that complies with applicable regulatory frameworks.
        
        Args:
            prompt: Input text to initiate generation
            context: Additional context information
            max_tokens: Maximum number of tokens to generate
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Dict containing generated text, compliance information, and explanations
        """
        # STAGE 1: Pre-generation compliance analysis
        pre_generation_result = self._perform_pre_generation_compliance(prompt, context, compliance_mode)
        
        if not pre_generation_result['is_compliant']:
            return {
                'text': None,
                'compliance_error': pre_generation_result['error'],
                'compliance_metadata': pre_generation_result['metadata']
            }
        
        # Extract applicable frameworks and constraints
        applicable_frameworks = pre_generation_result['applicable_frameworks']
        applicable_constraints = pre_generation_result['applicable_constraints']
        
        # Initialize generation
        generated_text = ""
        input_ids = self.tokenize(prompt)
        
        # Initialize semantic compliance state
        semantic_state = self.semantic_monitor.initialize(prompt, applicable_frameworks)
        
        # STAGE 2: In-generation compliance enforcement with token-level filtering
        for i in range(max_tokens):
            # Get next token logits from language model
            logits = self.language_model.get_next_token_logits(input_ids)
            
            # Apply token-level compliance gate to filter logits
            filtered_logits = self.token_gate.filter(
                logits,
                generated_text,
                semantic_state,
                applicable_constraints,
                compliance_mode
            )
            
            # Sample next token from filtered distribution
            next_token = self.sample_token(filtered_logits)
            if self.is_eos_token(next_token):
                break
                
            # Update generated text and input sequence
            token_text = self.decode_token(next_token)
            generated_text += token_text
            input_ids.append(next_token)
            
            # Update semantic compliance state
            semantic_state = self.semantic_monitor.update(
                semantic_state,
                token_text,
                generated_text,
                applicable_frameworks
            )
        
        # STAGE 3: Post-generation compliance verification
        verification_result = self._perform_post_generation_verification(
            generated_text,
            applicable_frameworks,
            compliance_mode
        )
        
        # Generate explanation if needed
        explanation = None
        if not verification_result['is_compliant'] or verification_result.get('has_modifications', False):
            explanation = self._generate_compliance_explanation(
                verification_result,
                prompt,
                generated_text
            )
        
        return {
            'text': generated_text,
            'is_compliant': verification_result['is_compliant'],
            'compliance_score': verification_result.get('compliance_score', 0.0),
            'modified': verification_result.get('has_modifications', False),
            'explanation': explanation,
            'compliance_metadata': verification_result.get('metadata', {})
        }
    
    def _perform_pre_generation_compliance(self, prompt, context, compliance_mode):
        """
        Analyze prompt and context to determine applicable regulatory frameworks
        and potential compliance issues before generation begins.
        """
        # Verify prompt compliance
        prompt_compliance = self.compliance_verifier.verify_content(
            prompt, 
            content_type="prompt",
            compliance_mode=compliance_mode
        )
        
        if not prompt_compliance['is_compliant']:
            return prompt_compliance
            
        # Get applicable regulatory frameworks
        applicable_frameworks = self.regulatory_kb.get_applicable_frameworks(
            prompt, 
            context, 
            compliance_mode
        )
        
        # Extract concrete compliance requirements
        applicable_constraints = []
        for framework in applicable_frameworks:
            framework_constraints = framework.get_requirements(context)
            applicable_constraints.extend(framework_constraints)
            
        # Resolve conflicts between constraints
        resolved_constraints = self.regulatory_kb.resolve_conflicts(applicable_constraints)
        
        # Project prompt to regulatory embedding space for efficient compliance checking
        prompt_embedding = self.language_model.get_embeddings(prompt)
        regulatory_embedding = self.regulatory_embedding.project_to_regulatory_space(prompt_embedding)
        
        # Compute initial compliance scores
        compliance_scores = self.regulatory_embedding.compute_compliance_scores(
            regulatory_embedding,
            applicable_frameworks
        )
        
        # Check if prompt is in high-risk category requiring special handling
        is_high_risk = self._check_high_risk_category(prompt, compliance_scores)
        
        return {
            'is_compliant': True,
            'applicable_frameworks': applicable_frameworks,
            'applicable_constraints': resolved_constraints,
            'regulatory_embedding': regulatory_embedding,
            'compliance_scores': compliance_scores,
            'is_high_risk': is_high_risk,
            'metadata': {
                'framework_count': len(applicable_frameworks),
                'constraint_count': len(resolved_constraints)
            }
        }
        
    def _perform_post_generation_verification(self, text, frameworks, compliance_mode):
        """
        Verify the complete generated text against all applicable regulatory frameworks.
        """
        # Convert text to neural representation
        text_embeddings = self.language_model.get_embeddings(text)
        
        # Project to regulatory embedding space
        regulatory_embedding = self.regulatory_embedding.project_to_regulatory_space(text_embeddings)
        
        # Translate to symbolic representation for verification
        symbolic_repr = self.interface.neural_to_symbolic.translate(
            text,
            compliance_mode
        )
        
        # Verify compliance for each framework
        framework_results = []
        for framework in frameworks:
            result = framework.verify_compliance(symbolic_repr, compliance_mode)
            framework_results.append(result)
        
        # Aggregate framework verification results
        aggregated_result = self.compliance_verifier.aggregate_framework_results(framework_results)
        
        # Add embedding-based compliance scores
        compliance_scores = self.regulatory_embedding.compute_compliance_scores(
            regulatory_embedding,
            frameworks
        )
        
        aggregated_result['embedding_compliance_scores'] = compliance_scores
        
        # Calculate overall compliance score (combining symbolic and embedding-based)
        overall_score = self._calculate_overall_compliance_score(
            aggregated_result['compliance_score'],
            compliance_scores
        )
        
        aggregated_result['overall_compliance_score'] = overall_score
        
        return aggregated_result
        
    def _generate_compliance_explanation(self, verification_result, prompt, generated_text):
        """
        Generate a human-readable explanation of compliance verification results.
        """
        # Convert compliance result to symbolic representation
        symbolic_explanation = self.interface.compliance_to_symbolic.translate(
            verification_result
        )
        
        # Generate natural language explanation using symbolic-to-neural translation
        natural_language_explanation = self.interface.symbolic_to_neural.translate_explanation(
            symbolic_explanation
        )
        
        return {
            'text': natural_language_explanation,
            'violations': verification_result.get('violations', []),
            'framework_details': verification_result.get('framework_details', {})
        }
    
    def _check_high_risk_category(self, text, compliance_scores):
        """
        Determine if text falls into a high-risk category requiring special handling.
        """
        # Implementation would check for specific risk patterns across frameworks
        # This is a simplified placeholder
        high_risk_threshold = 0.7
        
        for framework_id, score in compliance_scores.items():
            if score < high_risk_threshold:
                return True
                
        return False
    
    def _calculate_overall_compliance_score(self, symbolic_score, embedding_scores):
        """
        Calculate overall compliance score by combining symbolic and embedding-based scores.
        """
        # Average the embedding scores across frameworks
        avg_embedding_score = sum(embedding_scores.values()) / len(embedding_scores)
        
        # Combine symbolic and embedding scores (weighted average)
        symbolic_weight = 0.7  # Higher weight for symbolic verification
        embedding_weight = 0.3
        
        overall_score = (symbolic_weight * symbolic_score) + (embedding_weight * avg_embedding_score)
        
        return overall_score
    
    # Helper methods for token handling
    def tokenize(self, text):
        """Tokenize text using the language model's tokenizer."""
        return self.language_model.tokenize(text)
    
    def decode_token(self, token_id):
        """Decode a token ID to its string representation."""
        return self.language_model.decode_token(token_id)
    
    def is_eos_token(self, token_id):
        """Check if a token ID represents an end-of-sequence token."""
        return token_id == self.language_model.eos_token_id
    
    def sample_token(self, logits):
        """Sample a token from logits using appropriate sampling strategy."""
        return self.language_model.sample_token(logits)


   

    """
    Enhanced semantic rules with dynamic rule generation and adaptation based on
    regulatory changes and feedback.
    """
    def __init__(self, config):
        self.config = config
        self.rule_templates = self._initialize_rule_templates()
        self.static_rules = self._initialize_static_rules()
        self.dynamic_rules = {}
        self.rule_stats = RulePerformanceTracker()
        self.rule_adaptation_engine = SemanticRuleAdaptationEngine(config)
        
    def _initialize_rule_templates(self):
        """Initialize rule templates for semantic rules"""
        # Core templates from the base implementation
        templates = {
            'concept_threshold': {
                'type': 'semantic',
                'subtype': 'concept_threshold',
                'action': 'block_if_exceeds',
                'concept_template': "{concept}",
                'threshold': 0.7,
                'severity': 'medium'
            },
            'topic_restriction': {
                'type': 'semantic',
                'subtype': 'topic_restriction',
                'action': 'block_if_matches',
                'topic_template': "{topic}",
                'severity': 'high'
            },
            'sentiment_requirement': {
                'type': 'semantic',
                'subtype': 'sentiment_requirement',
                'action': 'require_sentiment',
                'sentiment_template': "{sentiment}",
                'threshold': 0.5,
                'severity': 'low'
            },
            'content_classification': {
                'type': 'semantic',
                'subtype': 'content_classification',
                'action': 'classify_and_restrict',
                'class_template': "{class}",
                'severity': 'medium'
            }
        }
        
        # Add advanced templates for more sophisticated rules
        advanced_templates = {
            'concept_combination': {
                'type': 'semantic',
                'subtype': 'concept_combination',
                'action': 'flag_if_combined',
                'concepts': [],
                'threshold': 0.6,
                'combination_method': 'all',  # 'all', 'any', 'weighted'
                'severity': 'medium',
                'description': 'Flags when multiple concepts appear together'
            },
            'concept_contrast': {
                'type': 'semantic',
                'subtype': 'concept_contrast',
                'action': 'flag_if_contrasted',
                'primary_concept': '',
                'contrasting_concepts': [],
                'threshold': 0.6,
                'severity': 'medium',
                'description': 'Flags when a concept appears without necessary contrasting concepts'
            },
            'semantic_context': {
                'type': 'semantic',
                'subtype': 'semantic_context',
                'action': 'verify_context',
                'target_concept': '',
                'required_context': [],
                'threshold': 0.5,
                'severity': 'medium',
                'description': 'Verifies that concepts appear in appropriate semantic contexts'
            },
            'regulatory_concept_accuracy': {
                'type': 'semantic',
                'subtype': 'regulatory_concept_accuracy',
                'action': 'verify_accuracy',
                'regulatory_concept': '',
                'framework_id': '',
                'accuracy_threshold': 0.8,
                'severity': 'high',
                'description': 'Verifies accurate representation of regulatory concepts'
            }
        }
        
        # Merge templates
        templates.update(advanced_templates)
        
        # Add custom templates from config
        custom_templates = self.config.get('custom_semantic_templates', {})
        for name, template in custom_templates.items():
            templates[name] = template
            
        return templates
    
    def _initialize_static_rules(self):
        """Initialize static rules from configuration or default set"""
        # Load static rules from configuration
        config_rules = self.config.get('semantic_rules', {})
        
        # Return rules, either from config or empty dict if none provided
        return config_rules
    
    def get_rules_for_framework(self, framework_id):
        """
        Get semantic rules for a specific regulatory framework
        
        Args:
            framework_id: ID of the regulatory framework
            
        Returns:
            List of semantic rules for the framework
        """
        # Get static rules for this framework
        static_framework_rules = self._get_static_rules_for_framework(framework_id)
        
        # Get dynamic rules for this framework
        dynamic_framework_rules = self._get_dynamic_rules_for_framework(framework_id)
        
        # Generate rules based on regulation text if available
        generated_rules = self._generate_rules_from_regulation(framework_id)
        
        # Combine rules with dynamic rules taking precedence over static if IDs clash
        combined_rules = self._combine_rules(
            static_framework_rules, dynamic_framework_rules, generated_rules
        )
        
        # Update rule stats
        self.rule_stats.record_rule_usage(framework_id, combined_rules)
        
        return combined_rules
    
    def _get_static_rules_for_framework(self, framework_id):
        """Get static rules for a specific framework"""
        # Check if framework rules exist
        if framework_id in self.static_rules:
            return self.static_rules[framework_id]
            
        # Framework not found, return empty list
        return []
    
    def _get_dynamic_rules_for_framework(self, framework_id):
        """Get dynamic rules for a specific framework"""
        # Check if dynamic framework rules exist
        if framework_id in self.dynamic_rules:
            return self.dynamic_rules[framework_id]
            
        # Initialize empty dynamic rule list for this framework
        self.dynamic_rules[framework_id] = []
        return []
    
    def _generate_rules_from_regulation(self, framework_id):
        """Generate rules from regulatory text if available"""
        # Check if regulatory text is available
        reg_text_provider = self.config.get('regulatory_text_provider')
        if not reg_text_provider:
            return []
            
        try:
            # Get regulation text
            reg_text = reg_text_provider.get_regulation_text(framework_id)
            if not reg_text:
                return []
                
            # Use rule generation engine to create rules
            generated_rules = self.rule_adaptation_engine.generate_rules_from_text(
                reg_text, framework_id
            )
            
            return generated_rules
        except Exception as e:
            logging.warning(f"Error generating semantic rules from regulation: {str(e)}")
            return []
    
    def _combine_rules(self, static_rules, dynamic_rules, generated_rules):
        """Combine rules, handling conflicts with priority order"""
        # Implementation similar to DynamicTextPatternRules._combine_rules
        all_rules = []
        rule_ids = set()
        
        # Add generated rules (lowest priority)
        for rule in generated_rules:
            rule_id = rule.get('id')
            if rule_id not in rule_ids:
                all_rules.append(rule)
                rule_ids.add(rule_id)
                
        # Add static rules (middle priority)
        for rule in static_rules:
            rule_id = rule.get('id')
            if rule_id not in rule_ids:
                all_rules.append(rule)
                rule_ids.add(rule_id)
            else:
                # Replace generated rule with static rule
                for i, existing_rule in enumerate(all_rules):
                    if existing_rule.get('id') == rule_id:
                        all_rules[i] = rule
                        break
                        
        # Add dynamic rules (highest priority)
        for rule in dynamic_rules:
            rule_id = rule.get('id')
            if rule_id not in rule_ids:
                all_rules.append(rule)
                rule_ids.add(rule_id)
            else:
                # Replace existing rule with dynamic rule
                for i, existing_rule in enumerate(all_rules):
                    if existing_rule.get('id') == rule_id:
                        all_rules[i] = rule
                        break
                        
        return all_rules
    
    def get_conflict_resolution_strategy(self):
        """Get conflict resolution strategy for semantic rules"""
        # Default strategy
        default_strategy = {
            'priority': 'most_specific',
            'resolution_method': 'combine_conditions',
            'threshold_handling': 'take_lowest'
        }
        
        # Get custom strategy from config if available
        custom_strategy = self.config.get('semantic_conflict_strategy')
        if custom_strategy:
            return {**default_strategy, **custom_strategy}
            
        return default_strategy
    
    def create_rule(self, template_name, parameters, framework_id=None, rule_id=None):
        """
        Create a new rule based on a template
        
        Args:
            template_name: Name of the template to use
            parameters: Parameters to apply to template
            framework_id: Optional framework to associate with rule
            rule_id: Optional explicit rule ID
            
        Returns:
            Created rule or None if creation failed
        """
        # Implementation similar to DynamicTextPatternRules.create_rule
        # Check if template exists
        if template_name not in self.rule_templates:
            logging.warning(f"Template '{template_name}' not found")
            return None
            
        # Get template
        template = self.rule_templates[template_name]
        
        # Generate rule
        try:
            rule = self._generate_rule_from_template(template, parameters, rule_id)
            
            # Add rule to dynamic rules
            if framework_id:
                if framework_id not in self.dynamic_rules:
                    self.dynamic_rules[framework_id] = []
                self.dynamic_rules[framework_id].append(rule)
            else:
                # Add to generic rules
                if 'GENERIC' not in self.dynamic_rules:
                    self.dynamic_rules['GENERIC'] = []
                self.dynamic_rules['GENERIC'].append(rule)
                
            # Record rule creation
            self.rule_stats.record_rule_creation(rule.get('id'), framework_id)
            
            return rule
        except Exception as e:
            logging.warning(f"Error creating semantic rule: {str(e)}")
            return None
    
    def _generate_rule_from_template(self, template, parameters, rule_id=None):
        """Generate a rule by applying parameters to a template"""
        # Create a copy of the template
        rule = template.copy()
        
        # Generate ID if not provided
        if not rule_id:
            rule_id = f"semantic_rule_{uuid.uuid4().hex[:8]}"
            
        # Add ID to rule
        rule['id'] = rule_id
        
        # Apply concept template if exists
        for template_field in ['concept_template', 'topic_template', 'sentiment_template', 'class_template']:
            if template_field in rule:
                field_name = template_field.split('_')[0]  # concept, topic, etc.
                template_value = rule[template_field]
                
                if f"{{{field_name}}}" in template_value:
                    if field_name in parameters:
                        rule[field_name] = template_value.replace(f"{{{field_name}}}", parameters[field_name])
                
                # Remove template field
                del rule[template_field]
                
        # Apply other parameters directly
        for param_name, param_value in parameters.items():
            if param_name not in rule:
                rule[param_name] = param_value
                
        # Add creation timestamp
        rule['created_at'] = datetime.datetime.now().isoformat()
        
        return rule
    
    def update_rule(self, rule_id, updates, framework_id=None):
        """
        Update an existing rule
        
        Args:
            rule_id: ID of rule to update
            updates: Dictionary of updates to apply
            framework_id: Optional framework ID to limit search
            
        Returns:
            Updated rule or None if update failed
        """
        # Implementation similar to DynamicTextPatternRules.update_rule
        # Find the rule
        rule_found = False
        updated_rule = None
        
        # Check dynamic rules
        if framework_id:
            frameworks_to_check = [framework_id]
        else:
            frameworks_to_check = list(self.dynamic_rules.keys())
            
        for fw_id in frameworks_to_check:
            for i, rule in enumerate(self.dynamic_rules.get(fw_id, [])):
                if rule.get('id') == rule_id:
                    # Apply updates
                    for key, value in updates.items():
                        rule[key] = value
                        
                    # Update timestamp
                    rule['updated_at'] = datetime.datetime.now().isoformat()
                    
                    # Update rule in dynamic rules
                    self.dynamic_rules[fw_id][i] = rule
                    updated_rule = rule
                    rule_found = True
                    break
                    
            if rule_found:
                break
                
        # Check static rules if not found in dynamic
        if not rule_found:
            for fw_id, rules in self.static_rules.items():
                if framework_id and fw_id != framework_id:
                    continue
                    
                for i, rule in enumerate(rules):
                    if rule.get('id') == rule_id:
                        # Create dynamic rule from static with updates
                        new_rule = rule.copy()
                        for key, value in updates.items():
                            new_rule[key] = value
                            
                        # Add timestamps
                        new_rule['created_at'] = datetime.datetime.now().isoformat()
                        new_rule['updated_at'] = datetime.datetime.now().isoformat()
                        new_rule['derived_from'] = rule_id
                        
                        # Add to dynamic rules
                        if fw_id not in self.dynamic_rules:
                            self.dynamic_rules[fw_id] = []
                        self.dynamic_rules[fw_id].append(new_rule)
                        updated_rule = new_rule
                        rule_found = True
                        break
                        
                if rule_found:
                    break
                    
        # Record update if successful
        if updated_rule:
            self.rule_stats.record_rule_update(rule_id, framework_id)
            
        return updated_rule
    
    def delete_rule(self, rule_id, framework_id=None):
        """
        Delete a dynamic rule
        
        Args:
            rule_id: ID of rule to delete
            framework_id: Optional framework ID to limit search
            
        Returns:
            True if rule was deleted, False otherwise
        """
        # Implementation similar to DynamicTextPatternRules.delete_rule
        # Only dynamic rules can be deleted
        if framework_id:
            frameworks_to_check = [framework_id]
        else:
            frameworks_to_check = list(self.dynamic_rules.keys())
            
        for fw_id in frameworks_to_check:
            for i, rule in enumerate(self.dynamic_rules.get(fw_id, [])):
                if rule.get('id') == rule_id:
                    # Remove rule
                    del self.dynamic_rules[fw_id][i]
                    
                    # Record deletion
                    self.rule_stats.record_rule_deletion(rule_id, fw_id)
                    
                    return True
                    
        return False
    
    def adapt_rules_based_on_feedback(self, feedback_data):
        """
        Adapt rules based on feedback data
        
        Args:
            feedback_data: Dictionary with feedback on rule performance
            
        Returns:
            List of rule adaptations made
        """
        # Use rule adaptation engine to adapt rules
        adaptations = self.rule_adaptation_engine.adapt_rules(
            feedback_data, self.dynamic_rules, self.static_rules, self.rule_stats
        )
        
        # Apply adaptations
        for adaptation in adaptations:
            adaptation_type = adaptation.get('type')
            
            if adaptation_type == 'update':
                self.update_rule(
                    adaptation['rule_id'], 
                    adaptation['updates'],
                    adaptation.get('framework_id')
                )
            elif adaptation_type == 'create':
                self.create_rule(
                    adaptation['template_name'],
                    adaptation['parameters'],
                    adaptation.get('framework_id'),
                    adaptation.get('rule_id')
                )
            elif adaptation_type == 'delete':
                self.delete_rule(
                    adaptation['rule_id'],
                    adaptation.get('framework_id')
                )
                
        return adaptations
    
    def get_rule_performance_report(self, framework_id=None):
        """
        Get report on rule performance
        
        Args:
            framework_id: Optional framework ID to limit report
            
        Returns:
            Report on rule performance
        """
        return self.rule_stats.generate_report(framework_id)

class EnhancedCompliantLanguageProcessor:
    """
    Enhanced implementation that combines the best optimization techniques
    from both sets of recommendations
    """
    def __init__(self, config):
        # System configuration
        self.config = config
        
        # Initialize component registry with domain specialists
        self.model_registry = ComplianceModelRegistry()
        self._initialize_specialist_models()
        
        # Initialize optimized knowledge base with efficient indexing
        self.regulatory_kb = self._initialize_optimized_knowledge_base()
        
        # Initialize main language model and quantized embedding space
        self.llm = self._initialize_optimized_llm()
        self.regulatory_embedding = QuantizedRegulatoryEmbeddingSpace(
            embedding_dim=self.llm.embedding_dim,
            regulatory_knowledge_base=self.regulatory_kb,
            quantization_bits=8
        )
        
        # Initialize neural-symbolic interface with bidirectional optimization
        self.interface = OptimizedNeuralSymbolicInterface(
            base_interface=None,
            language_model=self.llm,
            regulatory_knowledge_base=self.regulatory_kb
        )
        
        # Initialize compliance components with parallelization
        self.input_filter = ParallelCompliancePrefilter(self.config["input_filtering"])
        self.token_gate = OptimizedTokenLevelComplianceGate(
            language_model=self.llm,
            regulatory_constraints=self.regulatory_kb.get_all_constraints(),
            max_workers=self.config.get("max_workers", 8)
        )
        self.semantic_monitor = SemanticComplianceMonitor(
            compliance_config=self.config["semantic_monitoring"]
        )
        
        # Initialize verifier with proof tracing for auditability
        self.compliance_verifier = ComplianceProofTracer(
            ruleset=self.regulatory_kb,
            symbolic_engine=self._initialize_symbolic_engine()
        )
        
        # Initialize RAG components with context-aware retrieval
        self.retriever = OptimizedRegulatoryRetrieval(
            regulatory_document_store=self.regulatory_kb.document_store,
            embedding_model=self.llm
        )
        self.augmenter = EfficientRegulationAugmenter(
            regulatory_retriever=self.retriever,
            language_model=self.llm
        )
        
        # Initialize performance monitoring and dynamic optimization
        self.performance_optimizer = ComplianceSystemPerformanceOptimizer(
            compliance_system=self,
            monitoring_config=self.config["monitoring"]
        )
        
        # Caching and state management
        self._constraint_cache = LRUCache(maxsize=1000)
        self._framework_cache = LRUCache(maxsize=500)
        
        # Start performance monitoring
        self.performance_optimizer.start_monitoring()

    def generate_compliant_text(self, prompt, context=None, compliance_mode='strict',
                               max_tokens=100, rag_enabled=True):
        """Generate text with optimized compliance enforcement"""
        try:
            # Analyze request to determine optimal processing strategy
            processing_strategy = self._determine_optimal_strategy(prompt, context)
            
            # STAGE 1: Optimized parallel input filtering
            input_filter_result = self.input_filter.filter_input_parallel(prompt, context)
            
            if not input_filter_result['is_compliant']:
                return self._format_compliance_error(
                    input_filter_result['issues'],
                    {'stage': 'input_filtering'}
                )
            
            # STAGE 2: Select optimal model based on context and constraints
            selected_model = self._select_optimal_model(
                processing_strategy, prompt, context, compliance_mode
            )
            
            # STAGE 3: Context-aware RAG augmentation if enabled
            if rag_enabled:
                augmented_prompt, used_regulations = self.augmenter.augment_prompt(
                    input_filter_result['filtered_input'], 
                    context,
                    available_tokens=self._calculate_available_tokens(
                        input_filter_result['filtered_input'],
                        selected_model
                    )
                )
            else:
                augmented_prompt = input_filter_result['filtered_input']
                used_regulations = []
            
            # STAGE 4: Efficient framework and constraint determination
            applicable_frameworks = self._get_applicable_frameworks(
                augmented_prompt, context, compliance_mode
            )
            applicable_constraints = self._get_applicable_constraints(
                applicable_frameworks, context, compliance_mode
            )
            
            # STAGE 5: Initialize semantic state with optimized monitoring
            semantic_state = self.semantic_monitor.initialize_optimized(
                augmented_prompt, applicable_frameworks
            )
            
            # STAGE 6: Generate text with parallel constraint enforcement
            generation_result = self._generate_with_constraints(
                selected_model,
                augmented_prompt,
                semantic_state,
                applicable_constraints,
                max_tokens,
                compliance_mode
            )
            
            # STAGE 7: Verify output with proof tracing
            verification_result, proof_trace = self.compliance_verifier.verify_with_proof(
                generation_result['text'],
                applicable_frameworks,
                compliance_mode
            )
            
            # STAGE 8: Generate explanation if needed
            explanation = None
            if not verification_result['is_compliant'] or verification_result.get('has_modifications', False):
                explanation = self._generate_explanation_from_proof(
                    proof_trace, prompt, generation_result['text']
                )
            
            # Construct final result
            return {
                'text': generation_result['text'],
                'is_compliant': verification_result['is_compliant'],
                'compliance_score': verification_result.get('compliance_score', 0.0),
                'modified': verification_result.get('has_modifications', False),
                'explanation': explanation,
                'proof_trace': proof_trace if self.config.get('include_proof_trace', False) else None,
                'compliance_metadata': {
                    'frameworks': [f.id for f in applicable_frameworks],
                    'constraints_count': len(applicable_constraints),
                    'used_regulations': used_regulations,
                    'processing_strategy': processing_strategy,
                    'selected_model': getattr(selected_model, 'id', 'default'),
                    'verification_details': verification_result.get('metadata', {})
                }
            }
            
        except Exception as e:
            logging.error(f"Error in compliant text generation: {str(e)}")
            traceback.print_exc()
            return self._format_compliance_error(
                f"Internal error: {str(e)}",
                {'error': str(e)}
            )
    
    def _determine_optimal_strategy(self, prompt, context):
        """Determine optimal processing strategy based on request characteristics"""
        # Analyze request complexity and regulatory relevance
        complexity = self._analyze_complexity(prompt)
        regulatory_specificity = self._analyze_regulatory_specificity(context)
        
        # Determine processing strategy
        if complexity < 0.3 and regulatory_specificity < 0.3:
            return "lightweight"  # Use lightweight specialist model
        elif complexity > 0.7 or regulatory_specificity > 0.7:
            return "full_compliance"  # Use full compliance model with all verifications
        else:
            return "hybrid"  # Use hybrid approach
    
    def _select_optimal_model(self, strategy, prompt, context, compliance_mode):
        """Select optimal model based on processing strategy and constraints"""
        if strategy == "lightweight":
            # Try to find appropriate specialist model
            domain = self._detect_regulatory_domain(prompt, context)
            if domain and domain in self.model_registry.models:
                return self.model_registry.models[domain]["model"]
        
        # Default to main LLM for full compliance or when no specialist is available
        return self.llm
    
    def _get_applicable_frameworks(self, prompt, context, compliance_mode):
        """Get applicable frameworks with caching optimization"""
        cache_key = f"{hash(prompt)}:{hash(str(context))}:{compliance_mode}"
        if cache_key in self._framework_cache:
            return self._framework_cache[cache_key]
        
        frameworks = self.regulatory_kb.get_applicable_frameworks(
            prompt, context, compliance_mode
        )
        
        # Cache result
        self._framework_cache[cache_key] = frameworks
        return frameworks
    
    def _get_applicable_constraints(self, frameworks, context, compliance_mode):
        """Get applicable constraints with efficient processing"""
        cache_key = f"{','.join(f.id for f in frameworks)}:{compliance_mode}"
        if cache_key in self._constraint_cache:
            return self._constraint_cache[cache_key]
        
        # Use parallel constraint extraction for better performance
        constraints = []
        with ThreadPoolExecutor(max_workers=min(len(frameworks), 4)) as executor:
            framework_constraints = list(executor.map(
                lambda f: f.get_requirements(context),
                frameworks
            ))
            
        # Flatten constraint lists
        for framework_constraint_list in framework_constraints:
            constraints.extend(framework_constraint_list)
        
        # Resolve conflicts
        resolved_constraints = self.regulatory_kb.resolve_conflicts(constraints)
        
        # Cache result
        self._constraint_cache[cache_key] = resolved_constraints
        return resolved_constraints
    
    def _generate_with_constraints(self, model, prompt, semantic_state, constraints, 
                                  max_tokens, compliance_mode):
        """Generate text with optimized constraint enforcement"""
        if hasattr(model, 'generate_compliant_text'):
            # Use model's built-in compliance capability if available
            return model.generate_compliant_text(
                prompt, 
                semantic_state=semantic_state,
                constraints=constraints,
                max_tokens=max_tokens,
                compliance_mode=compliance_mode
            )
        else:
            # Use token gate for constraint enforcement
            return self.token_gate.generate_compliant_text(
                model,
                prompt,
                semantic_state=semantic_state,
                constraints=constraints,
                max_tokens=max_tokens,
                compliance_mode=compliance_mode
            )
    
    def _generate_explanation_from_proof(self, proof_trace, prompt, generated_text):
        """Generate explanation from proof trace with enhanced detail"""
        if not proof_trace or not proof_trace.get('steps'):
            # Fallback explanation if no detailed proof trace
            return {
                'summary': "The generated content does not fully comply with applicable regulations.",
                'details': "No detailed compliance proof is available."
            }
        
        # Extract key information from proof trace
        violated_rules = [
            step for step in proof_trace.get('steps', [])
            if step.get('intermediate_conclusion', '').find('violates') >= 0
        ]
        
        # Generate natural language explanation
        explanation = {
            'summary': proof_trace.get('conclusion', {}).get(
                'justification', 
                "Compliance verification completed."
            ),
            'violated_rules': [
                {
                    'rule_id': rule.get('rule_id'),
                    'rule_text': rule.get('rule_text'),
                    'explanation': self._simplify_rule_violation(rule)
                }
                for rule in violated_rules
            ],
            'compliance_status': "non_compliant" if violated_rules else "compliant"
        }
        
        return explanation
    
    def _simplify_rule_violation(self, rule_step):
        """Convert technical rule violation to simpler explanation"""
        # Implementation would convert technical details to plain language
        return f"This content appears to violate {rule_step.get('rule_id')} which requires {rule_step.get('rule_text', 'regulatory compliance')}."
    
    def _format_compliance_error(self, error, metadata):
        """Format compliance error response"""
        return {
            'text': None,
            'is_compliant': False,
            'compliance_error': error,
            'compliance_metadata': metadata
        }
    
    def clear_caches(self):
        """Clear all caches for testing or memory management"""
        self._constraint_cache.clear()
        self._framework_cache.clear()
        self.interface.neural_to_symbolic_cache.clear()
        self.interface.symbolic_to_neural_cache.clear()
        
    def _calculate_available_tokens(self, prompt, model):
        """Calculate available tokens for RAG augmentation"""
        prompt_tokens = len(self.llm.tokenize(prompt))
        if hasattr(model, 'get_context_size'):
            context_size = model.get_context_size()
        else:
            context_size = 4096  # Default assumption
            
        # Reserve tokens for response
        reserved_tokens = min(context_size // 3, 1000)
        
        # Calculate available tokens with safety margin
        available = context_size - prompt_tokens - reserved_tokens - 50
        return max(available, 0)  # Ensure non-negative


    def _create_pruned_specialist(self, domain):
        """
        Create a progressively pruned specialist model for a specific regulatory domain.
        
        Args:
            domain: Regulatory domain (e.g., "GDPR")
            
        Returns:
            Pruned specialist model
        """
        # In a real implementation, this would load and prune a pre-trained model
        # Placeholder implementation
        class PrunedSpecialistModel:
            def __init__(self, domain):
                self.domain = domain
                self.id = f"{domain}_pruned_specialist"
                self.embedding_dim = 768
                
            def generate_compliant_text(self, prompt, semantic_state=None, constraints=None, 
                                    max_tokens=100, compliance_mode="standard"):
                # Specialist model with built-in compliance for domain
                return {
                    "text": f"Compliant response for {self.domain} query",
                    "is_compliant": True,
                    "compliance_score": 0.95
                }
                
            def get_embeddings(self, text):
                # Placeholder embedding generation
                return np.random.randn(self.embedding_dim)
                
            def get_context_size(self):
                return 2048  # Smaller context size than base model
                
        return PrunedSpecialistModel(domain)
        
    def _create_quantized_specialist(self, domain):
        """
        Create a quantized specialist model for a specific regulatory domain.
        
        Args:
            domain: Regulatory domain (e.g., "HIPAA")
            
        Returns:
            Quantized specialist model
        """
        # In a real implementation, this would load and quantize a pre-trained model
        # Placeholder implementation
        class QuantizedSpecialistModel:
            def __init__(self, domain):
                self.domain = domain
                self.id = f"{domain}_quantized_specialist"
                self.embedding_dim = 768
                self.quantization_bits = 8
                
            def generate_compliant_text(self, prompt, semantic_state=None, constraints=None, 
                                    max_tokens=100, compliance_mode="standard"):
                # Specialist model with built-in compliance for domain
                return {
                    "text": f"Compliant response for {self.domain} query with quantized precision",
                    "is_compliant": True,
                    "compliance_score": 0.92
                }
                
            def get_embeddings(self, text):
                # Placeholder embedding generation
                return np.random.randn(self.embedding_dim)
                
            def get_context_size(self):
                return 4096
                
        return QuantizedSpecialistModel(domain)
        
    def _create_distilled_specialist(self, domain):
        """
        Create a knowledge-distilled specialist model for a specific regulatory domain.
        
        Args:
            domain: Regulatory domain (e.g., "FINREG")
            
        Returns:
            Distilled specialist model
        """
        # In a real implementation, this would load a knowledge-distilled model
        # Placeholder implementation
        class DistilledSpecialistModel:
            def __init__(self, domain):
                self.domain = domain
                self.id = f"{domain}_distilled_specialist"
                self.embedding_dim = 384  # Smaller embedding dimension due to distillation
                
            def generate_compliant_text(self, prompt, semantic_state=None, constraints=None, 
                                    max_tokens=100, compliance_mode="standard"):
                # Specialist model with built-in compliance for domain
                return {
                    "text": f"Compliant response for {self.domain} query with distilled knowledge",
                    "is_compliant": True,
                    "compliance_score": 0.9
                }
                
            def get_embeddings(self, text):
                # Placeholder embedding generation
                return np.random.randn(self.embedding_dim)
                
            def get_context_size(self):
                return 2048  # Smaller context window
                
        return DistilledSpecialistModel(domain)

    def count_parameters(self, model):
        """
        Count number of trainable parameters in a model.
        
        Args:
            model: Model to count parameters for
            
        Returns:
            Number of parameters
        """
        # In a real implementation, this would count actual model parameters
        # Placeholder implementation
        if hasattr(model, 'domain'):
            if 'pruned' in getattr(model, 'id', ''):
                return 350_000_000  # 350M parameters for pruned model
            elif 'quantized' in getattr(model, 'id', ''):
                return 750_000_000  # 750M parameters for quantized model
            elif 'distilled' in getattr(model, 'id', ''):
                return 250_000_000  # 250M parameters for distilled model
        
        # Default for unknown model
        return 1_000_000_000  # 1B parameters
        
    def estimate_memory_footprint(self, model):
        """
        Estimate memory footprint of a model in MB.
        
        Args:
            model: Model to estimate memory footprint for
            
        Returns:
            Estimated memory usage in MB
        """
        # In a real implementation, this would measure actual memory usage
        # Simplified estimation based on parameter count and compression
        params = self.count_parameters(model)
        
        # Base memory: 4 bytes per parameter for FP32
        base_memory = params * 4 / (1024 * 1024)  # Convert to MB
        
        # Apply compression factor based on model type
        if hasattr(model, 'id'):
            model_id = model.id
            if 'pruned' in model_id:
                return base_memory * 0.6  # 40% reduction from pruning
            elif 'quantized' in model_id:
                bits = getattr(model, 'quantization_bits', 8)
                return base_memory * (bits / 32)  # Reduction based on quantization bits
            elif 'distilled' in model_id:
                return base_memory * 0.25  # 75% reduction from distillation
        
        # Default for unknown model
        return base_memory
        
    def _initialize_optimized_knowledge_base(self):
        """
        Initialize optimized regulatory knowledge base with efficient indexing.
        
        Returns:
            Optimized regulatory knowledge base
        """
        class OptimizedRegulatoryKnowledgeBase:
            def __init__(self):
                # Initialize document store
                self.document_store = RegulatoryDocumentStore()
                
                # Initialize indices
                self.domain_index = {}
                self.framework_index = {}
                self.concept_index = {}
                
                # Load default regulatory frameworks
                self._load_default_frameworks()
                
            def get_applicable_frameworks(self, prompt=None, context=None, compliance_mode="standard"):
                """Get applicable regulatory frameworks for a prompt"""
                # Simple implementation returning all frameworks
                # A real implementation would analyze the prompt and context
                return self.get_all_frameworks()
                
            def get_all_frameworks(self):
                """Get all regulatory frameworks"""
                return list(self.framework_index.values())
                
            def get_domains(self):
                """Get all regulatory domains"""
                return list(self.domain_index.keys())
                
            def get_all_concepts(self):
                """Get all regulatory concepts"""
                return list(self.concept_index.values())
                
            def get_all_constraints(self):
                """Get all regulatory constraints"""
                constraints = []
                for framework in self.get_all_frameworks():
                    constraints.extend(framework.get_requirements())
                return constraints
                
            def resolve_conflicts(self, constraints):
                """Resolve conflicts between constraints"""
                # Simple implementation without actual conflict resolution
                # A real implementation would detect and resolve conflicts
                return constraints
                
            def get_concept_description(self, concept_id):
                """Get description for a concept"""
                if concept_id in self.concept_index:
                    return self.concept_index[concept_id].description
                return None
                
            def _load_default_frameworks(self):
                """Load default regulatory frameworks"""
                # Create sample frameworks
                gdpr = RegulatoryFramework(
                    id="GDPR",
                    name="General Data Protection Regulation",
                    domain="data_privacy"
                )
                
                hipaa = RegulatoryFramework(
                    id="HIPAA",
                    name="Health Insurance Portability and Accountability Act",
                    domain="health_privacy"
                )
                
                finreg = RegulatoryFramework(
                    id="FINREG",
                    name="Financial Regulatory Framework",
                    domain="financial_compliance"
                )
                
                # Add to indices
                self.framework_index = {
                    "GDPR": gdpr,
                    "HIPAA": hipaa,
                    "FINREG": finreg
                }
                
                self.domain_index = {
                    "data_privacy": [gdpr],
                    "health_privacy": [hipaa],
                    "financial_compliance": [finreg]
                }
                
                # Create sample concepts
                concepts = [
                    RegulatoryConceptDefinition(
                        id="personal_data",
                        name="Personal Data",
                        domain="data_privacy",
                        description="Information relating to an identified or identifiable natural person"
                    ),
                    RegulatoryConceptDefinition(
                        id="data_subject_rights",
                        name="Data Subject Rights",
                        domain="data_privacy",
                        description="Rights granted to individuals regarding their personal data"
                    ),
                    RegulatoryConceptDefinition(
                        id="phi",
                        name="Protected Health Information",
                        domain="health_privacy",
                        description="Individually identifiable health information"
                    ),
                    RegulatoryConceptDefinition(
                        id="aml",
                        name="Anti-Money Laundering",
                        domain="financial_compliance",
                        description="Regulations to prevent money laundering and financial crimes"
                    )
                ]
                
                # Add concepts to index
                self.concept_index = {concept.id: concept for concept in concepts}
                
        return OptimizedRegulatoryKnowledgeBase() 
       
    def _initialize_optimized_llm(self):
        """
        Initialize optimized base language model.
        
        Returns:
            Optimized language model
        """
        class OptimizedLanguageModel:
            def __init__(self):
                self.id = "base_llm"
                self.embedding_dim = 1024
                self.tokenizer = self._initialize_tokenizer()
                self.eos_token_id = 0  # End of sequence token ID
                
            def get_next_token_logits(self, input_ids):
                """Get next token logits for input sequence"""
                # Placeholder implementation
                return np.random.randn(50000)  # Random logits for vocabulary
                
            def sample_token(self, logits):
                """Sample token from logits"""
                # Simple sampling implementation
                # In real systems, this would use temperature, top-k, top-p, etc.
                probs = self._softmax(logits)
                return np.random.choice(len(probs), p=probs)
                
            def tokenize(self, text):
                """Tokenize text to token IDs"""
                # Simplified tokenization (placeholder)
                return [1, 2, 3, 4, 5]  # Dummy token IDs
                
            def decode_token(self, token_id):
                """Decode token ID to text"""
                # Simplified token decoding (placeholder)
                return f"<token_{token_id}>"
                
            def get_embeddings(self, text):
                """Get embeddings for text"""
                # Placeholder embedding generation
                return np.random.randn(self.embedding_dim)
                
            def is_eos_token(self, token_id):
                """Check if token is end-of-sequence token"""
                return token_id == self.eos_token_id
                
            def get_context_size(self):
                """Get maximum context size"""
                return 8192
                
            def _initialize_tokenizer(self):
                """Initialize tokenizer"""
                # Placeholder tokenizer
                class SimpleTokenizer:
                    def encode(self, text):
                        return [1, 2, 3, 4, 5]  # Dummy token IDs
                        
                    def decode(self, token_ids):
                        return "Decoded text"
                        
                return SimpleTokenizer()
                
            def _softmax(self, logits):
                """Convert logits to probabilities"""
                exp_logits = np.exp(logits - np.max(logits))
                return exp_logits / exp_logits.sum()
                
        return OptimizedLanguageModel()
        
    def _initialize_symbolic_engine(self):
        """
        Initialize symbolic reasoning engine for compliance verification.
        
        Returns:
            Symbolic reasoning engine
        """
        class SymbolicReasoningEngine:
            def __init__(self):
                # Initialize logical reasoner
                self.reasoner = self._initialize_reasoner()
                
            def text_to_symbolic(self, text):
                """Convert text to symbolic representation"""
                # Placeholder implementation
                return {
                    "statements": [
                        {"subject": "user", "predicate": "requests", "object": "information"},
                    ],
                    "concepts": ["information_request"],
                    "intents": ["query"],
                    "tone": "neutral"
                }
                
            def verify_compliance(self, symbolic_repr, rules):
                """Verify compliance against rules"""
                # Placeholder implementation
                return {
                    "is_compliant": True,
                    "rule_results": {},
                    "reasons": []
                }
                
            def _initialize_reasoner(self):
                """Initialize logical reasoning component"""
                class LogicalReasoner:
                    def apply_rules(self, facts, rules):
                        # Placeholder implementation
                        return {"is_valid": True, "conclusions": []}
                        
                return LogicalReasoner()
                
        return SymbolicReasoningEngine()
        
    def _analyze_complexity(self, prompt):
        """
        Analyze complexity of the prompt to determine optimal processing strategy.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Complexity score (0-1)
        """
        # In a real implementation, this would analyze various complexity factors
        # Simplified implementation based on length and structure
        
        # Length factor (longer prompts are more complex)
        length = len(prompt)
        length_factor = min(length / 1000, 1.0)
        
        # Structure factor (prompts with questions are more complex)
        question_factor = 0.3 if "?" in prompt else 0.0
        
        # Topic factor (certain topics increase complexity)
        complex_topics = ["compliance", "regulation", "legal", "technical", "medical"]
        topic_matches = sum(1 for topic in complex_topics if topic.lower() in prompt.lower())
        topic_factor = min(topic_matches * 0.15, 0.5)
        
        # Combine factors
        complexity = (0.5 * length_factor) + (0.3 * question_factor) + (0.2 * topic_factor)
        
        return min(complexity, 1.0)
        
    def _analyze_regulatory_specificity(self, context):
        """
        Analyze regulatory specificity of the context.
        
        Args:
            context: Context information
            
        Returns:
            Regulatory specificity score (0-1)
        """
        # If no context, low specificity
        if not context:
            return 0.1
            
        # Convert context to string if it's a dictionary
        if isinstance(context, dict):
            context_str = str(context)
        else:
            context_str = str(context)
            
        # Check for regulatory framework mentions
        frameworks = ["GDPR", "HIPAA", "FINREG", "CCPA", "PCI DSS", "SOX"]
        framework_matches = sum(1 for fw in frameworks if fw in context_str)
        framework_factor = min(framework_matches * 0.2, 0.6)
        
        # Check for regulatory concept mentions
        concepts = ["compliance", "regulation", "privacy", "security", "consent", "data subject"]
        concept_matches = sum(1 for concept in concepts if concept.lower() in context_str.lower())
        concept_factor = min(concept_matches * 0.1, 0.4)
        
        # Combine factors
        specificity = (0.7 * framework_factor) + (0.3 * concept_factor)
        
        return min(specificity, 1.0)
        
    def _detect_regulatory_domain(self, prompt, context):
        """
        Detect relevant regulatory domain for the request.
        
        Args:
            prompt: Input prompt text
            context: Context information
            
        Returns:
            Detected domain or None
        """
        # Combine prompt and context for analysis
        combined_text = prompt
        if context:
            if isinstance(context, dict):
                combined_text += " " + str(context)
            else:
                combined_text += " " + context
                
        combined_text = combined_text.lower()
        
        # Check for domain-specific indicators
        domain_indicators = {
            "GDPR": ["gdpr", "personal data", "data subject", "privacy", "consent", "eu regulation"],
            "HIPAA": ["hipaa", "health", "medical", "patient", "phi", "healthcare"],
            "FINREG": ["financial", "banking", "aml", "kyc", "transaction", "money laundering"]
        }
        
        # Score each domain
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(combined_text.count(indicator) for indicator in indicators)
            domain_scores[domain] = score
            
        # Get domain with highest score
        if not domain_scores:
            return None
            
        max_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        # Only return domain if score is above threshold
        if max_domain[1] > 0:
            return max_domain[0]
            
        return None
    
    def _format_compliance_error(self, error, metadata):
        """
        Format compliance error response.
        
        Args:
            error: Error message or description
            metadata: Additional metadata about the error
            
        Returns:
            Formatted error response
        """
        if isinstance(error, list):
            # Multiple issues
            primary_error = error[0] if error else "Compliance requirements not met"
            additional_issues = error[1:] if len(error) > 1 else []
            
            return {
                'text': None,
                'is_compliant': False,
                'compliance_error': primary_error,
                'additional_issues': additional_issues,
                'compliance_metadata': metadata
            }
        else:
            # Single issue
            return {
                'text': None,
                'is_compliant': False,
                'compliance_error': error,
                'compliance_metadata': metadata
            }