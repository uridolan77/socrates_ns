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
from src.compliance.monitoring.compliance_monitor import SemanticComplianceMonitor
from src.neuro.embeddings.domain_specific_space import RegulatoryEmbeddingSpace
from src.neuro.embeddings.quantized_space import QuantizedRegulatoryEmbeddingSpace
from src.compliance.verification.violation_analyzer import ViolationAnalyzer
from src.compliance.verification.constraint_optimizer import ComplianceConstraintOptimizer
from src.interface.optimized_interface import OptimizedNeuralSymbolicInterface
from src.symbolic.rules.rule_performance_tracker import RulePerformanceTracker
from src.utils.error.compliance_error import ComplianceError 
from src.utils.error.compliance_error_handler import ComplianceErrorHandler

# Import LLM Gateway components
from src.llm_gateway.core.client import LLMGatewayClient
from src.llm_gateway.core.models import (
    LLMRequest, 
    LLMResponse, 
    LLMConfig, 
    InterventionContext,
    ContentItem,
    FinishReason,
    GatewayConfig
)
from src.llm_gateway.core.factory import ProviderFactory

class GatewayCompliantLanguageProcessor:
    """
    Enhanced implementation that uses the LLM Gateway for generation
    while incorporating optimized compliance techniques.
    """
    def __init__(self, gateway_client, config):
        """
        Initialize the enhanced processor with gateway client.
        
        Args:
            gateway_client: LLMGatewayClient instance
            config: Configuration dictionary
        """
        # System configuration
        self.config = config
        self.gateway_client = gateway_client
        
        # Initialize component registry with domain specialists
        self.model_registry = ComplianceModelRegistry()
        self._initialize_specialist_models()
        
        # Initialize optimized knowledge base with efficient indexing
        self.regulatory_kb = self._initialize_optimized_knowledge_base()
        
        # Initialize quantized embedding space
        self.regulatory_embedding = QuantizedRegulatoryEmbeddingSpace(
            embedding_dim=config.get("embedding_dim", 1024),
            regulatory_knowledge_base=self.regulatory_kb,
            quantization_bits=8
        )
        
        # Initialize neural-symbolic interface with bidirectional optimization
        self.interface = OptimizedNeuralSymbolicInterface(
            base_interface=None,
            language_model=None,  # Use gateway instead of direct model
            regulatory_knowledge_base=self.regulatory_kb
        )
        
        # Initialize compliance components with parallelization
        self.input_filter = ParallelCompliancePrefilter(self.config["input_filtering"])
        self.token_gate = OptimizedTokenLevelComplianceGate(
            language_model=None,  # Use gateway instead of direct model
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
            embedding_model=None  # Will use gateway for embeddings
        )
        self.augmenter = EfficientRegulationAugmenter(
            regulatory_retriever=self.retriever,
            language_model=None  # Will use gateway for embeddings
        )
        
        # Initialize performance monitoring and dynamic optimization
        self.performance_optimizer = ComplianceSystemPerformanceOptimizer(
            compliance_system=self,
            monitoring_config=self.config["monitoring"]
        )
        
        # Caching and state management
        self._constraint_cache = LRUCache(maxsize=1000)
        self._framework_cache = LRUCache(maxsize=500)
        
        # Default model identifier for gateway requests
        self.default_model_identifier = config.get("default_model", "default_model")
        
        # Start performance monitoring
        self.performance_optimizer.start_monitoring()

        self.error_handler = ComplianceErrorHandler(self.logger, self.config["compliance_error"])
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized EnhancedGatewayCompliantLanguageProcessor with gateway client")

    async def generate_compliant_text(self, prompt, context=None, compliance_mode='strict',
                                    max_tokens=100, rag_enabled=True, model_identifier=None):
        """
        Generate text with optimized compliance enforcement using the gateway.
        
        Args:
            prompt: Input prompt text
            context: Additional context information
            compliance_mode: Compliance strictness mode
            max_tokens: Maximum tokens to generate
            rag_enabled: Whether to use RAG augmentation
            model_identifier: Optional model identifier
            
        Returns:
            Dict with generated text and compliance info
        """
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
            model_id = model_identifier or self._select_optimal_model(
                processing_strategy, prompt, context, compliance_mode
            )
            
            # STAGE 3: Context-aware RAG augmentation if enabled
            if rag_enabled:
                augmented_prompt, used_regulations = self._simulate_augmentation(
                    input_filter_result['filtered_input'], 
                    context
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
            
            # STAGE 6: Generate text through the gateway
            # Prepare intervention context with regulatory information
            intervention_context = InterventionContext(
                request_id=str(uuid.uuid4()),
                required_compliance_frameworks=[fw.id for fw in applicable_frameworks],
                compliance_mode=compliance_mode,
                user_info={"context": context} if context else {}
            )
            
            # Add semantic state to intervention data
            intervention_context.intervention_data.set("semantic_state", semantic_state)
            
            # Create config with constraints as metadata
            llm_config = LLMConfig(
                model_identifier=model_id,
                max_tokens=max_tokens,
                # Add any additional parameters from the strategy
                temperature=self._get_temperature_for_strategy(processing_strategy)
            )
            
            # Create request with strategy and constraint information
            llm_request = LLMRequest(
                prompt_content=augmented_prompt,
                config=llm_config,
                initial_context=intervention_context,
                # Pass strategy and constraints through extensions
                extensions={
                    "compliance_constraints": applicable_constraints,
                    "processing_strategy": processing_strategy
                }
            )
            
            # Generate through gateway
            response = await self.gateway_client.generate(llm_request)
            
            if response.error_details:
                return self._format_compliance_error(
                    response.error_details.message,
                    {
                        'code': response.error_details.code,
                        'level': response.error_details.level,
                        'provider_details': response.error_details.provider_error_details
                    }
                )
            
            # Extract generated text
            generated_text = ""
            if isinstance(response.generated_content, str):
                generated_text = response.generated_content
            elif isinstance(response.generated_content, list) and len(response.generated_content) > 0:
                # Concatenate text content items
                for item in response.generated_content:
                    if hasattr(item, 'text_content') and item.text_content:
                        generated_text += item.text_content + " "
                generated_text = generated_text.strip()
            
            # STAGE 7: Verify output with proof tracing
            verification_result, proof_trace = self.compliance_verifier.verify_with_proof(
                generated_text,
                applicable_frameworks,
                compliance_mode
            )
            
            # STAGE 8: Generate explanation if needed
            explanation = None
            if not verification_result['is_compliant'] or verification_result.get('has_modifications', False):
                explanation = self._generate_explanation_from_proof(
                    proof_trace, prompt, generated_text
                )
            
            # Construct final result
            return {
                'text': generated_text,
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
                    'selected_model': model_id,
                    'verification_details': verification_result.get('metadata', {})
                },
                'usage': response.usage.model_dump() if response.usage else None,
                'finish_reason': response.finish_reason
            }
            
        except Exception as e:
            self.logger.error(f"Error in compliant text generation: {str(e)}")
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
        """
        Select optimal model identifier based on strategy and constraints.
        
        Returns the model identifier string to use with the gateway.
        """
        if strategy == "lightweight":
            # Try to find appropriate specialist model
            domain = self._detect_regulatory_domain(prompt, context)
            if domain and domain in self.model_registry.models:
                specialist_model = self.model_registry.models[domain]["model"]
                return f"{domain}_specialist"
        
        # Default to configured model for full compliance or when no specialist is available
        return self.default_model_identifier
    
    def _get_temperature_for_strategy(self, strategy):
        """Get appropriate temperature value for the given strategy"""
        if strategy == "lightweight":
            return 0.7  # Higher creativity for non-critical content
        elif strategy == "full_compliance":
            return 0.2  # More conservative for high-compliance needs
        else:
            return 0.5  # Balanced approach
    
    def _simulate_augmentation(self, prompt, context):
        """
        Simulate prompt augmentation with regulatory content.
        
        In a real implementation, this would use the EfficientRegulationAugmenter.
        """
        # Placeholder implementation - would actually use the augmenter
        return prompt, ["GDPR_Article_5", "HIPAA_Section_164"]
    
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
    
    def _initialize_specialist_models(self):
        """
        Initialize specialist models for different regulatory domains.
        
        In a real implementation, these would be actual model configurations
        or model identifier strings for the gateway.
        """
        # Just register model identifiers
        self.model_registry.models = {
            "GDPR": {
                "model": "gdpr_specialist",
                "metadata": {
                    "domain": "data_privacy",
                    "parameter_count": 350_000_000,
                    "quantization_bits": 8
                }
            },
            "HIPAA": {
                "model": "hipaa_specialist",
                "metadata": {
                    "domain": "health_privacy",
                    "parameter_count": 750_000_000,
                    "quantization_bits": 8
                }
            },
            "FINREG": {
                "model": "finreg_specialist",
                "metadata": {
                    "domain": "financial_compliance",
                    "parameter_count": 250_000_000,
                    "quantization_bits": 8
                }
            }
        }

# Factory function to create a gateway client from config
def create_gateway_client(config):
    """
    Create and configure an LLMGatewayClient from a configuration dict.
    
    Args:
        config: Gateway configuration dictionary
        
    Returns:
        Configured LLMGatewayClient instance
    """
    # Convert config dict to GatewayConfig object
    gateway_config = GatewayConfig(
        gateway_id=config.get("gateway_id", "default_gateway"),
        default_provider=config.get("default_provider", "openai"),
        default_model_identifier=config.get("default_model", "gpt-4"),
        max_retries=config.get("max_retries", 2),
        retry_delay_seconds=config.get("retry_delay_seconds", 1.0),
        default_timeout_seconds=config.get("timeout_seconds", 60.0),
        allowed_providers=config.get("allowed_providers", []),
        caching_enabled=config.get("caching_enabled", True),
        cache_default_ttl_seconds=config.get("cache_ttl_seconds", 3600),
        default_compliance_mode=config.get("default_compliance_mode", "strict"),
        logging_level=config.get("logging_level", "INFO")
    )
    
    # Create provider factory
    provider_factory = ProviderFactory()
    
    # Create and return client
    return LLMGatewayClient(gateway_config, provider_factory)