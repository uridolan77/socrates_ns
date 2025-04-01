
from src.orchestration.strategies.text_to_reasoning_strategy import TextToReasoningStrategy
from src.orchestration.strategies.hybrid_text_reasoning_strategy import HybridTextReasoningStrategy
from src.orchestration.strategies.iterative_text_refinement_strategy import IterativeTextRefinementStrategy
from src.orchestration.strategies.reasoning_to_text_strategy import ReasoningToTextStrategy
from src.orchestration.language_model_meta_reasoner import LanguageModelMetaReasoner

class LanguageModelReasoningCoordinator:
    """
    Extends the Reasoning Coordinator to support language model integration
    with compliance verification throughout the reasoning process.
    """
    def __init__(self, base_coordinator, clmp, neural_symbolic_interface, regulatory_knowledge_base, compliance_config):
        """
        Initialize the language model reasoning coordinator.
        
        Args:
            base_coordinator: Base reasoning coordinator from SocratesNS
            clmp: Compliant Language Model Processor
            neural_symbolic_interface: Neural-symbolic interface
            regulatory_knowledge_base: Repository of regulatory frameworks
            compliance_config: Configuration for compliance enforcement
        """
        self.base_coordinator = base_coordinator
        self.clmp = clmp
        self.interface = neural_symbolic_interface
        self.regulatory_kb = regulatory_knowledge_base
        self.compliance_config = compliance_config
        
        # Initialize language-specific reasoning strategies
        self.language_strategies = {
            'text_to_reasoning': TextToReasoningStrategy(
                clmp, neural_symbolic_interface, base_coordinator, regulatory_knowledge_base, compliance_config
            ),
            'reasoning_to_text': ReasoningToTextStrategy(
                clmp, neural_symbolic_interface, base_coordinator, regulatory_knowledge_base, compliance_config
            ),
            'hybrid_text_reasoning': HybridTextReasoningStrategy(
                clmp, neural_symbolic_interface, base_coordinator, regulatory_knowledge_base, compliance_config
            ),
            'iterative_text_refinement': IterativeTextRefinementStrategy(
                clmp, neural_symbolic_interface, base_coordinator, regulatory_knowledge_base, compliance_config
            )
        }
        
        # Meta-reasoner for strategy selection
        self.meta_reasoner = LanguageModelMetaReasoner(compliance_config)
        
    def process_text_query(self, query, context=None, compliance_mode='strict'):
        """
        Process a text query using the optimal reasoning strategy with compliance verification.
        
        Args:
            query: Input text query
            context: Additional context information
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Processed result with compliance information
        """
        # Verify query compliance with applicable regulations
        query_compliance = self.clmp.compliance_verifier.verify_content(
            query, content_type="query", compliance_mode=compliance_mode
        )
        
        if not query_compliance['is_compliant']:
            return {
                'result': None,
                'compliance_error': query_compliance['error'],
                'compliance_metadata': query_compliance['metadata']
            }
        
        # Determine applicable regulatory frameworks
        applicable_frameworks = self.regulatory_kb.get_applicable_frameworks(
            query, context, compliance_mode
        )
        
        # Determine query type and suitable reasoning strategies
        query_analysis = self._analyze_query(query, context)
        
        # Select optimal reasoning strategy with compliance awareness
        strategy_result = self.meta_reasoner.select_reasoning_strategy(
            query,
            query_analysis,
            context,
            applicable_frameworks,
            compliance_mode
        )
        
        if not strategy_result['is_compliant']:
            return {
                'result': None,
                'compliance_error': strategy_result['error'],
                'compliance_metadata': strategy_result['metadata']
            }
        
        selected_strategy = strategy_result['strategy']
        
        # Execute selected reasoning strategy with compliance verification
        if selected_strategy == 'text_to_reasoning':
            result = self.language_strategies['text_to_reasoning'].execute(
                query, context, applicable_frameworks, compliance_mode
            )
        elif selected_strategy == 'reasoning_to_text':
            result = self.language_strategies['reasoning_to_text'].execute(
                query, context, applicable_frameworks, compliance_mode
            )
        elif selected_strategy == 'hybrid_text_reasoning':
            result = self.language_strategies['hybrid_text_reasoning'].execute(
                query, context, applicable_frameworks, compliance_mode
            )
        elif selected_strategy == 'iterative_text_refinement':
            result = self.language_strategies['iterative_text_refinement'].execute(
                query, context, applicable_frameworks, compliance_mode
            )
        else:
            # Fall back to a default strategy if none matched
            result = self._execute_fallback_strategy(
                query, context, applicable_frameworks, compliance_mode
            )
        
        # Generate final explanation if needed
        if not result['is_compliant'] or result.get('compliance_warnings', []):
            explanation = self._generate_compliance_explanation(
                result, query, context, compliance_mode
            )
            result['compliance_explanation'] = explanation
        
        return result
    
    def process_hybrid_query(self, text_query, symbolic_query, context=None, compliance_mode='strict'):
        """
        Process a hybrid query with both text and symbolic components.
        
        Args:
            text_query: Natural language component of query
            symbolic_query: Symbolic component of query
            context: Additional context information
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Processed result with compliance information
        """
        # Verify text query compliance
        text_compliance = self.clmp.compliance_verifier.verify_content(
            text_query, content_type="query", compliance_mode=compliance_mode
        )
        
        if not text_compliance['is_compliant']:
            return {
                'result': None,
                'compliance_error': text_compliance['error'],
                'compliance_metadata': text_compliance['metadata']
            }
        
        # Verify symbolic query compliance
        symbolic_compliance = self.base_coordinator.compliance_verifier.verify_query(
            symbolic_query, compliance_mode
        )
        
        if not symbolic_compliance['is_compliant']:
            return {
                'result': None,
                'compliance_error': symbolic_compliance['error'],
                'compliance_metadata': symbolic_compliance['metadata']
            }
        
        # Determine applicable regulatory frameworks (combining both sources)
        text_frameworks = self.regulatory_kb.get_applicable_frameworks(
            text_query, context, compliance_mode
        )
        
        symbolic_frameworks = self.base_coordinator.regulatory_kb.get_applicable_frameworks(
            symbolic_query, context, compliance_mode
        )
        
        # Combine frameworks
        applicable_frameworks = list(set(text_frameworks + symbolic_frameworks))
        
        # Use hybrid strategy
        result = self.language_strategies['hybrid_text_reasoning'].execute_hybrid(
            text_query, symbolic_query, context, applicable_frameworks, compliance_mode
        )
        
        # Generate final explanation if needed
        if not result['is_compliant'] or result.get('compliance_warnings', []):
            explanation = self._generate_compliance_explanation(
                result, text_query, context, compliance_mode
            )
            result['compliance_explanation'] = explanation
        
        return result
    
    def _analyze_query(self, query, context):
        """
        Analyze the type of query to determine suitable reasoning strategies.
        
        Args:
            query: Input text query
            context: Additional context information
            
        Returns:
            Query analysis result
        """
        # This would implement deeper query analysis
        # Placeholder implementation
        return {
            'query_type': 'language_generation',
            'complexity': 'medium',
            'regulatory_relevance': 'high',
            'requires_symbolic_reasoning': True,
            'requires_language_generation': True,
            'domain': context.get('domain') if context else None
        }
    
    def _execute_fallback_strategy(self, query, context, applicable_frameworks, compliance_mode):
        """
        Execute a fallback strategy when no specific strategy is selected.
        
        Args:
            query: Input text query
            context: Additional context information
            applicable_frameworks: Regulatory frameworks to apply
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Processed result with compliance information
        """
        # Use the most conservative strategy for fallback
        return self.language_strategies['text_to_reasoning'].execute(
            query, context, applicable_frameworks, compliance_mode
        )
    
    def _generate_compliance_explanation(self, result, query, context, compliance_mode):
        """
        Generate a human-readable explanation of compliance verification results.
        
        Args:
            result: Result of compliance verification
            query: Original query
            context: Additional context information
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Human-readable explanation
        """
        # Convert compliance result to symbolic representation
        symbolic_explanation = self.interface.compliance_to_symbolic(result)
        
        # Generate natural language explanation
        natural_language_explanation = self.interface.symbolic_to_natural_language_explanation(
            symbolic_explanation
        )
        
        return natural_language_explanation
