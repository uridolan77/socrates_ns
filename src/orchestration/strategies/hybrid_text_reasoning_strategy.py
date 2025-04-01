
class HybridTextReasoningStrategy:
    """
    Reasoning strategy that processes text and symbolic representations in parallel.
    """
    def __init__(self, clmp, neural_symbolic_interface, base_coordinator, regulatory_knowledge_base, compliance_config):
        self.clmp = clmp
        self.interface = neural_symbolic_interface
        self.base_coordinator = base_coordinator
        self.regulatory_kb = regulatory_knowledge_base
        self.compliance_config = compliance_config
        
    def execute(self, query, context, applicable_frameworks, compliance_mode):
        """
        Execute the hybrid text-reasoning strategy with compliance verification.
        
        Args:
            query: Input text query
            context: Additional context information
            applicable_frameworks: Regulatory frameworks to apply
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Processed result with compliance information
        """
        # This is a parallel strategy with cross-validation
        
        # STAGE 1: Process in parallel
        # A. Language model generation
        text_task = self._process_text(query, context, compliance_mode)
        
        # B. Symbolic reasoning
        reasoning_task = self._process_symbolic(query, context, compliance_mode)
        
        # Combine results (in a real implementation, these would be async tasks)
        text_result = text_task
        reasoning_result = reasoning_task
        
        # STAGE 2: Check individual compliance
        if not text_result['is_compliant'] and not reasoning_result['is_compliant']:
            # Both paths have compliance issues
            return {
                'result': None,
                'text_result': None,
                'is_compliant': False,
                'compliance_error': "Both text and reasoning paths encountered compliance issues",
                'compliance_metadata': {
                    'text_compliance': text_result.get('compliance_metadata', {}),
                    'reasoning_compliance': reasoning_result.get('compliance_metadata', {})
                }
            }
        
        # STAGE 3: Cross-validate results
        validation_result = self._cross_validate(
            text_result,
            reasoning_result,
            applicable_frameworks,
            compliance_mode
        )
        
        if not validation_result['is_compliant']:
            return {
                'result': reasoning_result.get('result'),
                'text_result': text_result.get('text'),
                'is_compliant': False,
                'compliance_error': validation_result['error'],
                'compliance_metadata': validation_result.get('metadata', {})
            }
        
        # STAGE 4: Merge results if needed
        final_result = self._merge_results(
            text_result, 
            reasoning_result, 
            validation_result,
            compliance_mode
        )
        
        # STAGE 5: Verify final compliance
        overall_compliance = self._verify_overall_compliance(
            final_result, applicable_frameworks, compliance_mode
        )
        
        return {
            'result': final_result['result'],
            'text_result': final_result['text'],
            'is_compliant': overall_compliance['is_compliant'],
            'compliance_error': overall_compliance.get('error'),
            'compliance_metadata': {
                'text_compliance': text_result.get('compliance_metadata', {}),
                'reasoning_compliance': reasoning_result.get('compliance_metadata', {}),
                'validation_compliance': validation_result.get('metadata', {}),
                'overall_compliance': overall_compliance.get('metadata', {})
            },
            'reasoning_trace': reasoning_result.get('reasoning_trace'),
            'compliance_warnings': overall_compliance.get('warnings', [])
        }
    
    def execute_hybrid(self, text_query, symbolic_query, context, applicable_frameworks, compliance_mode):
        """
        Execute hybrid processing with both text and symbolic inputs.
        
        Args:
            text_query: Natural language component of query
            symbolic_query: Symbolic component of query
            context: Additional context information
            applicable_frameworks: Regulatory frameworks to apply
            compliance_mode: 'strict' or 'soft' enforcement
            
        Returns:
            Processed result with compliance information
        """
        # STAGE 1: Process in parallel using provided inputs
        # A. Language model generation from text query
        text_result = self._process_text(text_query, context, compliance_mode)
        
        # B. Symbolic reasoning from symbolic query
        reasoning_result = self.base_coordinator.process_query(
            symbolic_query, context, compliance_mode
        )
        
        # STAGE 2: Check individual compliance
        if not text_result['is_compliant'] and not reasoning_result['is_compliant']:
            # Both paths have compliance issues
            return {
                'result': None,
                'text_result': None,
                'is_compliant': False,
                'compliance_error': "Both text and reasoning paths encountered compliance issues",
                'compliance_metadata': {
                    'text_compliance': text_result.get('compliance_metadata', {}),
                    'reasoning_compliance': reasoning_result.get('compliance_metadata', {})
                }
            }
        
        # STAGE 3: Cross-validate results
        validation_result = self._cross_validate(
            text_result,
            reasoning_result,
            applicable_frameworks,
            compliance_mode
        )
        
        if not validation_result['is_compliant']:
            return {
                'result': reasoning_result.get('result'),
                'text_result': text_result.get('text'),
                'is_compliant': False,
                'compliance_error': validation_result['error'],
                'compliance_metadata': validation_result.get('metadata', {})
            }
        
        # STAGE 4: Merge results if needed
        final_result = self._merge_results(
            text_result, 
            reasoning_result, 
            validation_result,
            compliance_mode
        )
        
        # STAGE 5: Verify final compliance
        overall_compliance = self._verify_overall_compliance(
            final_result, applicable_frameworks, compliance_mode
        )
        
        return {
            'result': final_result['result'],
            'text_result': final_result['text'],
            'is_compliant': overall_compliance['is_compliant'],
            'compliance_error': overall_compliance.get('error'),
            'compliance_metadata': {
                'text_compliance': text_result.get('compliance_metadata', {}),
                'reasoning_compliance': reasoning_result.get('compliance_metadata', {}),
                'validation_compliance': validation_result.get('metadata', {}),
                'overall_compliance': overall_compliance.get('metadata', {})
            },
            'reasoning_trace': reasoning_result.get('reasoning_trace'),
            'compliance_warnings': overall_compliance.get('warnings', [])
        }
    
    def _process_text(self, query, context, compliance_mode):
        """Process query using language model path."""
        result = self.clmp.generate_compliant_text(
            query, context, compliance_mode=compliance_mode
        )
        
        return result
    
    def _process_symbolic(self, query, context, compliance_mode):
        """Process query using symbolic reasoning path."""
        # Convert to symbolic
        symbolic_result = self.interface.neural_to_symbolic_text(
            query, compliance_mode
        )
        
        if not symbolic_result['is_compliant']:
            return {
                'is_compliant': False,
                'compliance_error': symbolic_result['compliance_error'],
                'compliance_metadata': symbolic_result.get('compliance_metadata', {})
            }
        
        # Process with symbolic reasoning
        reasoning_result = self.base_coordinator.process_query(
            symbolic_result['symbolic_repr'], context, compliance_mode
        )
        
        return reasoning_result
    
    def _cross_validate(self, text_result, reasoning_result, applicable_frameworks, compliance_mode):
        """Cross-validate text and reasoning results for consistency and compliance."""
        # Convert text to symbolic for comparison
        if 'text' in text_result:
            text_symbolic = self.interface.neural_to_symbolic_text(
                text_result['text'], compliance_mode
            )
            
            if not text_symbolic['is_compliant']:
                return {
                    'is_compliant': False,
                    'error': "Text result contains non-compliant content",
                    'metadata': text_symbolic.get('compliance_metadata', {})
                }
            
            # Compare with reasoning result for consistency
            consistency = self._check_consistency(
                text_symbolic['symbolic_repr'], 
                reasoning_result.get('result'),
                compliance_mode
            )
            
            if not consistency['is_consistent']:
                return {
                    'is_compliant': False,
                    'error': "Text and reasoning results are inconsistent",
                    'metadata': consistency.get('metadata', {})
                }
        
        return {
            'is_compliant': True,
            'metadata': {}
        }
    
    def _check_consistency(self, text_symbolic, reasoning_result, compliance_mode):
        """Check consistency between symbolic representations."""
        # This would implement consistency checking
        # Placeholder implementation
        return {
            'is_consistent': True,
            'metadata': {}
        }
    
    def _merge_results(self, text_result, reasoning_result, validation_result, compliance_mode):
        """Merge text and reasoning results into final output."""
        # If one path is non-compliant, use the other
        if not text_result['is_compliant'] and reasoning_result['is_compliant']:
            # Generate text from reasoning result
            merged_text = self._generate_text_from_reasoning(
                reasoning_result['result'], compliance_mode
            )
            return {
                'result': reasoning_result['result'],
                'text': merged_text
            }
        elif text_result['is_compliant'] and not reasoning_result['is_compliant']:
            # Extract reasoning from text
            merged_reasoning = self._extract_reasoning_from_text(
                text_result['text'], compliance_mode
            )
            return {
                'result': merged_reasoning,
                'text': text_result['text']
            }
        else:
            # Both are compliant, use both
            return {
                'result': reasoning_result['result'],
                'text': text_result['text']
            }
    
    def _generate_text_from_reasoning(self, reasoning_result, compliance_mode):
        """Generate text from reasoning result."""
        # This would generate text from reasoning
        # Placeholder implementation
        return "Generated text from reasoning"
    
    def _extract_reasoning_from_text(self, text, compliance_mode):
        """Extract reasoning from text."""
        # This would extract reasoning from text
        # Placeholder implementation
        return {'extracted_reasoning': True}
    
    def _verify_overall_compliance(self, final_result, applicable_frameworks, compliance_mode):
        """Verify overall compliance of final result."""
        # Similar to implementation in other strategies
        # Placeholder implementation
        return {
            'is_compliant': True,
            'warnings': [],
            'metadata': {}
        }
