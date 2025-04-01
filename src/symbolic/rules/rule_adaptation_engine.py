import datetime
import re

class RuleAdaptationEngine:
    """
    Engine for adapting text pattern rules based on feedback and performance.
    """
    def __init__(self, config):
        self.config = config
        
    def generate_rules_from_text(self, regulation_text, framework_id):
        """
        Generate rules from regulatory text
        
        Args:
            regulation_text: Text of regulatory document
            framework_id: ID of the regulatory framework
            
        Returns:
            List of generated rules
        """
        # This is a placeholder implementation
        # A real implementation would use NLP techniques to extract rules
        
        # Check for common regulatory keywords and patterns
        rules = []
        
        # Generate rules for prohibited terms
        prohibited_terms = self._extract_prohibited_terms(regulation_text)
        for term in prohibited_terms:
            rule = {
                'id': f"gen_{framework_id}_prohibited_{len(rules)}",
                'name': f"Generated {term} Prohibition",
                'description': f"Generated rule prohibiting use of '{term}'",
                'type': 'text_pattern',
                'subtype': 'prohibited_term',
                'action': 'block',
                'pattern': f"\\b{re.escape(term)}\\b",
                'severity': 'medium',
                'source': 'generated',
                'framework_id': framework_id,
                'created_at': datetime.datetime.now().isoformat()
            }
            rules.append(rule)
            
        # Generate rules for required disclaimers
        disclaimers = self._extract_required_disclaimers(regulation_text)
        for disclaimer in disclaimers:
            rule = {
                'id': f"gen_{framework_id}_disclaimer_{len(rules)}",
                'name': f"Generated Disclaimer Requirement",
                'description': f"Generated rule requiring disclaimer",
                'type': 'text_pattern',
                'subtype': 'disclaimer_required',
                'action': 'require',
                'pattern': f"\\b{re.escape(disclaimer)}\\b",
                'severity': 'low',
                'source': 'generated',
                'framework_id': framework_id,
                'created_at': datetime.datetime.now().isoformat()
            }
            rules.append(rule)
            
        return rules
    
    def _extract_prohibited_terms(self, text):
        """Extract potentially prohibited terms from text"""
        # This is a placeholder implementation
        # A real implementation would use more sophisticated NLP
        
        # Look for patterns indicating prohibitions
        prohibition_indicators = [
            'prohibited', 'forbidden', 'not allowed', 'shall not', 
            'must not', 'may not', 'cannot', 'restricted'
        ]
        
        prohibited_terms = []
        
        for indicator in prohibition_indicators:
            # Find sentences containing prohibition indicator
            pattern = f"[^.]*\\b{re.escape(indicator)}\\b[^.]*\\."
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Extract sentence
                sentence = match.group(0)
                
                # Try to identify the prohibited term
                # This is a simplified approach
                words = sentence.split()
                for i, word in enumerate(words):
                    if indicator in word.lower():
                        # Look for nouns or noun phrases after the indicator
                        if i < len(words) - 1:
                            potential_term = words[i + 1]
                            # Clean up term
                            potential_term = re.sub(r'[^\w\s]', '', potential_term)
                            if potential_term and len(potential_term) > 3:
                                prohibited_terms.append(potential_term)
                                
        return prohibited_terms
    
    def _extract_required_disclaimers(self, text):
        """Extract potentially required disclaimers from text"""
        # This is a placeholder implementation
        # A real implementation would use more sophisticated NLP
        
        # Look for patterns indicating requirements
        requirement_indicators = [
            'required', 'must', 'shall', 'necessary', 
            'disclosure', 'disclose', 'inform', 'disclaimer'
        ]
        
        disclaimers = []
        
        for indicator in requirement_indicators:
            # Find sentences containing requirement indicator
            pattern = f"[^.]*\\b{re.escape(indicator)}\\b[^.]*\\."
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Extract sentence
                sentence = match.group(0)
                
                # Check if this might be a disclaimer requirement
                disclaimer_words = ['notice', 'disclaimer', 'disclosure', 'statement', 'inform']
                if any(word in sentence.lower() for word in disclaimer_words):
                    # Use a simplified approach to extract the most relevant part
                    # In a real implementation, this would use more sophisticated NLP
                    cleaned = re.sub(r'[^\w\s]', ' ', sentence)
                    words = cleaned.split()
                    if len(words) > 5:
                        # Take a chunk of words as potential disclaimer
                        disclaimer = ' '.join(words[2:min(len(words), 10)])
                        disclaimers.append(disclaimer)
                        
        return disclaimers
    
    def adapt_rules(self, feedback_data, dynamic_rules, static_rules, rule_stats):
        """
        Adapt rules based on feedback data
        
        Args:
            feedback_data: Dictionary with feedback on rule performance
            dynamic_rules: Dictionary of dynamic rules by framework
            static_rules: Dictionary of static rules by framework
            rule_stats: RulePerformanceTracker instance
            
        Returns:
            List of rule adaptations made
        """
        adaptations = []
        
        # Process rule-specific feedback
        if 'rule_feedback' in feedback_data:
            for rule_feedback in feedback_data['rule_feedback']:
                rule_id = rule_feedback.get('rule_id')
                if not rule_id:
                    continue
                    
                feedback_type = rule_feedback.get('type')
                
                if feedback_type == 'false_positive':
                    # Rule triggered incorrectly
                    adaptation = self._handle_false_positive(rule_id, rule_feedback, dynamic_rules, static_rules, rule_stats)
                    if adaptation:
                        adaptations.append(adaptation)
                elif feedback_type == 'false_negative':
                    # Rule failed to trigger
                    adaptation = self._handle_false_negative(rule_id, rule_feedback, dynamic_rules, static_rules, rule_stats)
                    if adaptation:
                        adaptations.append(adaptation)
                elif feedback_type == 'update_suggestion':
                    # Suggested update to rule
                    adaptation = self._handle_update_suggestion(rule_id, rule_feedback, dynamic_rules, static_rules)
                    if adaptation:
                        adaptations.append(adaptation)
                        
        # Process content feedback (for creating new rules)
        if 'content_feedback' in feedback_data:
            for content_feedback in feedback_data['content_feedback']:
                adaptation = self._handle_content_feedback(content_feedback, dynamic_rules, static_rules)
                if adaptation:
                    adaptations.append(adaptation)
                    
        # Process performance-based adaptations
        performance_adaptations = self._adapt_based_on_performance(rule_stats, dynamic_rules)
        adaptations.extend(performance_adaptations)
        
        return adaptations
    
    def _handle_false_positive(self, rule_id, feedback, dynamic_rules, static_rules, rule_stats):
        """Handle false positive feedback"""
        # Record performance data
        rule_stats.record_rule_performance(rule_id, {'false_positives': 1})
        
        # Find the rule
        rule = self._find_rule(rule_id, dynamic_rules, static_rules)
        if not rule:
            return None
            
        # Determine framework ID
        framework_id = feedback.get('framework_id')
        if not framework_id:
            # Try to determine from rule
            framework_ids = rule_stats.rule_framework_map.get(rule_id, [])
            if framework_ids:
                framework_id = framework_ids[0]
                
        # Get rule type
        rule_type = rule.get('type')
        
        if rule_type == 'text_pattern':
            # For text patterns, we can make the pattern more specific
            return self._make_text_pattern_more_specific(rule, rule_id, framework_id, feedback)
        elif rule_type == 'semantic':
            # For semantic rules, we can adjust the threshold
            return self._adjust_semantic_threshold(rule, rule_id, framework_id, feedback, increase=True)
        
        return None
    
    def _handle_false_negative(self, rule_id, feedback, dynamic_rules, static_rules, rule_stats):
        """Handle false negative feedback"""
        # Record performance data
        rule_stats.record_rule_performance(rule_id, {'false_negatives': 1})
        
        # Find the rule
        rule = self._find_rule(rule_id, dynamic_rules, static_rules)
        if not rule:
            return None
            
        # Determine framework ID
        framework_id = feedback.get('framework_id')
        if not framework_id:
            # Try to determine from rule
            framework_ids = rule_stats.rule_framework_map.get(rule_id, [])
            if framework_ids:
                framework_id = framework_ids[0]
                
        # Get rule type
        rule_type = rule.get('type')
        
        if rule_type == 'text_pattern':
            # For text patterns, we can make the pattern more general
            return self._make_text_pattern_more_general(rule, rule_id, framework_id, feedback)
        elif rule_type == 'semantic':
            # For semantic rules, we can adjust the threshold
            return self._adjust_semantic_threshold(rule, rule_id, framework_id, feedback, increase=False)
        
        return None
    
    def _handle_update_suggestion(self, rule_id, feedback, dynamic_rules, static_rules):
        """Handle update suggestion feedback"""
        # Find the rule
        rule = self._find_rule(rule_id, dynamic_rules, static_rules)
        if not rule:
            return None
            
        # Determine framework ID
        framework_id = feedback.get('framework_id')
        if not framework_id:
            # Try to determine from rule
            for fw_id, rules in dynamic_rules.items():
                for r in rules:
                    if r.get('id') == rule_id:
                        framework_id = fw_id
                        break
                        
        # Get suggested updates
        updates = feedback.get('updates', {})
        if not updates:
            return None
            
        # Create adaptation
        return {
            'type': 'update',
            'rule_id': rule_id,
            'framework_id': framework_id,
            'updates': updates,
            'reason': 'user_suggestion',
            'source': 'feedback'
        }
    
    def _handle_content_feedback(self, feedback, dynamic_rules, static_rules):
        """Handle content feedback to create new rules"""
        feedback_type = feedback.get('type')
        content = feedback.get('content')
        framework_id = feedback.get('framework_id')
        
        if not content:
            return None
            
        if feedback_type == 'should_block':
            # Content should be blocked but wasn't
            # Create a new rule to block similar content
            return self._create_rule_to_block_content(content, framework_id)
        elif feedback_type == 'should_not_block':
            # Content was incorrectly blocked
            # Could create exception or modify existing rules
            return self._create_exception_for_content(content, framework_id, dynamic_rules, static_rules)
            
        return None
    
    def _adapt_based_on_performance(self, rule_stats, dynamic_rules):
        """Adapt rules based on performance statistics"""
        adaptations = []
        
        # Get problematic rules
        for framework_id in dynamic_rules:
            # Generate report for this framework
            report = rule_stats.generate_report(framework_id)
            
            # Check for problematic rules
            for rule_info in report.get('problematic_rules', []):
                rule_id = rule_info.get('rule_id')
                if not rule_id:
                    continue
                    
                problem_type = rule_info.get('problem_type')
                
                if problem_type == 'high_false_positives':
                    # Rule has too many false positives
                    # Find rule in dynamic rules
                    rule = None
                    for r in dynamic_rules.get(framework_id, []):
                        if r.get('id') == rule_id:
                            rule = r
                            break
                            
                    if rule:
                        # Adjust rule to reduce false positives
                        if rule.get('type') == 'text_pattern':
                            # Make pattern more specific
                            adaptation = self._make_text_pattern_more_specific(rule, rule_id, framework_id, None)
                            if adaptation:
                                adaptations.append(adaptation)
                        elif rule.get('type') == 'semantic':
                            # Increase threshold
                            adaptation = self._adjust_semantic_threshold(rule, rule_id, framework_id, None, increase=True)
                            if adaptation:
                                adaptations.append(adaptation)
                                
        return adaptations
    
    def _find_rule(self, rule_id, dynamic_rules, static_rules):
        """Find a rule by ID in dynamic or static rules"""
        # Check dynamic rules
        for framework_id, rules in dynamic_rules.items():
            for rule in rules:
                if rule.get('id') == rule_id:
                    return rule
                    
        # Check static rules
        for framework_id, rules in static_rules.items():
            for rule in rules:
                if rule.get('id') == rule_id:
                    return rule
                    
        return None
    
    def _make_text_pattern_more_specific(self, rule, rule_id, framework_id, feedback):
        """Make a text pattern rule more specific to reduce false positives"""
        pattern = rule.get('pattern')
        if not pattern:
            return None
            
        # Make the pattern more specific
        updated_pattern = pattern
        
        if feedback and 'content' in feedback:
            # Use feedback content to guide specificity
            content = feedback.get('content', '')
            matched_text = feedback.get('matched_text', '')
            
            if matched_text and matched_text in content:
                # Look at context around matched text
                before, after = content.split(matched_text, 1)
                
                # Get a few words before and after
                before_words = before.strip().split()[-3:]
                after_words = after.strip().split()[:3]
                
                if before_words:
                    # Add word before as context
                    context_word = re.escape(before_words[-1])
                    updated_pattern = f"\\b{context_word}\\s+{pattern}"
                elif after_words:
                    # Add word after as context
                    context_word = re.escape(after_words[0])
                    updated_pattern = f"{pattern}\\s+{context_word}\\b"
        else:
            # Without specific feedback, make general adjustments
            if '\\b' not in pattern:
                # Add word boundaries if not present
                updated_pattern = f"\\b{pattern}\\b"
                
        # Only create adaptation if pattern was changed
        if updated_pattern != pattern:
            return {
                'type': 'update',
                'rule_id': rule_id,
                'framework_id': framework_id,
                'updates': {'pattern': updated_pattern},
                'reason': 'reduce_false_positives',
                'source': 'performance_adaptation'
            }
            
        return None
    
    def _make_text_pattern_more_general(self, rule, rule_id, framework_id, feedback):
        """Make a text pattern rule more general to reduce false negatives"""
        pattern = rule.get('pattern')
        if not pattern:
            return None
            
        # Make the pattern more general
        updated_pattern = pattern
        
        if feedback and 'content' in feedback:
            # Use feedback content to guide generalization
            content = feedback.get('content', '')
            missed_text = feedback.get('missed_text', '')
            
            if missed_text:
                # Calculate similarity between pattern and missed text
                import difflib
                similarity = difflib.SequenceMatcher(None, pattern, missed_text).ratio()
                
                if similarity > 0.5:
                    # Texts are somewhat similar, create a more general pattern
                    # This is a simplified approach
                    # A real implementation would use more sophisticated pattern generalization
                    common_substring = difflib.SequenceMatcher(None, pattern, missed_text).find_longest_match(
                        0, len(pattern), 0, len(missed_text)
                    )
                    
                    if common_substring.size > 3:
                        # Use common substring as base for new pattern
                        common_text = pattern[common_substring.a:common_substring.a + common_substring.size]
                        updated_pattern = common_text
        else:
            # Without specific feedback, make general adjustments
            if '\\b' in pattern:
                # Remove some word boundaries to make more general
                updated_pattern = pattern.replace('\\b', '', 1)
                
        # Only create adaptation if pattern was changed
        if updated_pattern != pattern:
            return {
                'type': 'update',
                'rule_id': rule_id,
                'framework_id': framework_id,
                'updates': {'pattern': updated_pattern},
                'reason': 'reduce_false_negatives',
                'source': 'performance_adaptation'
            }
            
        return None
    
    def _adjust_semantic_threshold(self, rule, rule_id, framework_id, feedback, increase=True):
        """Adjust threshold for semantic rule"""
        current_threshold = rule.get('threshold')
        if current_threshold is None:
            return None
            
        # Convert to float if necessary
        try:
            current_threshold = float(current_threshold)
        except (ValueError, TypeError):
            return None
            
        # Adjust threshold
        adjustment = 0.1
        if increase:
            # Increasing threshold reduces false positives
            new_threshold = min(0.9, current_threshold + adjustment)
        else:
            # Decreasing threshold reduces false negatives
            new_threshold = max(0.1, current_threshold - adjustment)
            
        # Only create adaptation if threshold was changed
        if new_threshold != current_threshold:
            return {
                'type': 'update',
                'rule_id': rule_id,
                'framework_id': framework_id,
                'updates': {'threshold': new_threshold},
                'reason': 'adjust_threshold',
                'source': 'performance_adaptation'
            }
            
        return None
    
    def _create_rule_to_block_content(self, content, framework_id):
        """Create a new rule to block content similar to the provided content"""
        # Extract key phrases from content
        key_phrases = self._extract_key_phrases(content)
        
        if not key_phrases:
            return None
            
        # Use the first key phrase as basis for new rule
        key_phrase = key_phrases[0]
        
        # Create a new rule
        return {
            'type': 'create',
            'template_name': 'prohibited_term',
            'parameters': {
                'term': re.escape(key_phrase),
                'name': f"Generated rule for '{key_phrase}'",
                'description': f"Generated rule to block content containing '{key_phrase}'",
                'severity': 'medium'
            },
            'framework_id': framework_id,
            'reason': 'user_feedback',
            'source': 'content_feedback'
        }
    
    def _create_exception_for_content(self, content, framework_id, dynamic_rules, static_rules):
        """Create exception for incorrectly blocked content"""
        # This is a more complex adaptation that would require
        # identifying which rule(s) incorrectly blocked the content
        # and creating exceptions for them
        
        # Placeholder implementation - would need to be expanded
        return None
    
    def _extract_key_phrases(self, text):
        """Extract key phrases from text"""
        # This is a placeholder implementation
        # A real implementation would use NLP techniques
        
        # Simple approach: extract noun phrases
        words = text.split()
        key_phrases = []
        
        for i in range(len(words)):
            phrase = words[i]
            if len(phrase) > 3 and phrase[0].isupper():
                key_phrases.append(phrase)
                
        # If no key phrases found, just use the longest word
        if not key_phrases and words:
            longest_word = max(words, key=len)
            if len(longest_word) > 3:
                key_phrases.append(longest_word)
                
        return key_phrases
