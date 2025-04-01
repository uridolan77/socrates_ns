
import src.rules.rule_performance_tracker as RulePerformanceTracker
import src.rules.semantic_rule_adaptation_engine as SemanticRuleAdaptationEngine
import logging
import uuid
import datetime

class DynamicSemanticRules:
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
