import datetime

class RulePerformanceTracker:
    """
    Tracks rule performance and usage statistics to guide rule adaptation.
    """
    def __init__(self, max_history=1000):
        self.rule_usage = {}  # Usage count by rule ID
        self.framework_usage = {}  # Usage count by framework ID
        self.rule_framework_map = {}  # Maps rule IDs to frameworks
        self.rule_events = []  # History of rule events
        self.max_history = max_history
        
        # Rule performance data
        self.rule_performance = {}  # Performance metrics by rule ID
        
    def record_rule_usage(self, framework_id, rules):
        """Record usage of rules for a framework"""
        # Update framework usage
        if framework_id not in self.framework_usage:
            self.framework_usage[framework_id] = 0
        self.framework_usage[framework_id] += 1
        
        # Update rule usage and mapping
        for rule in rules:
            rule_id = rule.get('id')
            if not rule_id:
                continue
                
            # Update rule usage count
            if rule_id not in self.rule_usage:
                self.rule_usage[rule_id] = 0
            self.rule_usage[rule_id] += 1
            
            # Update rule-framework mapping
            if rule_id not in self.rule_framework_map:
                self.rule_framework_map[rule_id] = []
            if framework_id not in self.rule_framework_map[rule_id]:
                self.rule_framework_map[rule_id].append(framework_id)
                
        # Add event to history
        event = {
            'type': 'usage',
            'framework_id': framework_id,
            'rule_count': len(rules),
            'timestamp': datetime.datetime.now().isoformat()
        }
        self._add_event(event)
    
    def record_rule_creation(self, rule_id, framework_id=None):
        """Record creation of a rule"""
        # Initialize rule in usage tracking
        if rule_id not in self.rule_usage:
            self.rule_usage[rule_id] = 0
            
        # Initialize rule-framework mapping
        if rule_id not in self.rule_framework_map:
            self.rule_framework_map[rule_id] = []
        if framework_id and framework_id not in self.rule_framework_map[rule_id]:
            self.rule_framework_map[rule_id].append(framework_id)
            
        # Initialize rule performance tracking
        if rule_id not in self.rule_performance:
            self.rule_performance[rule_id] = {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0
            }
            
        # Add event to history
        event = {
            'type': 'creation',
            'rule_id': rule_id,
            'framework_id': framework_id,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self._add_event(event)
    
    def record_rule_update(self, rule_id, framework_id=None):
        """Record update of a rule"""
        # Ensure rule exists in tracking
        if rule_id not in self.rule_usage:
            self.rule_usage[rule_id] = 0
            
        # Update rule-framework mapping if needed
        if rule_id not in self.rule_framework_map:
            self.rule_framework_map[rule_id] = []
        if framework_id and framework_id not in self.rule_framework_map[rule_id]:
            self.rule_framework_map[rule_id].append(framework_id)
            
        # Add event to history
        event = {
            'type': 'update',
            'rule_id': rule_id,
            'framework_id': framework_id,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self._add_event(event)
    
    def record_rule_deletion(self, rule_id, framework_id=None):
        """Record deletion of a rule"""
        # Add event to history
        event = {
            'type': 'deletion',
            'rule_id': rule_id,
            'framework_id': framework_id,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self._add_event(event)
        
        # Note: We don't remove rule from usage tracking to preserve history
    
    def record_rule_performance(self, rule_id, performance_data):
        """
        Record performance data for a rule
        
        Args:
            rule_id: ID of the rule
            performance_data: Dictionary with performance metrics
                - true_positives: Number of true positives
                - false_positives: Number of false positives
                - true_negatives: Number of true negatives
                - false_negatives: Number of false negatives
        """
        # Initialize rule performance if needed
        if rule_id not in self.rule_performance:
            self.rule_performance[rule_id] = {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0
            }
            
        # Update performance metrics
        for metric in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
            if metric in performance_data:
                self.rule_performance[rule_id][metric] += performance_data[metric]
                
        # Recalculate derived metrics
        self._recalculate_performance_metrics(rule_id)
        
        # Add event to history
        event = {
            'type': 'performance_update',
            'rule_id': rule_id,
            'performance_data': performance_data,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self._add_event(event)
    
    def _recalculate_performance_metrics(self, rule_id):
        """Recalculate derived performance metrics for a rule"""
        perf = self.rule_performance[rule_id]
        
        # Calculate precision
        if perf['true_positives'] + perf['false_positives'] > 0:
            perf['precision'] = perf['true_positives'] / (perf['true_positives'] + perf['false_positives'])
        else:
            perf['precision'] = 0.0
            
        # Calculate recall
        if perf['true_positives'] + perf['false_negatives'] > 0:
            perf['recall'] = perf['true_positives'] / (perf['true_positives'] + perf['false_negatives'])
        else:
            perf['recall'] = 0.0
            
        # Calculate F1 score
        if perf['precision'] + perf['recall'] > 0:
            perf['f1_score'] = 2 * (perf['precision'] * perf['recall']) / (perf['precision'] + perf['recall'])
        else:
            perf['f1_score'] = 0.0
            
        # Calculate accuracy
        total = perf['true_positives'] + perf['true_negatives'] + perf['false_positives'] + perf['false_negatives']
        if total > 0:
            perf['accuracy'] = (perf['true_positives'] + perf['true_negatives']) / total
        else:
            perf['accuracy'] = 0.0
    
    def _add_event(self, event):
        """Add event to history, managing size"""
        self.rule_events.append(event)
        
        # Trim history if needed
        if len(self.rule_events) > self.max_history:
            self.rule_events = self.rule_events[-self.max_history:]
    
    def get_rule_stats(self, rule_id):
        """Get statistics for a specific rule"""
        usage = self.rule_usage.get(rule_id, 0)
        frameworks = self.rule_framework_map.get(rule_id, [])
        performance = self.rule_performance.get(rule_id, {})
        
        # Get rule events
        events = [e for e in self.rule_events if e.get('rule_id') == rule_id]
        
        return {
            'rule_id': rule_id,
            'usage_count': usage,
            'frameworks': frameworks,
            'performance': performance,
            'events': events
        }
    
    def generate_report(self, framework_id=None):
        """Generate a report on rule performance"""
        if framework_id:
            # Framework-specific report
            rules = []
            for rule_id, frameworks in self.rule_framework_map.items():
                if framework_id in frameworks:
                    rules.append(rule_id)
                    
            return {
                'framework_id': framework_id,
                'usage_count': self.framework_usage.get(framework_id, 0),
                'rule_count': len(rules),
                'rules': {
                    rule_id: {
                        'usage_count': self.rule_usage.get(rule_id, 0),
                        'performance': self.rule_performance.get(rule_id, {})
                    }
                    for rule_id in rules
                },
                'top_rules': self._get_top_rules(rules, 5),
                'problematic_rules': self._get_problematic_rules(rules, 5)
            }
        else:
            # Overall report
            return {
                'total_rule_count': len(self.rule_usage),
                'total_framework_count': len(self.framework_usage),
                'top_frameworks': self._get_top_frameworks(5),
                'top_rules': self._get_top_rules(list(self.rule_usage.keys()), 5),
                'problematic_rules': self._get_problematic_rules(list(self.rule_usage.keys()), 5),
                'rule_type_distribution': self._get_rule_type_distribution()
            }
    
    def _get_top_rules(self, rule_ids, limit=5):
        """Get top-performing rules by usage"""
        # Sort rules by usage
        sorted_rules = sorted(
            [(rule_id, self.rule_usage.get(rule_id, 0)) for rule_id in rule_ids],
            key=lambda x: x[1], reverse=True
        )
        
        # Return top rules
        return [
            {
                'rule_id': rule_id,
                'usage_count': usage,
                'performance': self.rule_performance.get(rule_id, {})
            }
            for rule_id, usage in sorted_rules[:limit]
        ]
    
    def _get_problematic_rules(self, rule_ids, limit=5):
        """Get problematic rules (high false positives or negatives)"""
        problematic = []
        
        for rule_id in rule_ids:
            if rule_id not in self.rule_performance:
                continue
                
            perf = self.rule_performance[rule_id]
            
            # Rules with low F1 score or high false positives are problematic
            if (perf.get('f1_score', 0) < 0.5 and perf.get('true_positives', 0) + perf.get('false_positives', 0) > 0 or
                perf.get('false_positives', 0) > perf.get('true_positives', 0)):
                problematic.append({
                    'rule_id': rule_id,
                    'usage_count': self.rule_usage.get(rule_id, 0),
                    'performance': perf,
                    'problem_type': 'high_false_positives' if perf.get('false_positives', 0) > perf.get('true_positives', 0) else 'low_f1_score'
                })
                
        # Sort by problem severity (more false positives or lower F1 score)
        sorted_problematic = sorted(
            problematic,
            key=lambda x: (
                x['performance'].get('false_positives', 0) / max(1, x['performance'].get('true_positives', 1)),
                -x['performance'].get('f1_score', 0)
            ),
            reverse=True
        )
        
        return sorted_problematic[:limit]
    
    def _get_top_frameworks(self, limit=5):
        """Get top frameworks by usage"""
        # Sort frameworks by usage
        sorted_frameworks = sorted(
            self.framework_usage.items(),
            key=lambda x: x[1], reverse=True
        )
        
        # Return top frameworks
        return [
            {
                'framework_id': fw_id,
                'usage_count': usage,
                'rule_count': len([
                    rule_id for rule_id, frameworks in self.rule_framework_map.items()
                    if fw_id in frameworks
                ])
            }
            for fw_id, usage in sorted_frameworks[:limit]
        ]
    
    def _get_rule_type_distribution(self):
        """Get distribution of rule types"""
        # Count rules by type
        type_counts = {}
        
        for rule_id in self.rule_usage:
            # Determine rule type from events
            rule_events = [e for e in self.rule_events if e.get('rule_id') == rule_id]
            if not rule_events:
                continue
                
            # Find creation event to get rule type
            creation_events = [e for e in rule_events if e['type'] == 'creation']
            if not creation_events:
                continue
                
            # TODO: This would need to be enhanced to actually extract rule type
            # For now, we'll just use a placeholder
            rule_type = 'unknown'
            
            if rule_type not in type_counts:
                type_counts[rule_type] = 0
            type_counts[rule_type] += 1
            
        return type_counts


