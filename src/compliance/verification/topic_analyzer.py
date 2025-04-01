
import uuid
from src.compliance.models.compliance_issue import ComplianceIssue

class ComplianceTopicAnalyzer:
    """
    Analyzes text for sensitive topics that may require compliance review.
    """
    def __init__(self, config):
        self.config = config
        self.sensitive_topics = config.get("sensitive_topics", [])
        self.topic_threshold = config.get("topic_threshold", 0.6)
        
    def analyze_topics(self, text, context=None):
        """
        Analyze text for sensitive topics.
        
        Args:
            text: Input text to analyze
            context: Optional context information
            
        Returns:
            Dict with topic analysis results
        """
        if not text or not self.sensitive_topics:
            return {"is_compliant": True, "filtered_input": text, "issues": []}
            
        issues = []
        is_compliant = True
        
        # Simple keyword-based topic detection
        # In a real system, this would likely use a more sophisticated ML model
        for topic in self.sensitive_topics:
            topic_id = topic.get("id", str(uuid.uuid4()))
            keywords = topic.get("keywords", [])
            threshold = topic.get("threshold", self.topic_threshold)
            
            # Count keyword occurrences
            keyword_count = sum(text.lower().count(keyword.lower()) for keyword in keywords)
            
            # Crude calculation of topic relevance
            topic_score = min(1.0, keyword_count / max(10, len(text.split()) / 5))
            
            if topic_score >= threshold:
                is_compliant = False
                issues.append(ComplianceIssue(
                    rule_id=topic_id,
                    severity=topic.get("severity", "medium"),
                    description=f"Sensitive topic detected: {topic.get('name', 'Unnamed')}",
                    metadata={"topic_score": topic_score, "topic_name": topic.get("name", "")}
                ))
        
        return {
            "is_compliant": is_compliant,
            "filtered_input": text,
            "issues": [issue.__dict__ for issue in issues],
            "modified": False
        }