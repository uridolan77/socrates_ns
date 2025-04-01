import asyncio
import json
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("compliance.monitor")


class ContentEventType(Enum):
    """Types of content events in the system"""
    CONTENT_CREATED = "content_created"
    CONTENT_UPDATED = "content_updated"
    CONTENT_PUBLISHED = "content_published"
    CONTENT_ARCHIVED = "content_archived"
    CONTENT_DELETED = "content_deleted"


class ComplianceEventType(Enum):
    """Types of compliance-related events"""
    COMPLIANCE_VERIFIED = "compliance_verified"
    COMPLIANCE_VIOLATION = "compliance_violation"
    FRAMEWORK_UPDATED = "framework_updated"
    RULE_UPDATED = "rule_updated"
    REMEDIATION_APPLIED = "remediation_applied"
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"


@dataclass
class ContentEvent:
    """Event representing a content lifecycle action"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: ContentEventType = ContentEventType.CONTENT_CREATED
    content_id: str = ""
    content_type: str = "text"
    user_id: str = ""
    timestamp: float = field(default_factory=time.time)
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceEvent:
    """Event representing a compliance-related action"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: ComplianceEventType = ComplianceEventType.COMPLIANCE_VERIFIED
    content_id: str = ""
    framework_ids: List[str] = field(default_factory=list)
    is_compliant: bool = True
    user_id: str = ""
    timestamp: float = field(default_factory=time.time)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    Simple event bus for publishing and subscribing to events.
    In a production system, this would be replaced with a robust
    message broker like Kafka, RabbitMQ, AWS SNS/SQS, etc.
    """
    
    def __init__(self):
        self.subscribers = {}
        
    async def publish(self, topic: str, event: Any):
        """
        Publish an event to a topic
        
        Args:
            topic: Topic to publish to
            event: Event data to publish
        """
        if topic not in self.subscribers:
            return
            
        tasks = []
        for subscriber in self.subscribers[topic]:
            # Convert dataclass to dict if needed
            event_data = asdict(event) if hasattr(event, "__dataclass_fields__") else event
            tasks.append(asyncio.create_task(subscriber(event_data)))
            
        if tasks:
            await asyncio.gather(*tasks)
            
    def subscribe(self, topic: str, callback: Callable[[Any], Awaitable[None]]):
        """
        Subscribe to a topic
        
        Args:
            topic: Topic to subscribe to
            callback: Async function to call when an event is published
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []
            
        self.subscribers[topic].append(callback)
        return len(self.subscribers[topic]) - 1
        
    def unsubscribe(self, topic: str, subscription_id: int):
        """
        Unsubscribe from a topic
        
        Args:
            topic: Topic to unsubscribe from
            subscription_id: ID of the subscription to remove
        """
        if topic in self.subscribers and 0 <= subscription_id < len(self.subscribers[topic]):
            self.subscribers[topic].pop(subscription_id)


class ComplianceMonitor:
    """
    Event-driven compliance monitoring system that listens to content
    events and triggers compliance verification.
    """
    
    def __init__(self, 
                event_bus: EventBus,
                compliance_verifier: Any,
                config: Dict[str, Any] = None):
        """
        Initialize the compliance monitor
        
        Args:
            event_bus: Event bus for publishing/subscribing to events
            compliance_verifier: Verifier for checking compliance
            config: Configuration options
        """
        self.event_bus = event_bus
        self.verifier = compliance_verifier
        self.config = config or {}
        
        # Subscribe to content events
        self.event_bus.subscribe("content", self.handle_content_event)
        
        # Configure monitoring
        self.applicable_frameworks = self.config.get("frameworks", ["GDPR", "HIPAA", "CCPA"])
        self.verification_mode = self.config.get("verification_mode", "strict")
        self.sample_rate = self.config.get("sample_rate", 1.0)  # Percentage of content to verify
        
        logger.info(f"Compliance monitor initialized with frameworks: {self.applicable_frameworks}")
    
    async def handle_content_event(self, event_data: Dict[str, Any]):
        """
        Handle a content event by verifying compliance
        
        Args:
            event_data: Content event data
        """
        try:
            # Convert dict back to ContentEvent
            event_type = ContentEventType(event_data["event_type"])
            
            # Check if we should verify this content based on event type and sampling
            if not self._should_verify_content(event_type):
                return
                
            content_id = event_data["content_id"]
            content = event_data.get("content")
            
            if not content:
                logger.warning(f"Content missing in event: {event_data['id']}")
                return
                
            # Determine which frameworks apply based on content metadata
            applicable_frameworks = self._determine_applicable_frameworks(event_data)
            
            logger.info(f"Verifying compliance for content {content_id} against {applicable_frameworks}")
            
            # Verify compliance
            verification_result = self.verifier.verify_content(
                content,
                content_type=event_data.get("content_type", "text"),
                frameworks=applicable_frameworks,
                compliance_mode=self.verification_mode
            )
            
            # Create compliance event
            compliance_event = ComplianceEvent(
                event_type=(ComplianceEventType.COMPLIANCE_VERIFIED 
                          if verification_result["is_compliant"] 
                          else ComplianceEventType.COMPLIANCE_VIOLATION),
                content_id=content_id,
                framework_ids=applicable_frameworks,
                is_compliant=verification_result["is_compliant"],
                user_id=event_data.get("user_id", ""),
                violations=verification_result.get("violations", []),
                metadata={
                    "content_type": event_data.get("content_type", "text"),
                    "compliance_score": verification_result.get("compliance_score", 1.0),
                    "original_event_id": event_data["id"]
                }
            )
            
            # Publish compliance event
            await self.event_bus.publish("compliance", compliance_event)
            
            # If violations detected, also publish to violations topic
            if not verification_result["is_compliant"]:
                await self.event_bus.publish("violations", compliance_event)
                
            logger.info(f"Compliance verification completed for content {content_id}: " 
                      f"{'Compliant' if verification_result['is_compliant'] else 'Non-compliant'}")
                
        except Exception as e:
            logger.error(f"Error handling content event: {str(e)}")
    
    def _should_verify_content(self, event_type: ContentEventType) -> bool:
        """
        Determine if we should verify content based on event type and sampling
        
        Args:
            event_type: Type of content event
            
        Returns:
            True if content should be verified, False otherwise
        """
        # Always verify published content
        if event_type == ContentEventType.CONTENT_PUBLISHED:
            return True
            
        # For other event types, apply sampling
        if event_type in [ContentEventType.CONTENT_CREATED, ContentEventType.CONTENT_UPDATED]:
            import random
            return random.random() < self.sample_rate
            
        # Don't verify archived or deleted content
        return False
    
    def _determine_applicable_frameworks(self, event_data: Dict[str, Any]) -> List[str]:
        """
        Determine which frameworks apply to this content
        
        Args:
            event_data: Content event data
            
        Returns:
            List of applicable framework IDs
        """
        # Get content metadata
        metadata = event_data.get("metadata", {})
        
        # Check if specific frameworks are requested
        if "frameworks" in metadata:
            requested_frameworks = metadata["frameworks"]
            # Filter to only include supported frameworks
            return [fw for fw in requested_frameworks if fw in self.applicable_frameworks]
            
        # Check content domain/category to determine appropriate frameworks
        domain = metadata.get("domain")
        if domain == "healthcare":
            return [fw for fw in self.applicable_frameworks if fw in ["HIPAA", "GDPR"]]
        elif domain == "finance":
            return [fw for fw in self.applicable_frameworks if fw in ["GLBA", "GDPR", "PCI-DSS"]]
        elif domain == "consumer":
            return [fw for fw in self.applicable_frameworks if fw in ["CCPA", "GDPR"]]
            
        # Default to all configured frameworks
        return self.applicable_frameworks


class ComplianceReportGenerator:
    """
    Generates compliance reports based on compliance events.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the report generator
        
        Args:
            event_bus: Event bus for subscribing to events
        """
        self.event_bus = event_bus
        self.event_bus.subscribe("compliance", self.handle_compliance_event)
        
        # Store recent events for reporting
        self.recent_events = []
        self.max_events = 1000
        
    async def handle_compliance_event(self, event_data: Dict[str, Any]):
        """
        Handle a compliance event by storing it for reporting
        
        Args:
            event_data: Compliance event data
        """
        # Store event for reporting
        self.recent_events.append(event_data)
        
        # Trim to max size
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]
            
    def generate_report(self, start_time: Optional[float] = None, 
                       end_time: Optional[float] = None,
                       frameworks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a compliance report for a time period
        
        Args:
            start_time: Start time for report period (None for all)
            end_time: End time for report period (None for all)
            frameworks: List of frameworks to include (None for all)
            
        Returns:
            Compliance report data
        """
        # Filter events by time period
        filtered_events = self.recent_events
        
        if start_time is not None:
            filtered_events = [e for e in filtered_events if e["timestamp"] >= start_time]
            
        if end_time is not None:
            filtered_events = [e for e in filtered_events if e["timestamp"] <= end_time]
            
        # Filter by frameworks if specified
        if frameworks:
            filtered_events = [
                e for e in filtered_events 
                if any(fw in e["framework_ids"] for fw in frameworks)
            ]
            
        # Calculate compliance statistics
        total_events = len(filtered_events)
        compliant_events = sum(1 for e in filtered_events if e["is_compliant"])
        compliance_rate = compliant_events / total_events if total_events > 0 else 1.0
        
        # Count violations by rule
        violation_counts = {}
        for event in filtered_events:
            if not event["is_compliant"]:
                for violation in event["violations"]:
                    rule_id = violation.get("rule_id", "unknown")
                    if rule_id not in violation_counts:
                        violation_counts[rule_id] = 0
                    violation_counts[rule_id] += 1
                    
        # Generate report
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": time.time(),
            "start_time": start_time,
            "end_time": end_time,
            "frameworks": frameworks,
            "total_events": total_events,
            "compliant_events": compliant_events,
            "non_compliant_events": total_events - compliant_events,
            "compliance_rate": compliance_rate,
            "violation_summary": {
                "total_violations": sum(violation_counts.values()),
                "unique_rules": len(violation_counts),
                "top_violations": sorted(
                    [{"rule_id": rule, "count": count} 
                     for rule, count in violation_counts.items()],
                    key=lambda x: x["count"],
                    reverse=True
                )[:10]  # Top 10 violations
            }
        }
        
        return report


class ComplianceAlertHandler:
    """
    Handles alerts for compliance violations.
    """
    
    def __init__(self, 
                event_bus: EventBus,
                notification_config: Dict[str, Any] = None):
        """
        Initialize the alert handler
        
        Args:
            event_bus: Event bus for subscribing to events
            notification_config: Configuration for notifications
        """
        self.event_bus = event_bus
        self.event_bus.subscribe("violations", self.handle_violation_event)
        
        self.notification_config = notification_config or {}
        
        # Configure alert thresholds
        self.severity_thresholds = self.notification_config.get("severity_thresholds", {
            "high": 1,      # Alert on any high severity violation
            "medium": 3,    # Alert after 3 medium severity violations
            "low": 10       # Alert after 10 low severity violations
        })
        
        # Track violation counts for alerting
        self.violation_counts = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        logger.info("Compliance alert handler initialized")
    
    async def handle_violation_event(self, event_data: Dict[str, Any]):
        """
        Handle a compliance violation event
        
        Args:
            event_data: Compliance event data with violations
        """
        # Count violations by severity
        for violation in event_data["violations"]:
            severity = violation.get("severity", "medium").lower()
            if severity in self.violation_counts:
                self.violation_counts[severity] += 1
                
        # Check thresholds and send alerts
        alerts_sent = []
        
        for severity, count in self.violation_counts.items():
            threshold = self.severity_thresholds.get(severity, 999)
            if count >= threshold:
                # Send alert
                alert_info = {
                    "severity": severity,
                    "violation_count": count,
                    "content_id": event_data["content_id"],
                    "frameworks": event_data["framework_ids"]
                }
                
                await self._send_alert(alert_info)
                alerts_sent.append(severity)
                
                # Reset counter
                self.violation_counts[severity] = 0
                
        if alerts_sent:
            logger.info(f"Sent compliance alerts for severity levels: {', '.join(alerts_sent)}")
    
    async def _send_alert(self, alert_info: Dict[str, Any]):
        """
        Send an alert through configured channels
        
        Args:
            alert_info: Information about the alert
        """
        # Get notification channels
        channels = self.notification_config.get("channels", ["log"])
        
        for channel in channels:
            if channel == "log":
                # Log the alert
                logger.warning(
                    f"COMPLIANCE ALERT: {alert_info['severity']} severity, "
                    f"{alert_info['violation_count']} violations for content {alert_info['content_id']}"
                )
            elif channel == "email":
                # Send email alert
                await self._send_email_alert(alert_info)
            elif channel == "slack":
                # Send Slack alert
                await self._send_slack_alert(alert_info)
            elif channel == "webhook":
                # Send webhook alert
                await self._send_webhook_alert(alert_info)
    
    async def _send_email_alert(self, alert_info: Dict[str, Any]):
        """Send email alert (placeholder implementation)"""
        # In a real implementation, this would send an actual email
        recipients = self.notification_config.get("email_recipients", [])
        logger.info(f"Would send email alert to {recipients}: {alert_info}")
    
    async def _send_slack_alert(self, alert_info: Dict[str, Any]):
        """Send Slack alert (placeholder implementation)"""
        # In a real implementation, this would post to a Slack webhook
        slack_webhook = self.notification_config.get("slack_webhook")
        logger.info(f"Would send Slack alert to {slack_webhook}: {alert_info}")
    
    async def _send_webhook_alert(self, alert_info: Dict[str, Any]):
        """Send webhook alert (placeholder implementation)"""
        # In a real implementation, this would post to a webhook
        webhook_url = self.notification_config.get("webhook_url")
        logger.info(f"Would send webhook alert to {webhook_url}: {alert_info}")


async def main():
    """Example usage of the event-driven compliance monitoring system"""
    # Create a mock compliance verifier
    class MockVerifier:
        def verify_content(self, content, content_type="text", frameworks=None, compliance_mode="strict"):
            frameworks = frameworks or ["GDPR"]
            has_pii = "passport" in content.lower() or "ssn" in content.lower()
            
            return {
                "is_compliant": not has_pii,
                "compliance_score": 0.5 if has_pii else 1.0,
                "violations": [
                    {
                        "rule_id": "PII_DETECTION",
                        "description": "Personal Identifiable Information detected",
                        "severity": "high" 
                    }
                ] if has_pii else []
            }
    
    # Initialize components
    event_bus = EventBus()
    compliance_verifier = MockVerifier()
    
    # Initialize monitoring system
    monitor = ComplianceMonitor(
        event_bus,
        compliance_verifier,
        config={
            "frameworks": ["GDPR", "HIPAA", "CCPA"],
            "verification_mode": "strict",
            "sample_rate": 1.0
        }
    )
    
    # Initialize report generator
    report_generator = ComplianceReportGenerator(event_bus)
    
    # Initialize alert handler
    alert_handler = ComplianceAlertHandler(
        event_bus,
        notification_config={
            "channels": ["log", "email"],
            "email_recipients": ["compliance@example.com"],
            "severity_thresholds": {
                "high": 1,
                "medium": 3,
                "low": 10
            }
        }
    )
    
    # Simulate some content events
    content_events = [
        ContentEvent(
            event_type=ContentEventType.CONTENT_CREATED,
            content_id="doc-123",
            content_type="text",
            user_id="user-456",
            content="This is a test document with no PII data.",
            metadata={"domain": "healthcare"}
        ),
        ContentEvent(
            event_type=ContentEventType.CONTENT_UPDATED,
            content_id="doc-456",
            content_type="text",
            user_id="user-789",
            content="This document contains a passport number: AB123456.",
            metadata={"domain": "finance"}
        ),
        ContentEvent(
            event_type=ContentEventType.CONTENT_PUBLISHED,
            content_id="doc-789",
            content_type="text",
            user_id="user-123",
            content="Patient data includes SSN 123-45-6789.",
            metadata={"domain": "healthcare"}
        )
    ]
    
    # Process events
    for event in content_events:
        print(f"Publishing event: {event.event_type.value} for content {event.content_id}")
        await event_bus.publish("content", event)
        await asyncio.sleep(0.1)  # Small delay for demonstration
    
    # Wait a moment for processing
    await asyncio.sleep(1)
    
    # Generate a report
    report = report_generator.generate_report()
    print("\nCompliance Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    asyncio.run(main())