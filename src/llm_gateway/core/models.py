# llm_gateway/core/models.py

"""
Core Pydantic models for the LLM Gateway, with comprehensive support for
the Model Context Protocol (MCP) standard alongside gateway-specific features.
Defines contracts for data flow, configuration, context, and results.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type, AsyncGenerator, cast

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict

# --- Base Model with Common Config ---

class BaseGatewayModel(BaseModel):
    """Base model for common configuration like assignment validation."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',  # Allow extra fields for forward compatibility
    )

# --- Enums ---

# Gateway-Internal Enums (Can be mapped from/to MCP equivalents)
class FinishReason(str, Enum):
    """Reasons why a generation completed."""
    STOP = "stop"  # Natural stop
    LENGTH = "length"  # Hit max tokens
    TOOL_CALLS = "tool_calls"  # Stopped to make tool calls
    CONTENT_FILTERED = "content_filtered"  # Content was filtered
    ERROR = "error"  # Error occurred
    UNKNOWN = "unknown"  # Unknown reason

class ComplianceStatus(str, Enum):
    """Status of compliance verification."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_CHECKED = "not_checked"
    ERROR = "error"

class ViolationSeverity(str, Enum):
    """Severity levels for compliance violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorLevel(str, Enum):
    """Severity levels for errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# Enums potentially aligning with MCP standard
class MCPRole(str, Enum):
    """Roles in an MCP conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"  # Represents the model's response
    TOOL = "tool"  # Represents the result of a tool execution

class MCPContentType(str, Enum):
    """Content types potentially alignable with MCP."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"  # Assuming MCP might support these
    FILE = "file"
    TOOL_USE = "tool_use"  # Model requests to use a tool
    TOOL_RESULT = "tool_result"  # Result provided back to the model

class MCPStopReason(str, Enum):
    """Stop reasons potentially alignable with MCP standard."""
    END_TURN = "endTurn"  # Natural end of model response
    TOOL_USE = "tool_use"  # Model stopped to use a tool
    MAX_TOKENS = "maxTokens"  # Output length limit reached
    STOP_SEQUENCE = "stopSequence"  # Stop sequence detected
    CONTENT_FILTERED = "content_filtered"  # Provider filtered content
    ERROR = "error"  # Provider error

# --- Core Content & Message Structures (Alignable with MCP) ---

class ContentItem(BaseGatewayModel):
    """A single piece of potentially multimodal content, alignable with MCP content blocks."""
    model_config = ConfigDict(frozen=True)

    type: MCPContentType = Field(..., description="Type of content.")
    # Example structures based on potential MCP format:
    # For TEXT: {"text": "Hello world"}
    # For IMAGE: {"image": {"format": "jpeg", "source": {"type": "base64", "data": "..."}}}
    # For TOOL_USE: {"tool_use": {"tool_use_id": "...", "name": "...", "input": {...}}}
    data: Dict[str, Any] = Field(..., description="Structured data representing the content based on its type.")
    mime_type: Optional[str] = Field(None, description="MIME type for the content, e.g., 'image/jpeg'")
    text_content: Optional[str] = Field(None, description="Convenience field for accessing text content")

    @model_validator(mode='after')
    def set_text_content(self) -> 'ContentItem':
        """Set text_content for TEXT type to simplify access."""
        if self.type == MCPContentType.TEXT and not self.text_content:
            object.__setattr__(self, 'text_content', self.data.get("text"))
        return self

    # Convenience methods
    @classmethod
    def from_text(cls, text: str) -> 'ContentItem':
        """Create a text content item."""
        return cls(type=MCPContentType.TEXT, data={"text": text}, text_content=text)

    @classmethod
    def from_image_base64(cls, base64_data: str, mime_type: str = "image/jpeg") -> 'ContentItem':
        """Create an image content item from base64 data."""
        return cls(
            type=MCPContentType.IMAGE,
            data={"image": {"source": {"type": "base64", "data": base64_data}}},
            mime_type=mime_type
        )

    @classmethod
    def from_image_url(cls, url: str, mime_type: Optional[str] = None) -> 'ContentItem':
        """Create an image content item from a URL."""
        return cls(
            type=MCPContentType.IMAGE,
            data={"image": {"source": {"type": "url", "url": url}}},
            mime_type=mime_type
        )

class Message(BaseGatewayModel):
    """Represents a message in the conversation, alignable with MCP messages."""
    # Messages in history/request/response should generally be immutable records
    model_config = ConfigDict(frozen=True)

    role: MCPRole = Field(..., description="The role of the message author.")
    # MCP typically supports a list of content blocks per message
    content: List[ContentItem] = Field(..., description="A list of content items comprising the message.")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this message instance.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata.")

class ConversationTurn(BaseGatewayModel):
    """A single turn in a conversation history."""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: Union[str, List[ContentItem], Dict[str, Any]]
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Convenience methods
    @classmethod
    def from_text(cls, role: str, text: str) -> 'ConversationTurn':
        """Create a turn from plain text."""
        return cls(role=role, content=text)
    
    @classmethod
    def from_message(cls, message: Message) -> 'ConversationTurn':
        """Create a turn from a Message object."""
        return cls(
            role=message.role.value,
            content=message.content,
            timestamp=message.timestamp,
            metadata=message.metadata
        )

# --- Tool Definition and Call Structures (Alignable with MCP) ---

class ToolParameterSchema(BaseGatewayModel):
    """Parameter schema for a tool, potentially alignable with MCP/OpenAPI."""
    model_config = ConfigDict(frozen=True)

    type: str = Field("object", description="Typically 'object' for tool parameters.")
    properties: Dict[str, Dict[str, Any]] = Field(..., description="Schema for each parameter (e.g., {'param_name': {'type': 'string', 'description': ...}}).")
    required: Optional[List[str]] = Field(None, description="List of required parameter names.")


class ToolFunction(BaseGatewayModel):
    """Definition of a tool function."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Name of the tool.")
    description: Optional[str] = Field(None, description="Description of what the tool does and when to use it.")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Schema for the tool's input parameters.")


class ToolDefinition(BaseGatewayModel):
    """Definition of a tool, potentially alignable with MCP tool definition."""
    model_config = ConfigDict(frozen=True)

    function: ToolFunction = Field(..., description="Function definition for the tool.")

    @classmethod
    def from_function(cls, name: str, description: Optional[str] = None, 
                     parameters: Optional[Dict[str, Any]] = None) -> 'ToolDefinition':
        """Create a tool definition from function details."""
        function = ToolFunction(name=name, description=description, parameters=parameters)
        return cls(function=function)


class ToolUseRequest(BaseGatewayModel):
    """Represents a request from the model to use a tool, alignable with MCP tool_use."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this specific tool use request.")
    type: str = Field("function", description="Type of tool use (typically 'function').")
    function: ToolFunction = Field(..., description="Function details including name and parameters.")


class ToolResult(BaseGatewayModel):
    """Represents the result of executing a tool, alignable with MCP tool_result."""
    # The result itself is immutable, forms content for a TOOL role message
    model_config = ConfigDict(frozen=True)

    tool_call_id: str = Field(..., description="ID matching the corresponding ToolUseRequest.")
    output: Union[str, Dict[str, Any]] = Field(..., description="Result of the tool execution.")
    is_error: bool = Field(False, description="Indicates if the tool execution resulted in an error.")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Details about the error if is_error is True.")


# --- Gateway Configuration & Metadata ---

class ModelVersion(BaseGatewayModel):
    """Version information."""
    model_config = ConfigDict(frozen=True)
    
    api_version: str = "v1"
    model_schema_version: str = "4.0.0"  # Incremented version

    def __str__(self) -> str:
        return f"{self.api_version}-{self.model_schema_version}"


class UsageStats(BaseGatewayModel):
    """Token usage stats (can be gateway calculated or mapped from provider)."""
    model_config = ConfigDict(frozen=True)
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode='after')
    def calculate_total(self) -> 'UsageStats':
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            object.__setattr__(self, 'total_tokens', self.prompt_tokens + self.completion_tokens)
        return self


class PerformanceMetrics(BaseGatewayModel):
    """Performance metrics."""
    model_config = ConfigDict(frozen=True)
    
    total_duration_ms: Optional[float] = None
    llm_latency_ms: Optional[float] = None
    pre_processing_duration_ms: Optional[float] = None
    post_processing_duration_ms: Optional[float] = None
    compliance_check_duration_ms: Optional[float] = None
    gateway_overhead_ms: Optional[float] = None

    @model_validator(mode='after')
    def calculate_overhead(self) -> 'PerformanceMetrics':
        """Calculate gateway overhead if possible."""
        if (self.total_duration_ms is not None and self.llm_latency_ms is not None and 
            self.gateway_overhead_ms is None):
            object.__setattr__(self, 'gateway_overhead_ms', 
                             max(0, self.total_duration_ms - self.llm_latency_ms))
        return self


class ExtensionPoints(BaseGatewayModel):
    """Container for forward compatibility."""
    
    provider_extensions: Dict[str, Any] = Field(default_factory=dict)
    experimental_features: Dict[str, Any] = Field(default_factory=dict)


class ErrorDetails(BaseGatewayModel):
    """Structured error information."""
    model_config = ConfigDict(frozen=True)
    
    code: str
    message: str
    level: ErrorLevel = ErrorLevel.ERROR
    provider_error_details: Optional[Dict[str, Any]] = None
    intervention_error_details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retryable: bool = False
    retry_after_seconds: Optional[int] = None


class RequestMetadata(BaseGatewayModel):
    """Observability metadata."""
    # Allow mutation if enriched during processing
    
    client_id: Optional[str] = None
    application_name: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    correlation_id: Optional[str] = None  # Trace ID


class QuotaInfo(BaseGatewayModel):
    """Rate limit quota info."""
    model_config = ConfigDict(frozen=True)
    
    requests_remaining: Optional[int] = None
    tokens_remaining: Optional[int] = None
    reset_at: Optional[datetime] = None


class CacheMetadata(BaseGatewayModel):
    """Information about cache hits."""
    model_config = ConfigDict(frozen=True)
    
    cache_hit: bool = False
    cache_key: Optional[str] = None
    ttl_seconds_remaining: Optional[int] = None
    stored_at: Optional[datetime] = None


# --- Compliance, Guardrails, RAG, Feedback (Gateway Concerns) ---

class Violation(BaseGatewayModel):
    """Compliance violation details."""
    model_config = ConfigDict(frozen=True)
    
    rule_id: Optional[str] = None
    framework_id: Optional[str] = None
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    type: str = "Unknown"
    description: str
    affected_text: Optional[str] = None
    affected_elements: Optional[List[str]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    handler_action: Optional[str] = None


class ComplianceResult(BaseGatewayModel):
    """Aggregated compliance check results."""
    model_config = ConfigDict(frozen=True)
    
    status: ComplianceStatus = ComplianceStatus.NOT_CHECKED
    violations: List[Violation] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    proof_id: Optional[str] = None
    checked_at: Optional[datetime] = None


class GuardrailConfig(BaseGatewayModel):
    """Guardrail settings."""
    
    content_filter_categories: Dict[str, bool] = Field(
        default_factory=lambda: {
            "hate": True, 
            "sexual": True, 
            "violence": True, 
            "self_harm": True
        }
    )
    content_filter_severity_threshold: ViolationSeverity = ViolationSeverity.MEDIUM
    detect_prompt_injection: bool = True
    detect_jailbreak: bool = True
    pii_detection_level: Optional[str] = "medium"
    custom_topic_blocklist: Optional[List[str]] = None
    custom_term_blocklist: Optional[List[str]] = None


class Citation(BaseGatewayModel):
    """RAG citation details."""
    model_config = ConfigDict(frozen=True)
    
    source_id: str
    source_type: str = "document"
    title: Optional[str] = None
    url: Optional[str] = None
    retrieval_score: Optional[float] = None
    content_snippet: Optional[str] = None


class RetrievalMetadata(BaseGatewayModel):
    """RAG retrieval metadata (stored in InterventionData)."""
    
    retrieval_strategy: str
    citations: List[Citation] = Field(default_factory=list)
    embedding_model: Optional[str] = None
    query_transformations: List[str] = Field(default_factory=list)
    retrieval_latency_ms: Optional[float] = None


class HumanFeedbackRequest(BaseGatewayModel):
    """Request structure for human feedback."""
    
    response_id: str
    reason_code: str
    reason_description: Optional[str] = None
    priority: str = Field("medium", pattern="^(low|medium|high)$")
    requested_by: str = "system"
    deadline: Optional[datetime] = None
    queue_id: Optional[str] = None


class HumanFeedbackResult(BaseGatewayModel):
    """Result structure for human feedback."""
    model_config = ConfigDict(frozen=True)
    
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str
    reviewer_id: str
    is_compliant: Optional[bool] = None
    rating: Optional[int] = None
    comments: Optional[str] = None
    corrected_content: Optional[Union[str, List[ContentItem]]] = None
    identified_issues: List[str] = Field(default_factory=list)
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)


# --- MCP Interaction Specific ---

class MCPUsage(BaseGatewayModel):
    """Usage reported by MCP endpoint."""
    model_config = ConfigDict(frozen=True)
    
    input_tokens: int = 0
    output_tokens: int = 0


class MCPMetadata(BaseGatewayModel):
    """Metadata from an MCP interaction."""
    model_config = ConfigDict(frozen=True)
    
    mcp_version: str = "1.0"
    model_version_reported: Optional[str] = None  # Underlying model version
    context_id: Optional[str] = None  # For stateful context
    # Raw usage from MCP provider (might differ slightly from gateway's UsageStats)
    provider_usage: Optional[MCPUsage] = None


# --- Intervention Pipeline & Context ---

class InterventionData(BaseGatewayModel):
    """Flexible container for intervention data sharing."""
    
    data: Dict[str, Any] = Field(default_factory=dict)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the data dictionary."""
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the data dictionary."""
        return self.data.get(key, default)


class InterventionConfig(BaseGatewayModel):
    """Configuration for intervention pipeline execution."""
    
    enabled_pre_interventions: List[str] = Field(default_factory=list)
    enabled_post_interventions: List[str] = Field(default_factory=list)
    total_intervention_timeout_ms: int = Field(10000, gt=0)
    fail_open: bool = Field(False)


class InterventionContext(BaseGatewayModel):
    """Mutable context object passed through the intervention pipeline."""
    # Allow mutation during pipeline execution
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    user_info: Dict[str, Any] = Field(default_factory=dict)
    timestamp_start: datetime = Field(default_factory=datetime.utcnow)
    target_domain: Optional[str] = None
    required_compliance_frameworks: List[str] = Field(default_factory=list)
    compliance_mode: str = Field("strict")
    # History can store gateway-native ConversationTurn or potentially raw MCP Messages
    # Let's stick to gateway native ConversationTurn for internal consistency before MCP conversion
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    intervention_data: InterventionData = Field(default_factory=InterventionData)
    trace_id: Optional[str] = None
    guardrail_config: Optional[GuardrailConfig] = None
    request_metadata: Optional[RequestMetadata] = None
    quota_info: Optional[QuotaInfo] = None
    intervention_config: InterventionConfig = Field(default_factory=InterventionConfig)

    # Convenience methods
    def add_conversation_turn(self, role: str, content: Union[str, List[ContentItem]]) -> None:
        """Add a new turn to the conversation history."""
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.utcnow()
        )
        self.conversation_history.append(turn)


# --- Core LLM Config, Request, Response ---

class LLMConfig(BaseGatewayModel):
    """Configuration parameters requested for the LLM call."""
    # Configuration provided in the request, may be overridden by gateway defaults/policies
    
    model_identifier: str = Field(..., description="Requested model identifier.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = Field(None)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    timeout_seconds: Optional[float] = None
    system_prompt: Optional[str] = None

    model_config = ConfigDict(extra='allow')  # Allow provider-specific parameters


class LLMRequest(BaseGatewayModel):
    """Input request object for the LLM Gateway."""
    
    version: str = "1.0"
    # Input prompt can be simple text or multimodal
    prompt_content: Union[str, List[ContentItem]] = Field(..., description="Input prompt (text or multimodal items).")
    config: LLMConfig = Field(..., description="Requested LLM configuration.")
    initial_context: InterventionContext = Field(default_factory=InterventionContext)
    stream: bool = Field(False)
    # Tools the USER wants the model to potentially use
    tools: Optional[List[ToolDefinition]] = Field(None, description="Definitions of tools available for the model to use.")
    extensions: ExtensionPoints = Field(default_factory=ExtensionPoints)


class LLMResponse(BaseGatewayModel):
    """Output response object from the LLM Gateway."""
    # Final response object is immutable
    model_config = ConfigDict(frozen=True)
    
    version: str = "1.0"
    request_id: str  # Matches LLMRequest's initial_context.request_id
    # Final generated content after interventions
    generated_content: Optional[Union[str, List[ContentItem]]] = None
    # Reason generation stopped (mapped from provider/MCP)
    finish_reason: Optional[FinishReason] = None
    # Tool use requests made by the model (mapped from provider/MCP)
    tool_use_requests: Optional[List[ToolUseRequest]] = Field(None, description="Tool use requested by the model.")
    # Gateway-level usage stats (mapped/calculated)
    usage: Optional[UsageStats] = None
    # Gateway-level compliance result
    compliance_result: Optional[ComplianceResult] = None
    # Final state of the context after all processing
    final_context: InterventionContext
    # Error details if processing failed
    error_details: Optional[ErrorDetails] = None
    # Timestamps and performance
    timestamp_end: datetime = Field(default_factory=datetime.utcnow)
    performance_metrics: Optional[PerformanceMetrics] = None
    # Metadata fields
    cache_metadata: Optional[CacheMetadata] = None  # If from cache
    quota_info: Optional[QuotaInfo] = None  # Status after call
    extensions: ExtensionPoints = Field(default_factory=ExtensionPoints)
    # MCP-specific metadata if MCP provider was used
    mcp_metadata: Optional[MCPMetadata] = None
    # Raw provider response for debugging (optional, excluded from serialization)
    raw_provider_response: Optional[Any] = Field(None, exclude=True)


class StreamChunk(BaseGatewayModel):
    """A chunk of a streaming response."""
    model_config = ConfigDict(frozen=True)
    
    chunk_id: int
    request_id: str
    delta_text: Optional[str] = None
    delta_content_items: Optional[List[ContentItem]] = None
    delta_tool_calls: Optional[List[ToolUseRequest]] = None
    finish_reason: Optional[FinishReason] = None
    usage_update: Optional[UsageStats] = None
    provider_specific_data: Optional[Dict[str, Any]] = None


# --- Batch Processing Models ---

class BatchLLMRequest(BaseGatewayModel):
    """Container for batch LLM requests."""
    
    version: str = "1.0"
    batch_id: str = Field(default_factory=lambda: f"batch_{uuid.uuid4()}")
    requests: List[LLMRequest] = Field(..., min_length=1)
    parallel: bool = True


class BatchLLMResponse(BaseGatewayModel):
    """Container for batch LLM responses."""
    model_config = ConfigDict(frozen=True)
    
    version: str = "1.0"
    batch_id: str
    responses: List[LLMResponse]  # Order matches requests
    aggregated_usage: Optional[UsageStats] = None
    total_duration_ms: Optional[float] = None
    error_count: int = Field(0)
    success_count: int = Field(0)

    @model_validator(mode='after')
    def calculate_aggregated_stats(self) -> 'BatchLLMResponse':
        """Calculate aggregated usage and counts if not provided."""
        # Calculate usage
        if self.aggregated_usage is None:
            total_prompt = sum(r.usage.prompt_tokens for r in self.responses if r.usage)
            total_completion = sum(r.usage.completion_tokens for r in self.responses if r.usage)
            if total_prompt > 0 or total_completion > 0:
                object.__setattr__(self, 'aggregated_usage', UsageStats(
                    prompt_tokens=total_prompt, 
                    completion_tokens=total_completion
                ))
        
        # Calculate success/error counts
        error_count = sum(1 for r in self.responses if r.error_details is not None)
        success_count = len(self.responses) - error_count
        
        object.__setattr__(self, 'error_count', error_count)
        object.__setattr__(self, 'success_count', success_count)
        
        return self


# --- Gateway / Provider Static Config Models ---

class ProviderModelInfo(BaseGatewayModel):
    """Capabilities/limits for a specific provider model."""
    model_config = ConfigDict(frozen=True)
    
    model_name: str
    context_window: int
    max_output_tokens: Optional[int] = None
    supports_streaming: bool = True
    supports_tools: bool = True
    input_modalities: List[MCPContentType] = Field(default_factory=lambda: [MCPContentType.TEXT])
    output_modalities: List[MCPContentType] = Field(default_factory=lambda: [MCPContentType.TEXT])


class ProviderConfig(BaseGatewayModel):
    """Configuration for a specific LLM provider."""
    model_config = ConfigDict(frozen=True)
    
    provider_id: str
    provider_type: str
    display_name: Optional[str] = None
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    # Credential management handled separately (e.g., via SecureConfigManager)
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    default_timeout_seconds: float = 60.0
    additional_config: Dict[str, Any] = Field(default_factory=dict)


class GatewayConfig(BaseGatewayModel):
    """Overall static configuration for the LLM Gateway service."""
    model_config = ConfigDict(frozen=True)
    
    gateway_id: str
    default_provider: str = "openai"
    default_model_identifier: Optional[str] = None
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    default_timeout_seconds: float = 60.0
    allowed_providers: List[str] = Field(default_factory=list)
    default_intervention_config: InterventionConfig = Field(default_factory=InterventionConfig)
    caching_enabled: bool = True
    cache_default_ttl_seconds: int = Field(3600, ge=0)
    default_compliance_mode: str = Field("strict", pattern="^(strict|audit|permissive)$")
    logging_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    additional_config: Dict[str, Any] = Field(default_factory=dict)
    # MCP specific global config
    mcp_api_endpoint: Optional[str] = Field(None, description="Global endpoint for MCP providers if applicable.")
    mcp_enabled_models: List[str] = Field(default_factory=list, description="List of model_identifiers explicitly supporting MCP.")


# --- Synchronous API Typehint ---
class AsyncIterableStreamChunk(AsyncGenerator[StreamChunk, None]):
    """Type hint for async generators of stream chunks."""
    pass


# --- Gateway Pipeline Interface (Abstract) ---

class GatewayPipeline:
    """Abstract interface for Gateway pipelines, implemented by providers."""
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process a request synchronously and return a response."""
        raise NotImplementedError("Subclasses must implement process_request")
    
    async def process_stream(self, request: LLMRequest) -> AsyncIterableStreamChunk:
        """Process a request and yield stream chunks."""
        raise NotImplementedError("Subclasses must implement process_stream")
    
    async def process_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """Process a batch of requests."""
        raise NotImplementedError("Subclasses must implement process_batch")

class MCPUsage(BaseGatewayModel):
    """Usage reported by MCP endpoint."""
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0

# --- Conversion Utilities ---

class MCPConverter:
    """Utility class for converting between Gateway and MCP formats."""
    
    @staticmethod
    def gateway_to_mcp_message(turn: ConversationTurn) -> Dict[str, Any]:
        """Convert a Gateway ConversationTurn to MCP message format."""
        content_list = []
        
        # Handle different content types
        if isinstance(turn.content, str):
            # Plain text becomes a single text content block
            content_list = [{"type": "text", "text": turn.content}]
        elif isinstance(turn.content, list):
            # Already a list of ContentItems
            content_list = [MCPConverter._content_item_to_mcp(item) for item in turn.content]
        elif isinstance(turn.content, dict):
            # Single dict content that needs conversion
            if "type" in turn.content:
                # If it looks like a ContentItem-like dict
                content_list = [turn.content]
            else:
                # Default to text if structure unknown
                content_list = [{"type": "text", "text": str(turn.content)}]
        
        # Create MCP message
        return {
            "role": turn.role,
            "content": content_list,
            "id": turn.turn_id,
            "metadata": turn.metadata
        }
    
    @staticmethod
    def _content_item_to_mcp(item: Union[ContentItem, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert ContentItem to MCP content block."""
        if isinstance(item, ContentItem):
            # Extract from ContentItem model
            return {"type": item.type.value, **item.data}
        else:
            # Already a dict, assume correctly structured
            return item
    
    @staticmethod
    def mcp_to_gateway_message(mcp_message: Dict[str, Any]) -> ConversationTurn:
        """Convert an MCP message to a Gateway ConversationTurn."""
        role = mcp_message.get("role", "unknown")
        
        # Process content array from MCP
        content_items = []
        mcp_content = mcp_message.get("content", [])
        
        for content_block in mcp_content:
            content_type = content_block.get("type")
            
            if content_type == "text":
                # Text is common, handle specially
                text = content_block.get("text", "")
                content_items.append(ContentItem.from_text(text))
            elif content_type == "image":
                # Handle image data
                if "url" in content_block.get("image", {}).get("source", {}):
                    url = content_block["image"]["source"]["url"]
                    content_items.append(ContentItem.from_image_url(url))
                elif "data" in content_block.get("image", {}).get("source", {}):
                    data = content_block["image"]["source"]["data"]
                    mime = content_block.get("image", {}).get("format", "jpeg")
                    content_items.append(ContentItem.from_image_base64(data, f"image/{mime}"))
            else:
                # For other types, create generic ContentItem
                try:
                    content_items.append(ContentItem(
                        type=MCPContentType(content_type),
                        data=content_block
                    ))
                except ValueError:
                    # Handle unknown content types
                    content_items.append(ContentItem(
                        type=MCPContentType.FILE,
                        data={"unknown": content_block}
                    ))
        
        # Create turn with proper metadata
        return ConversationTurn(
            turn_id=mcp_message.get("id", str(uuid.uuid4())),
            role=role,
            content=content_items,
            timestamp=datetime.fromisoformat(mcp_message.get("created_at")) if "created_at" in mcp_message else datetime.utcnow(),
            metadata=mcp_message.get("metadata", {})
        )
    
    @staticmethod
    def gateway_finish_to_mcp_stop(reason: FinishReason) -> MCPStopReason:
        """Map Gateway finish reason to MCP stop reason."""
        mapping = {
            FinishReason.STOP: MCPStopReason.END_TURN,
            FinishReason.LENGTH: MCPStopReason.MAX_TOKENS,
            FinishReason.TOOL_CALLS: MCPStopReason.TOOL_USE,
            FinishReason.CONTENT_FILTERED: MCPStopReason.CONTENT_FILTERED,
            FinishReason.ERROR: MCPStopReason.ERROR,
            FinishReason.UNKNOWN: MCPStopReason.END_TURN,
        }
        return mapping.get(reason, MCPStopReason.END_TURN)
    
    @staticmethod
    def mcp_stop_to_gateway_finish(reason: MCPStopReason) -> FinishReason:
        """Map MCP stop reason to Gateway finish reason."""
        mapping = {
            MCPStopReason.END_TURN: FinishReason.STOP,
            MCPStopReason.MAX_TOKENS: FinishReason.LENGTH,
            MCPStopReason.TOOL_USE: FinishReason.TOOL_CALLS,
            MCPStopReason.CONTENT_FILTERED: FinishReason.CONTENT_FILTERED,
            MCPStopReason.ERROR: FinishReason.ERROR,
            MCPStopReason.STOP_SEQUENCE: FinishReason.STOP
        }
        return mapping.get(reason, FinishReason.UNKNOWN)



# --- Example Usage ---
if __name__ == "__main__":
    print("--- LLM Gateway Models Demo ---")

    # Example ContentItem creation using convenience methods
    text_content = ContentItem.from_text("What is in this image?")
    image_content = ContentItem.from_image_url("http://example.com/image.png", "image/png")

    # Example LLM Request using simple text prompt
    llm_config = LLMConfig(model_identifier="mcp-compatible-model-v1")
    context = InterventionContext(session_id="sid-123")
    context.add_conversation_turn(MCPRole.USER.value, "Hello, how are you?")
    context.add_conversation_turn(MCPRole.ASSISTANT.value, "I'm doing well, thank you for asking!")
    
    llm_req = LLMRequest(
        prompt_content="Can you analyze this image for me?",
        config=llm_config,
        initial_context=context
    )
    
    # Example tool definition
    weather_tool = ToolDefinition.from_function(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or zip code"},
                "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"}
            },
            "required": ["location"]
        }
    )
    
    # Simulate a response
    response = LLMResponse(
        version="1.0",
        request_id=llm_req.initial_context.request_id,
        generated_content="The image shows a landscape with mountains and a lake.",
        finish_reason=FinishReason.STOP,
        usage=UsageStats(prompt_tokens=45, completion_tokens=12),
        compliance_result=ComplianceResult(status=ComplianceStatus.PASSED),
        final_context=context,
        performance_metrics=PerformanceMetrics(
            total_duration_ms=350.0,
            llm_latency_ms=320.0
        )
    )
    
    print(f"Request ID: {llm_req.initial_context.request_id}")
    print(f"Generated Content: {response.generated_content}")
    print(f"Finish Reason: {response.finish_reason}")
    print(f"Total Tokens: {response.usage.total_tokens}")
    print(f"Gateway Overhead: {response.performance_metrics.gateway_overhead_ms}ms")