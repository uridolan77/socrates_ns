from fastapi import FastAPI, HTTPException, Depends, Header, Body, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import uvicorn
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager

# Import compliance components
# from src.compliance.compliance_verifier import ComplianceVerifier
# from src.compliance.compliance_proof_tracer import ComplianceProofTracer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("compliance_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("compliance_api")

# Models for API
class ContentRequest(BaseModel):
    content: str
    content_type: str = "text"
    frameworks: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    compliance_mode: str = "strict"
    trace_id: Optional[str] = None
    generate_proof: bool = False


class ViolationResponse(BaseModel):
    rule_id: str
    description: str
    severity: str = "medium"
    locations: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceResponse(BaseModel):
    is_compliant: bool
    compliance_score: float
    trace_id: str
    timestamp: str
    mode: str
    frameworks: List[str]
    violations: List[ViolationResponse] = Field(default_factory=list)
    proof_available: bool = False
    proof_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    trace_id: str
    timestamp: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load compliance components
    app.state.verifier = create_compliance_verifier()
    app.state.proof_cache = {}
    logger.info("Compliance API startup complete")
    yield
    # Shutdown: Clean up resources
    logger.info("Compliance API shutting down")


# Create FastAPI app
app = FastAPI(
    title="Compliance Verification API",
    description="API for verifying content against regulatory compliance frameworks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for API key validation
async def verify_api_key(api_key: str = Depends(api_key_header)):
    # In production, use a secure API key validation mechanism
    # This is a simplified example
    valid_api_keys = ["test_key_1", "test_key_2"]
    
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return api_key

# Create mock compliance verifier for demonstration
def create_compliance_verifier():
    # In a real implementation, this would create an actual verifier
    class MockVerifier:
        def verify_content(self, content, content_type="text", frameworks=None, compliance_mode="strict", context=None):
            # Simple mock implementation for demonstration
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
            
        def generate_proof(self, content, frameworks=None, compliance_mode="strict", context=None):
            # Generate a mock proof trace
            return {
                "proof_id": str(uuid.uuid4()),
                "framework_count": len(frameworks or ["GDPR"]),
                "rule_count": 5,
                "verification_steps": 10
            }
    
    return MockVerifier()

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/verify", response_model=ComplianceResponse, dependencies=[Depends(verify_api_key)])
async def verify_content(request: ContentRequest, request_obj: Request):
    """
    Verify content against compliance frameworks
    
    This endpoint checks content against specified regulatory frameworks
    and returns a detailed compliance assessment including any violations.
    """
    trace_id = request.trace_id or str(uuid.uuid4())
    logger.info(f"Verification request received | Trace ID: {trace_id}")
    
    try:
        # Get verifier from app state
        verifier = request_obj.app.state.verifier
        
        # Verify content
        result = verifier.verify_content(
            request.content,
            content_type=request.content_type,
            frameworks=request.frameworks,
            compliance_mode=request.compliance_mode,
            context=request.context
        )
        
        # Generate proof if requested
        proof_id = None
        if request.generate_proof:
            proof_result = verifier.generate_proof(
                request.content,
                frameworks=request.frameworks,
                compliance_mode=request.compliance_mode,
                context=request.context
            )
            proof_id = proof_result["proof_id"]
            request_obj.app.state.proof_cache[proof_id] = proof_result
        
        # Create response
        response = {
            "is_compliant": result["is_compliant"],
            "compliance_score": result["compliance_score"],
            "trace_id": trace_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": request.compliance_mode,
            "frameworks": request.frameworks,
            "violations": result.get("violations", []),
            "proof_available": proof_id is not None,
            "proof_id": proof_id,
            "metadata": {
                "content_type": request.content_type,
                "content_length": len(request.content)
            }
        }
        
        logger.info(f"Verification completed | Trace ID: {trace_id} | Compliant: {result['is_compliant']}")
        return response
        
    except Exception as e:
        logger.error(f"Error during verification | Trace ID: {trace_id} | Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Verification failed",
                "trace_id": trace_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detail": str(e)
            }
        )

@app.get("/proof/{proof_id}", dependencies=[Depends(verify_api_key)])
async def get_proof(proof_id: str, format: str = Query("json", description="Output format (json, html, text)")):
    """
    Get a verification proof by ID
    
    Retrieves a previously generated compliance verification proof.
    """
    # Check if proof exists in cache
    if proof_id not in app.state.proof_cache:
        raise HTTPException(
            status_code=404,
            detail={"error": "Proof not found", "proof_id": proof_id}
        )
    
    proof = app.state.proof_cache[proof_id]
    
    # Return proof in requested format
    if format == "html":
        # Generate HTML representation
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Verification Proof</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }}
                .compliant {{ background-color: #d4edda; }}
                .non-compliant {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>Compliance Verification Proof</h1>
            <div class="section">
                <h2>Summary</h2>
                <p>Proof ID: {proof_id}</p>
                <p>Frameworks: {proof.get('framework_count', 0)}</p>
                <p>Rules Checked: {proof.get('rule_count', 0)}</p>
                <p>Verification Steps: {proof.get('verification_steps', 0)}</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    else:
        # Default to JSON
        return proof

@app.get("/frameworks", dependencies=[Depends(verify_api_key)])
async def list_frameworks():
    """
    List available compliance frameworks
    
    Returns information about all supported regulatory frameworks.
    """
    # In a real implementation, this would return actual framework information
    frameworks = [
        {
            "id": "GDPR",
            "name": "General Data Protection Regulation",
            "description": "EU regulation on data protection and privacy",
            "version": "2016/679",
            "effective_date": "2018-05-25"
        },
        {
            "id": "HIPAA",
            "name": "Health Insurance Portability and Accountability Act",
            "description": "US regulation for medical information privacy",
            "version": "1996",
            "effective_date": "1996-08-21"
        },
        {
            "id": "CCPA",
            "name": "California Consumer Privacy Act",
            "description": "California law on consumer data privacy",
            "version": "2018",
            "effective_date": "2020-01-01"
        }
    ]
    
    return {"frameworks": frameworks}

@app.get("/rules/{framework_id}", dependencies=[Depends(verify_api_key)])
async def list_framework_rules(framework_id: str):
    """
    List rules for a specific framework
    
    Returns detailed information about all rules in a regulatory framework.
    """
    # In a real implementation, this would return actual rules for the framework
    if framework_id == "GDPR":
        rules = [
            {
                "id": "GDPR.Art.5.1.a",
                "description": "Personal data shall be processed lawfully, fairly and transparently",
                "severity": "high",
                "category": "data_processing_principles"
            },
            {
                "id": "GDPR.Art.5.1.b",
                "description": "Personal data must be collected for specified, explicit and legitimate purposes",
                "severity": "high",
                "category": "purpose_limitation"
            }
        ]
    elif framework_id == "HIPAA":
        rules = [
            {
                "id": "HIPAA.164.502.a",
                "description": "A covered entity or business associate may not use or disclose protected health information",
                "severity": "high",
                "category": "use_and_disclosure"
            }
        ]
    else:
        raise HTTPException(
            status_code=404,
            detail={"error": "Framework not found", "framework_id": framework_id}
        )
    
    return {"framework_id": framework_id, "rules": rules}

@app.get("/statistics", dependencies=[Depends(verify_api_key)])
async def get_statistics():
    """
    Get API usage statistics
    
    Returns statistics about API usage and compliance verification results.
    """
    # In a real implementation, this would return actual statistics
    return {
        "total_requests": 1250,
        "compliant_requests": 875,
        "non_compliant_requests": 375,
        "compliance_rate": 0.7,
        "average_compliance_score": 0.85,
        "framework_distribution": {
            "GDPR": 650,
            "HIPAA": 350,
            "CCPA": 250
        },
        "top_violations": [
            {"rule_id": "PII_DETECTION", "count": 120},
            {"rule_id": "GDPR.Art.5.1.a", "count": 95},
            {"rule_id": "HIPAA.164.502.a", "count": 75}
        ]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {
            "error": exc.detail,
            "trace_id": str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    trace_id = str(uuid.uuid4())
    logger.error(f"Unhandled exception | Trace ID: {trace_id} | Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "trace_id": trace_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detail": str(exc)
        }
    )

# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)