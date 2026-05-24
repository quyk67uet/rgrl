"""
VHAS TraceStore REST API

This module provides a FastAPI-based REST API for interacting with the VHAS TraceStore.
It exposes endpoints for querying guideline-compliant multi-agent workflow traces,
retrieving trace details, and adding expert feedback.

Features:
- GET /traces: List all workflow traces with pagination
- GET /traces/{trace_id}: Get detailed trace information
- POST /traces/{trace_id}/feedback: Add expert feedback to a trace
- Automatic API documentation at /docs
- Type-safe request/response models using Pydantic
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to Python path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from collections import Counter

from web.backend.store import VHASTraceStore
from web.backend.models import WorkflowTrace, WorkflowSpan

# Initialize FastAPI app
app = FastAPI(
    title="VHAS TraceStore API",
    description="REST API for managing and querying guideline-compliant multi-agent workflow traces",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TraceStore with PostgreSQL
# Use environment variable for database URL or default
DATABASE_URL = os.getenv(
    "VHAS_DATABASE_URL",
    "postgresql://vhas:vhas2024@localhost:5433/vhas_tracestore"
)

store = VHASTraceStore(backend="postgres", db_url=DATABASE_URL)

# Pydantic Models for API Schema

class SpanOut(BaseModel):
    """Response model for a single workflow span."""
    
    span_id: str = Field(..., description="Unique span identifier")
    parent_span_id: Optional[str] = Field(None, description="Parent span ID (null for root spans)")
    name: str = Field(..., description="Human-readable operation name")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom VHAS attributes")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "span_id": "1b356d696b974817",
                "parent_span_id": None,
                "name": "Orchestrator Decision: Start Triage",
                "attributes": {
                    "vhas.span.type": "orchestrator_decision",
                    "vhas.orchestrator.input_state": "Initial state: New workflow request received.",
                    "vhas.orchestrator.thought": "The request requires immediate routing.",
                    "vhas.orchestrator.action_selected": "TriageAgent"
                }
            }
        }


class TraceOut(BaseModel):
    """Response model for a workflow trace."""
    
    id: str = Field(..., description="Unique trace identifier")
    task_id: Optional[str] = Field(None, description="Scenario task ID")
    nl_command: Optional[str] = Field(None, description="Natural language command from operator")
    created_at: datetime = Field(..., description="Timestamp when trace was created")
    feedback: Optional[Dict[str, Any]] = Field(None, description="Expert feedback on workflow quality")
    spans: List[SpanOut] = Field(default_factory=list, description="List of spans in this trace")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "116ebb84cbaad37dfc1061311b4b331f",
                "task_id": "WORKFLOW_SCENARIO_00001",
                "nl_command": "Process priority workflow request for access review",
                "created_at": "2025-12-19T15:30:00Z",
                "feedback": {"rating": 5, "comment": "Excellent workflow!"},
                "spans": []
            }
        }


class TraceListOut(BaseModel):
    """Response model for trace list with metadata."""
    
    total: int = Field(..., description="Total number of traces returned")
    skip: int = Field(..., description="Number of traces skipped")
    limit: int = Field(..., description="Maximum number of traces requested")
    traces: List[TraceOut] = Field(..., description="List of traces")


class FeedbackIn(BaseModel):
    """Request model for adding feedback to a trace."""
    
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5")
    workflow_efficiency: Optional[int] = Field(None, ge=1, le=5, description="Workflow efficiency score (1-5)")
    decision_accuracy: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Decision accuracy score (1-5) of the orchestrator.",
    )
    comment: Optional[str] = Field(None, description="Text comment or feedback")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing feedback")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rating": 5,
                "workflow_efficiency": 5,
                "decision_accuracy": 4,
                "comment": "Excellent orchestration! Proper routing and task assessment.",
                "tags": ["high-quality", "efficient-workflow"],
                "metadata": {"expert_id": "EX001", "session_id": "abc"}
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback operations."""
    
    success: bool = Field(..., description="Whether the operation succeeded")
    message: str = Field(..., description="Status message")
    trace_id: str = Field(..., description="The trace ID that was updated")


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "VHAS TraceStore API",
        "version": "1.0.0",
        "description": "REST API for managing guideline-compliant multi-agent workflow traces",
        "docs": "/docs",
        "endpoints": {
            "list_traces": "GET /traces",
            "get_trace": "GET /traces/{trace_id}",
            "add_feedback": "POST /traces/{trace_id}/feedback",
            "stats": "GET /stats",
            "agent_usage": "GET /analytics/agent_usage"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API and database connectivity.
    """
    try:
        # Try to query the database
        traces = store.list_traces(limit=1)
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )


@app.get("/traces", response_model=TraceListOut, tags=["Traces"])
async def list_traces(
    skip: int = Query(0, ge=0, description="Number of traces to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of traces to return")
):
    """
    List all workflow traces with pagination.
    
    Returns traces ordered by creation time (most recent first).
    """
    try:
        # Get traces from store
        traces = store.list_traces(limit=limit, skip=skip)
        
        # Convert to Pydantic models
        trace_outs = []
        for trace in traces:
            span_outs = [SpanOut.model_validate(span) for span in trace.spans]
            trace_out = TraceOut(
                id=trace.id,
                task_id=trace.task_id,
                nl_command=trace.nl_command,
                created_at=trace.created_at,
                feedback=trace.feedback,
                spans=span_outs
            )
            trace_outs.append(trace_out)
        
        return TraceListOut(
            total=len(trace_outs),
            skip=skip,
            limit=limit,
            traces=trace_outs
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list traces: {str(e)}"
        )


@app.get("/traces/{trace_id}", response_model=TraceOut, tags=["Traces"])
async def get_trace(trace_id: str):
    """
    Get detailed information for a specific workflow trace.
    
    Args:
        trace_id: The unique identifier of the trace.
    
    Returns:
        Complete trace information including all spans and attributes.
    
    Raises:
        404: If the trace is not found.
    """
    try:
        trace = store.get_trace(trace_id)
        
        if not trace:
            raise HTTPException(
                status_code=404,
                detail=f"Trace with ID '{trace_id}' not found"
            )
        
        # Convert to Pydantic models
        span_outs = [SpanOut.model_validate(span) for span in trace.spans]
        trace_out = TraceOut(
            id=trace.id,
            task_id=trace.task_id,
            nl_command=trace.nl_command,
            created_at=trace.created_at,
            feedback=trace.feedback,
            spans=span_outs
        )
        
        return trace_out
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve trace: {str(e)}"
        )


@app.post("/traces/{trace_id}/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def add_feedback(trace_id: str, feedback: FeedbackIn):
    """
    Add or update expert feedback for a specific workflow trace.
    
    Args:
        trace_id: The unique identifier of the trace.
        feedback: Feedback data including rating, comments, tags, etc.
    
    Returns:
        Success confirmation with trace ID.
    
    Raises:
        404: If the trace is not found.
        500: If the operation fails.
    """
    try:
        # Convert Pydantic model to dict, excluding None values
        feedback_data = feedback.model_dump(exclude_none=True)
        
        # Add timestamp
        feedback_data["timestamp"] = datetime.utcnow().isoformat()
        
        # Store feedback
        store.add_feedback(trace_id, feedback_data)
        
        return FeedbackResponse(
            success=True,
            message="Feedback added successfully",
            trace_id=trace_id
        )
        
    except ValueError as e:
        # Trace not found
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add feedback: {str(e)}"
        )


# Analytics Endpoints

@app.get("/stats", tags=["Analytics"])
async def get_stats():
    """
    Get overall statistics about the VHAS TraceStore.
    
    Returns:
        Dictionary with key metrics including total traces, spans, etc.
    """
    try:
        session = store.get_session()
        try:
            # Count total traces
            total_traces = session.query(WorkflowTrace).count()
            
            # Count total spans
            total_spans = session.query(WorkflowSpan).count()
            
            # Count traces with feedback
            traces_with_feedback = session.query(WorkflowTrace).filter(
                WorkflowTrace.feedback.isnot(None)
            ).count()
            
            # Count traces from last 24 hours
            from datetime import timedelta
            yesterday = datetime.utcnow() - timedelta(days=1)
            traces_last_24h = session.query(WorkflowTrace).filter(
                WorkflowTrace.created_at >= yesterday
            ).count()
            
            # Average spans per trace
            avg_spans = total_spans / total_traces if total_traces > 0 else 0
            
            return {
                "total_traces": total_traces,
                "total_spans": total_spans,
                "traces_with_feedback": traces_with_feedback,
                "traces_last_24h": traces_last_24h,
                "avg_spans_per_trace": round(avg_spans, 2)
            }
        finally:
            session.close()
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


@app.get("/analytics/agent_usage", tags=["Analytics"])
async def get_agent_usage():
    """
    Get agent usage statistics from all workflow traces.
    
    Returns:
        List of agent names with their usage counts.
    """
    try:
        session = store.get_session()
        try:
            # Get all spans with agent execution type
            spans = session.query(WorkflowSpan).filter(
                WorkflowSpan.attributes["vhas.span.type"].astext == "agent_execution"
            ).all()
            
            # Count agent usage
            agent_counts = Counter()
            for span in spans:
                agent_name = span.attributes.get("vhas.agent.name")
                if agent_name:
                    agent_counts[agent_name] += 1
            
            # Convert to list of dicts for easy charting
            agent_usage = [
                {"agent": agent, "count": count}
                for agent, count in agent_counts.most_common(10)
            ]
            
            return {
                "agents": agent_usage,
                "total_agent_calls": sum(agent_counts.values())
            }
        finally:
            session.close()
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent usage: {str(e)}"
        )


@app.get("/analytics/pathway_distribution", tags=["Analytics"])
async def get_pathway_distribution():
    """
    Get distribution of workflow pathways used in traces.
    
    Returns:
        Statistics about which pathways are most common.
    """
    try:
        session = store.get_session()
        try:
            # Get all traces with their spans
            traces = session.query(WorkflowTrace).options(
                joinedload(WorkflowTrace.spans)
            ).all()
            
            pathway_counts = {
                "Monitor & Release": 0,
                "Quick Intervention": 0,
                "Complex Care Loop": 0,
                "Unknown": 0
            }
            
            for trace in traces:
                # Analyze agent sequence to determine pathway
                agent_sequence = []
                for span in sorted(trace.spans, key=lambda s: s.span_id):
                    if span.attributes.get("vhas.span.type") == "agent_execution":
                        agent_name = span.attributes.get("vhas.agent.name")
                        if agent_name:
                            agent_sequence.append(agent_name)
                
                # Classify pathway
                if agent_sequence == ["TriageAgent", "EHRAgent", "SummaryAgent"]:
                    pathway_counts["Monitor & Release"] += 1
                elif agent_sequence == ["TriageAgent", "DispensationAgent", "SummaryAgent"]:
                    pathway_counts["Quick Intervention"] += 1
                elif "ReconciliationAgent" in agent_sequence:
                    pathway_counts["Complex Care Loop"] += 1
                else:
                    pathway_counts["Unknown"] += 1
            
            return {
                "pathways": [
                    {"pathway": name, "count": count}
                    for name, count in pathway_counts.items()
                ],
                "total_traces": sum(pathway_counts.values())
            }
        finally:
            session.close()
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve pathway distribution: {str(e)}"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("VHAS TraceStore API Server")
    print("=" * 70)
    print(f"Database: {DATABASE_URL}")
    print(f"Starting server on http://localhost:8001")
    print(f"API Documentation: http://localhost:8001/docs")
    print(f"Alternative Docs: http://localhost:8001/redoc")
    print("=" * 70)
    print()
    
    uvicorn.run(
        "web.backend.api:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )

