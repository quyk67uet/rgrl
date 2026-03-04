"""
SQLAlchemy ORM models for VHAS trace storage.

This module defines the database schema for storing clinical ED workflow traces
conforming to the VHAS OTLP Trace Schema.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, DateTime, ForeignKey, Index, Text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import JSON

Base = declarative_base()


class WorkflowTrace(Base):
    """
    Represents a complete clinical workflow execution trace.
    
    A trace contains metadata about the clinical scenario and orchestrator decisions,
    along with a collection of spans representing individual steps (orchestrator decisions,
    agent executions, tool calls, etc.).
    
    Attributes:
        id (str): The unique trace_id from the OTLP trace (primary key).
        task_id (str): The clinical scenario task identifier.
        nl_command (str): The natural language command from the clinician.
        created_at (datetime): Timestamp when the trace was created.
        feedback (dict): Optional clinician feedback (rating, comments, etc.).
        spans (list[WorkflowSpan]): Collection of spans belonging to this trace.
    """
    __tablename__ = "workflow_traces"
    
    id = Column(String, primary_key=True, comment="Trace ID from OTLP trace")
    task_id = Column(String, nullable=True, index=True, comment="Clinical scenario task ID")
    nl_command = Column(Text, nullable=True, comment="Natural language command from clinician")
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        comment="Timestamp when trace was created"
    )
    feedback = Column(
        JSONB,
        nullable=True,
        default=None,
        comment="Clinician feedback on workflow quality (rating, comments, etc.)"
    )
    
    # Relationship to spans
    spans = relationship(
        "WorkflowSpan",
        back_populates="trace",
        cascade="all, delete-orphan",
        lazy="select"
    )
    
    def __repr__(self):
        return f"<WorkflowTrace(id='{self.id}', task_id='{self.task_id}', spans={len(self.spans)})>"


class WorkflowSpan(Base):
    """
    Represents a single step in a clinical workflow execution trace.
    
    Spans capture individual operations like orchestrator decisions, agent executions,
    or tool calls. They form a hierarchy through parent-child relationships and store
    flexible custom attributes in JSONB format.
    
    Attributes:
        id (int): Internal database ID (auto-incremented primary key).
        span_id (str): The unique span_id from the OTLP trace.
        trace_id (str): Foreign key reference to the parent trace.
        parent_span_id (str): The span_id of the parent span (null for root spans).
        name (str): Human-readable name describing the operation.
        attributes (dict): JSONB column storing custom VHAS attributes.
        trace (WorkflowTrace): Relationship back to the parent trace.
    """
    __tablename__ = "workflow_spans"
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment="Internal DB ID")
    span_id = Column(
        String,
        unique=True,
        nullable=False,
        index=True,
        comment="Unique span ID from OTLP trace"
    )
    trace_id = Column(
        String,
        ForeignKey("workflow_traces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to parent trace"
    )
    parent_span_id = Column(
        String,
        nullable=True,
        index=True,
        comment="Parent span ID for hierarchical tracing"
    )
    name = Column(String, nullable=False, comment="Human-readable operation name")
    
    # JSONB for PostgreSQL, JSON for SQLite
    # This stores all custom VHAS attributes (vhas.*)
    attributes = Column(
        JSONB,
        nullable=False,
        default={},
        comment="Custom VHAS semantic attributes"
    )
    
    # Relationship to trace
    trace = relationship("WorkflowTrace", back_populates="spans")
    
    # Composite index for efficient queries
    __table_args__ = (
        Index("idx_span_trace_parent", "trace_id", "parent_span_id"),
    )
    
    def __repr__(self):
        return f"<WorkflowSpan(span_id='{self.span_id}', name='{self.name}', trace_id='{self.trace_id}')>"

