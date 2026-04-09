"""
TraceStore implementation for VHAS clinical workflow traces.

This module provides a flexible storage system for clinical ED workflow traces,
supporting both SQLite (for development) and PostgreSQL (for production).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, joinedload
from sqlalchemy.types import JSON

from web.backend.models import Base, WorkflowTrace, WorkflowSpan

logger = logging.getLogger(__name__)


class BaseStorageBackend(ABC):
    """
    Abstract base class for workflow trace storage backends.
    
    This class defines the interface that all storage backends must implement.
    It handles SQLAlchemy engine and session setup, and provides the core
    method for adding workflow traces to the database.
    
    Attributes:
        engine: SQLAlchemy engine for database connections.
        SessionLocal: SQLAlchemy session factory.
    """
    
    def __init__(self, db_url: str):
        """
        Initialize the storage backend.
        
        Args:
            db_url (str): SQLAlchemy database connection URL.
        """
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create all tables defined in the models."""
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database tables created/verified for {self.__class__.__name__}")
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            Session: A new SQLAlchemy session.
        """
        return self.SessionLocal()
    
    def add_trace_from_dict(self, trace_data: Dict[str, Any]) -> str:
        """
        Add a workflow trace to the database from a dictionary (parsed JSON).
        
        This method parses the VHAS OTLP Trace Schema format and creates
        corresponding WorkflowTrace and WorkflowSpan ORM objects.
        
        Args:
            trace_data (dict): A dictionary representing a complete workflow trace,
                conforming to the VHAS OTLP schema.
        
        Returns:
            str: The trace_id of the inserted trace.
        
        Raises:
            ValueError: If the trace_data is invalid or missing required fields.
            Exception: If database insertion fails.
        """
        session = self.get_session()
        try:
            # Extract trace-level data
            trace_id = trace_data.get("trace_id")
            if not trace_id:
                raise ValueError("trace_id is required in trace_data")
            
            # Extract input scenario
            input_scenario = trace_data.get("input_scenario", {})
            task_id = input_scenario.get("task_id")
            nl_command = input_scenario.get("nl_command")
            
            # Create WorkflowTrace object
            trace = WorkflowTrace(
                id=trace_id,
                task_id=task_id,
                nl_command=nl_command,
                created_at=datetime.utcnow()
            )
            
            # Process spans
            spans_data = trace_data.get("spans", [])
            if not spans_data:
                logger.warning(f"Trace {trace_id} has no spans")
            
            for span_data in spans_data:
                span = self._create_span_from_dict(span_data, trace_id)
                trace.spans.append(span)
            
            # Add to session and commit
            session.add(trace)
            session.commit()
            
            logger.info(f"Successfully added trace {trace_id} with {len(trace.spans)} spans")
            return trace_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add trace: {e}", exc_info=True)
            raise
        finally:
            session.close()
    
    def _create_span_from_dict(self, span_data: Dict[str, Any], trace_id: str) -> WorkflowSpan:
        """
        Create a WorkflowSpan object from a dictionary.
        
        Args:
            span_data (dict): Dictionary containing span data.
            trace_id (str): The ID of the parent trace.
        
        Returns:
            WorkflowSpan: A new WorkflowSpan ORM object.
        """
        span_id = span_data.get("span_id")
        if not span_id:
            raise ValueError("span_id is required in span_data")
        
        # Create WorkflowSpan object
        span = WorkflowSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=span_data.get("parent_span_id"),
            name=span_data.get("name", "Unknown"),
            attributes=span_data.get("attributes", {})
        )
        
        return span
    
    def get_trace(self, trace_id: str) -> Optional[WorkflowTrace]:
        """
        Retrieve a workflow trace by its ID.
        
        Args:
            trace_id (str): The ID of the trace to retrieve.
        
        Returns:
            WorkflowTrace: The trace object with eagerly loaded spans, or None if not found.
        """
        session = self.get_session()
        try:
            trace = session.query(WorkflowTrace).options(
                joinedload(WorkflowTrace.spans)
            ).filter(WorkflowTrace.id == trace_id).first()
            
            if trace:
                session.expunge(trace)
            
            return trace
        finally:
            session.close()
    
    def list_traces(self, limit: int = 100, skip: int = 0) -> List[WorkflowTrace]:
        """
        List all workflow traces in the database.
        
        Args:
            limit (int): Maximum number of traces to return.
            skip (int): Number of traces to skip (for pagination).
        
        Returns:
            list[WorkflowTrace]: List of trace objects with eagerly loaded spans.
        """
        session = self.get_session()
        try:
            traces = session.query(WorkflowTrace).options(
                joinedload(WorkflowTrace.spans)
            ).order_by(WorkflowTrace.created_at.desc()).offset(skip).limit(limit).all()
            
            # Expunge objects from session to detach them
            for trace in traces:
                session.expunge(trace)
            
            return traces
        finally:
            session.close()
    
    def add_feedback(self, trace_id: str, feedback_data: Dict[str, Any]) -> bool:
        """
        Add or update feedback for a workflow trace.
        
        Args:
            trace_id (str): The ID of the trace to add feedback to.
            feedback_data (dict): Feedback data (e.g., {"rating": 5, "comment": "Excellent workflow!"}).
        
        Returns:
            bool: True if feedback was added successfully.
        
        Raises:
            ValueError: If trace_id does not exist.
        """
        session = self.get_session()
        try:
            trace = session.query(WorkflowTrace).filter(WorkflowTrace.id == trace_id).first()
            
            if not trace:
                raise ValueError(f"Trace with ID '{trace_id}' not found")
            
            trace.feedback = feedback_data
            session.commit()
            
            logger.info(f"Added feedback to trace {trace_id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add feedback: {e}", exc_info=True)
            raise
        finally:
            session.close()


class SQLiteBackend(BaseStorageBackend):
    """
    SQLite storage backend for development and testing.
    
    This backend uses SQLite as the database, which is ideal for local
    development, testing, and small-scale deployments.
    """
    
    def __init__(self, db_url: str = "sqlite:///vhas_tracestore.db"):
        """
        Initialize SQLite backend.
        
        Args:
            db_url (str): SQLite database URL. Defaults to 'sqlite:///vhas_tracestore.db'.
        """
        # Override JSONB with JSON for SQLite compatibility
        from sqlalchemy.dialects import sqlite
        if not hasattr(sqlite, 'JSONB'):
            # Monkey patch for SQLite to use JSON instead of JSONB
            import web.backend.models as models_module
            original_jsonb = models_module.JSONB
            models_module.JSONB = JSON
        
        super().__init__(db_url)
        logger.info(f"SQLite backend initialized: {db_url}")


class PostgresBackend(BaseStorageBackend):
    """
    PostgreSQL storage backend for production.
    
    This backend uses PostgreSQL with JSONB support for efficient
    storage and querying of workflow trace attributes.
    """
    
    def __init__(self, db_url: str):
        """
        Initialize PostgreSQL backend.
        
        Args:
            db_url (str): PostgreSQL database URL.
                Example: 'postgresql://user:password@localhost/dbname'
        """
        if not db_url:
            raise ValueError("db_url is required for PostgreSQL backend")
        
        super().__init__(db_url)
        logger.info(f"PostgreSQL backend initialized")


class VHASTraceStore:
    """
    Factory class for creating workflow trace storage backends.
    
    This class implements the Factory pattern to instantiate the appropriate
    storage backend based on the user's configuration.
    
    Usage:
        # SQLite (development)
        store = VHASTraceStore(backend="sqlite")
        
        # PostgreSQL (production)
        store = VHASTraceStore(
            backend="postgres",
            db_url="postgresql://user:password@localhost/vhas_tracestore"
        )
    """
    
    def __new__(cls, backend: str = "sqlite", db_url: Optional[str] = None):
        """
        Create a new storage backend instance.
        
        Args:
            backend (str): The backend type ('sqlite' or 'postgres').
            db_url (str, optional): Database connection URL.
                Required for 'postgres', optional for 'sqlite'.
        
        Returns:
            BaseStorageBackend: An instance of the appropriate backend.
        
        Raises:
            ValueError: If backend is invalid or required parameters are missing.
        """
        if backend == "sqlite":
            db_url = db_url or "sqlite:///vhas_tracestore.db"
            return SQLiteBackend(db_url=db_url)
        
        elif backend == "postgres":
            if not db_url:
                raise ValueError(
                    "db_url is required for PostgreSQL backend. "
                    "Example: 'postgresql://user:password@localhost/vhas_tracestore'"
                )
            return PostgresBackend(db_url=db_url)
        
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Supported backends: 'sqlite', 'postgres'"
            )

