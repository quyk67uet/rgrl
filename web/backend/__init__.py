"""
VHAS Tracing Module

This module provides tracing capabilities for storing and querying
clinical ED workflow traces using the VHAS OTLP Trace Schema.
"""

from .models import WorkflowTrace, WorkflowSpan
from .store import VHASTraceStore

__all__ = ["WorkflowTrace", "WorkflowSpan", "VHASTraceStore"]

