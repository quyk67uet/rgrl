"""
Seed the VHAS TraceStore with clinical workflow trace files.

This script demonstrates how to use the VHASTraceStore to ingest
clinical ED workflow traces from JSON files conforming to the
VHAS OTLP Trace Schema.

Usage:
    1. Start PostgreSQL: docker compose -f docker-compose-vhas-db.yml up -d
    2. Run this script: python web/backend/seed_tracestore.py
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from web.backend.store import VHASTraceStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trace_files(directory: str) -> List[Dict[str, Any]]:
    """
    Load all JSON trace files from a directory.
    
    Args:
        directory (str): Path to the directory containing trace JSON files.
    
    Returns:
        list[dict]: List of parsed trace dictionaries.
    """
    traces = []
    trace_dir = Path(directory)
    
    if not trace_dir.exists():
        logger.error(f"Directory not found: {directory}")
        return traces
    
    # Get all .json files
    json_files = sorted(trace_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {directory}")
        return traces
    
    logger.info(f"Found {len(json_files)} trace files in {directory}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                # Each file contains an array of traces
                trace_array = json.load(f)
                if isinstance(trace_array, list):
                    traces.extend(trace_array)
                    logger.info(f"✓ Loaded {len(trace_array)} traces from {json_file.name}")
                else:
                    logger.warning(f"✗ {json_file.name} does not contain an array")
        except Exception as e:
            logger.error(f"✗ Failed to load {json_file.name}: {e}")
    
    return traces


def seed_tracestore(backend: str = "postgres", db_url: str = None):
    """
    Seed the VHAS TraceStore with clinical workflow traces.
    
    Args:
        backend (str): Storage backend ('sqlite' or 'postgres').
        db_url (str): Database connection URL.
    """
    print("=" * 70)
    print("VHAS TraceStore Seeding Script")
    print("=" * 70)
    print()
    
    # Determine the path to trace files
    project_root = Path(__file__).parent.parent.parent
    
    # Load traces from train/validation/test split files.
    all_traces = []
    split_paths = [
        project_root / "ai" / "data" / "workflow_traces" / "train_traces.json",
        project_root / "ai" / "data" / "workflow_traces" / "val_traces.json",
        project_root / "ai" / "data" / "workflow_traces" / "test_traces.json",
    ]
    for split_path in split_paths:
        if split_path.exists():
            print(f"📂 Loading traces from: {split_path}")
            with split_path.open("r", encoding="utf-8") as handle:
                trace_array = json.load(handle)
            if isinstance(trace_array, list):
                all_traces.extend(trace_array)
                print(f"   Loaded {len(trace_array)} traces from {split_path.name}")
                print()
    
    if not all_traces:
        logger.error("No traces loaded. Exiting.")
        return
    
    print(f"📦 Total traces loaded: {len(all_traces)}")
    print()
    
    # Initialize TraceStore
    print(f"🔌 Connecting to {backend} database...")
    try:
        store = VHASTraceStore(backend=backend, db_url=db_url)
        print(f"✓ Connected to {backend} backend")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return
    
    print()
    print("💾 Ingesting traces into VHAS TraceStore...")
    print("-" * 70)
    
    # Ingest traces
    success_count = 0
    failed_count = 0
    
    for i, trace_data in enumerate(all_traces, 1):
        trace_id = trace_data.get("trace_id", "unknown")
        num_spans = len(trace_data.get("spans", []))
        task_id = trace_data.get("input_scenario", {}).get("task_id", "N/A")
        
        try:
            store.add_trace_from_dict(trace_data)
            if i % 50 == 0:  # Print every 50 traces
                print(f"✓ [{i}/{len(all_traces)}] Ingested trace {trace_id[:16]}... ({num_spans} spans)")
            success_count += 1
        except Exception as e:
            print(f"✗ [{i}/{len(all_traces)}] Failed to ingest trace {trace_id}: {e}")
            failed_count += 1
    
    # Summary
    print("-" * 70)
    print()
    print("📊 Summary:")
    print(f"   ✓ Successfully ingested: {success_count} traces")
    if failed_count > 0:
        print(f"   ✗ Failed: {failed_count} traces")
    print()
    
    # Query verification
    print("🔍 Verifying ingestion...")
    try:
        all_traces_db = store.list_traces(limit=100)
        print(f"✓ Found {len(all_traces_db)} traces in the database (showing first 100)")
        print()
        
        if all_traces_db:
            print("Sample traces:")
            for trace in all_traces_db[:5]:
                print(f"   - {trace.id[:16]}...: {len(trace.spans)} spans | Task: {trace.task_id}")
    except Exception as e:
        logger.error(f"Failed to query traces: {e}")
    
    print()
    print("=" * 70)
    print("✅ Seeding complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Start the API server: python web/backend/api.py")
    print("  2. Open API docs: http://localhost:8001/docs")


if __name__ == "__main__":
    # Configuration
    BACKEND = "postgres"  # Change to "sqlite" for local testing
    
    # PostgreSQL connection URL
    # Format: postgresql://username:password@host:port/database
    DB_URL = "postgresql://vhas:vhas2024@localhost:5433/vhas_tracestore"
    
    # For SQLite testing, uncomment:
    # BACKEND = "sqlite"
    # DB_URL = "sqlite:///vhas_tracestore_sample.db"
    
    # Run seeding
    seed_tracestore(backend=BACKEND, db_url=DB_URL)

