# scripts/build_simulation_kb.py

import json
import os
from tqdm import tqdm

def build_knowledge_base(scenarios_data_dir: str, output_file: str):
    """
    Read all VHAS OTLP trace files from batch folders and build a knowledge base (KB)
    for the simulation env. The KB maps inputs to outputs for tools and state transitions.
    
    Folder layout:
    scenarios_data_dir/
      ├── batch_1/traces_1.json
      ├── batch_2/traces_2.json
      ├── ...
      └── batch_8/traces_8.json
    """
    print("--- Starting: Building Simulation Knowledge Base ---")
    
    knowledge_base = {
        "tools": {},           # Tool input/output mappings
        "state_transitions": {}  # (State, Action) -> Next State mappings
    }
    
    if not os.path.isdir(scenarios_data_dir):
        print(f"ERROR: Scenarios data directory not found at '{scenarios_data_dir}'. Aborting.")
        return

    # Find all batch folders
    batch_folders = [f for f in os.listdir(scenarios_data_dir) if f.startswith('batch_') and os.path.isdir(os.path.join(scenarios_data_dir, f))]
    batch_folders.sort()  # batch_1, batch_2, ..., batch_8
    
    print(f"Found {len(batch_folders)} batch folders: {', '.join(batch_folders)}")

    # Track stats
    total_tool_calls = 0
    total_state_transitions = 0
    total_traces = 0

    for batch_folder in tqdm(batch_folders, desc="Processing batches"):
        # Trace file path for this batch
        batch_num = batch_folder.split('_')[1]
        trace_file = os.path.join(scenarios_data_dir, batch_folder, f'traces_{batch_num}.json')
        
        if not os.path.exists(trace_file):
            print(f"\nWarning: Trace file not found: {trace_file}")
            continue
            
        print(f"\n  Processing {trace_file}...")
        
        with open(trace_file, 'r', encoding='utf-8') as f:
            try:
                # File contains an array of traces
                traces = json.load(f)
                print(f"    Found {len(traces)} traces in {batch_folder}")
                total_traces += len(traces)
                
                for trace in traces:
                    spans = trace.get('spans', [])
                    
                    # --- EXTRACT STATE TRANSITIONS ---
                    # Keep only orchestrator_decision spans in order
                    orchestrator_spans = [
                        span for span in spans 
                        if span.get('attributes', {}).get('vhas.span.type') == 'orchestrator_decision'
                    ]
                    
                    # Build state transitions from the decision chain
                    for i in range(len(orchestrator_spans) - 1):
                        current_span = orchestrator_spans[i]
                        next_span = orchestrator_spans[i + 1]
                        
                        current_state = current_span.get('attributes', {}).get('vhas.orchestrator.input_state')
                        action = current_span.get('attributes', {}).get('vhas.orchestrator.action_selected')
                        next_state = next_span.get('attributes', {}).get('vhas.orchestrator.input_state')
                        
                        if all([current_state, action, next_state]):
                            # Use a JSON array key [state, action] for easy parsing
                            transition_key = json.dumps([current_state, action], ensure_ascii=False)
                            knowledge_base["state_transitions"][transition_key] = next_state
                            total_state_transitions += 1
                    
                    # --- EXTRACT TOOL CALLS (same logic as before) ---
                    
                    for span in spans:
                        attributes = span.get('attributes', {})
                        
                        # Only inspect tool_call spans
                        if attributes.get('vhas.span.type') == 'tool_call':
                            tool_name = attributes.get('vhas.tool.name')
                            tool_input_str = attributes.get('vhas.tool.input')
                            tool_output_str = attributes.get('vhas.tool.output')
                            
                            if not all([tool_name, tool_input_str, tool_output_str]):
                                continue
                                
                            # Normalize the input key by parsing and re-dumping with sort_keys
                            try:
                                tool_input_dict = json.loads(tool_input_str)
                                input_key = json.dumps(tool_input_dict, sort_keys=True)
                            except json.JSONDecodeError:
                                # Fallback to the raw string if input is not valid JSON
                                input_key = tool_input_str

                            # Parse output so we store Python objects, not strings
                            try:
                                output_value = json.loads(tool_output_str)
                            except json.JSONDecodeError:
                                output_value = tool_output_str

                            # Add to the knowledge base
                            if tool_name not in knowledge_base["tools"]:
                                knowledge_base["tools"][tool_name] = {}
                            
                            # Overwrite duplicates because the gold data is assumed consistent
                            knowledge_base["tools"][tool_name][input_key] = output_value
                            total_tool_calls += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"\n  Warning: Could not process {trace_file}. Error: {e}")
                continue
    
    # Save the knowledge base
    print(f"\n{'='*70}")
    print(f"Processed {total_traces} traces across {len(batch_folders)} batches")
    print(f"Extracted {total_tool_calls} tool calls")
    print(f"Extracted {total_state_transitions} state transitions")
    print(f"\nKnowledge Base Statistics:")
    print(f"  - {len(knowledge_base['tools'])} unique tools")
    print(f"  - {len(knowledge_base['state_transitions'])} unique state transitions")
    print(f"\nSaving compiled Knowledge Base to '{output_file}'...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
        
    print("--- Simulation Knowledge Base built successfully! ---")
    print(f"\n💡 Sample state transition:")
    if knowledge_base["state_transitions"]:
        sample_key = list(knowledge_base["state_transitions"].keys())[0]
        sample_value = knowledge_base["state_transitions"][sample_key]
        print(f"  Key: {sample_key[:100]}...")
        print(f"  Next State: {sample_value[:100]}...")
    
    return knowledge_base

if __name__ == "__main__":
    import os
    
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Folder containing the batch directories (batch_1, batch_2, ..., batch_8)
    SCENARIOS_DATA_DIR = os.path.join(script_dir, '..', '..', 'data', 'scenarios', 'data')
    
    # Output file path
    OUTPUT_FILE = os.path.join(script_dir, '..', 'data', 'simulation_kb.json')
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    build_knowledge_base(
        scenarios_data_dir=SCENARIOS_DATA_DIR,
        output_file=OUTPUT_FILE
    )