# scripts/prepare_finetuning_data.py

import json
import os
from sentence_transformers.readers import InputExample
from tqdm import tqdm

def create_finetuning_pairs(batch_dir: str, universe_file: str, output_file: str):
    """Create positive (State, Action) pairs from clinical traces for encoder fine-tuning.

    Args:
        batch_dir: Directory containing batch folders (batch_1/traces_1.json, ...)
        universe_file: JSON file with agent definitions
        output_file: Output JSON file to store training examples
    """
    print("--- Starting WRL Fine-tuning Data Preparation ---")
    
    # 1. Build agent corpus map (Agent Corpus Map)
    print(f"Building agent corpus map from {universe_file}...")
    agent_corpus_map = {}
    try:
        with open(universe_file, 'r', encoding='utf-8') as f:
            universe = json.load(f)
        for agent in universe.get('agents', []):
            agent_corpus_map[agent['name']] = agent['description']
        print(f"   ✓ Loaded {len(agent_corpus_map)} agents")
    except FileNotFoundError:
        print(f"ERROR: Universe file not found at {universe_file}. Aborting.")
        return

    # 2. Iterate batch files and extract (State, Action) pairs
    print(f"Processing clinical traces from batch files in '{batch_dir}'...")
    training_examples = []
    
    if not os.path.isdir(batch_dir):
        print(f"ERROR: Batch directory not found at '{batch_dir}'.")
        return
    
    # Read up to 8 batch folders
    total_traces = 0
    for batch_num in range(1, 9):  # batch_1 to batch_8
        batch_folder = os.path.join(batch_dir, f'batch_{batch_num}')
        trace_file = os.path.join(batch_folder, f'traces_{batch_num}.json')
        
        if not os.path.exists(trace_file):
            print(f"   ⚠️  Warning: {trace_file} not found, skipping...")
            continue
        
        print(f"   Processing batch_{batch_num}...")
        with open(trace_file, 'r', encoding='utf-8') as f:
            try:
                traces = json.load(f)  # Array of trace objects
                total_traces += len(traces)
                
                for trace_data in traces:
                    spans = trace_data.get('spans', [])
                    
                    for span in spans:
                        attributes = span.get('attributes', {})
                        if attributes.get('vhas.span.type') == 'orchestrator_decision':
                            
                            # Extract State and Action
                            state_description = attributes.get('vhas.orchestrator.input_state')
                            action_name = attributes.get('vhas.orchestrator.action_selected')

                            # Lookup full action description from the agent corpus
                            action_description = agent_corpus_map.get(action_name)

                            # Only add pair when both state and action description are present
                            if state_description and action_description:
                                training_examples.append({"texts": [state_description, action_description]})

            except (json.JSONDecodeError, KeyError) as e:
                print(f"   ⚠️  Warning: Could not process {trace_file}. Error: {e}")
                continue
    
    print(f"\n✓ Processed {total_traces} traces from 8 batches")
    print(f"✓ Generated {len(training_examples)} (State, Action) pairs")

    # 3. Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nSaving {len(training_examples)} fine-tuning examples to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2)
        
    print("--- WRL Fine-tuning Data Preparation Complete! ---")
    print("You now have the specialized 'textbooks' to fine-tune the Encoder.")

if __name__ == "__main__":
    # Build relative paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    finetuning_dir = os.path.dirname(script_dir)  
    data_root = os.path.dirname(finetuning_dir)  
    vhas_root = os.path.dirname(data_root)       

    # Input paths
    BATCH_DIR = os.path.join(data_root, 'scenarios', 'data')
    UNIVERSE_FILE = os.path.join(vhas_root, 'vhas-demo', 'backend', 'vhas_universe.json')

    # Output path
    OUTPUT_FILE = os.path.join(finetuning_dir, 'data', 'wrl_finetuning_examples.json')

    print(f"Input paths:")
    print(f"   Batch dir: {BATCH_DIR}")
    print(f"   Universe: {UNIVERSE_FILE}")
    print(f"   Output: {OUTPUT_FILE}\n")

    create_finetuning_pairs(
        batch_dir=BATCH_DIR,
        universe_file=UNIVERSE_FILE,
        output_file=OUTPUT_FILE
    )