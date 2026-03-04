# scripts/prepare_pretraining_data.py

import json
import os
from sentence_transformers.readers import InputExample
from tqdm import tqdm

def create_pretraining_pairs(sequences_file: str, output_dir: str):
    """
    Tạo các cặp dương cho contrastive learning.
    """
    print("--- Starting WRL Data Preparation ---")
    
    # 1. Tải các chuỗi hành động
    print(f"Loading action sequences from {sequences_file}...")
    try:
        with open(sequences_file, 'r', encoding='utf-8') as f:
            all_sequences = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {sequences_file}. Please run previous steps.")
        return

    # 2. Tạo các InputExample (cặp dương)
    print(f"Generating positive pairs from {len(all_sequences)} sequences...")
    training_examples = []
    for sequence in tqdm(all_sequences, desc="Creating pairs"):
        if len(sequence) < 2:
            continue
            
        for i in range(len(sequence) - 1):
            anchor = sequence[i]
            positive = sequence[i+1]
            
            # Không tạo cặp cho các hành động tự lặp lại
            if anchor != positive:
                # Lưu các cặp dương dưới dạng dict đơn giản
                # Mô hình sẽ học biểu diễn trực tiếp từ các tên này.
                training_examples.append({"texts": [anchor, positive]})

    # 4. Lưu các file output
    os.makedirs(output_dir, exist_ok=True)

    examples_output_file = os.path.join(output_dir, 'wrl_pretraining_examples.json')

    print(f"Saving {len(training_examples)} training examples to '{examples_output_file}'...")
    with open(examples_output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f)
        
    print("--- WRL Data Preparation Complete! ---")

if __name__ == "__main__":
    # Xây dựng đường dẫn tương đối dựa trên vị trí script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pretraining_dir = os.path.dirname(script_dir)  # d:/VHAS/data/pretraining
    
    SEQUENCES_FILE = os.path.join(pretraining_dir, 'data', 'pretraining_sequences_combined.json')
    OUTPUT_DIR = os.path.join(pretraining_dir, 'data')
    
    create_pretraining_pairs(
        sequences_file=SEQUENCES_FILE,
        output_dir=OUTPUT_DIR
    )