# merge_data.py
import json
import os

def merge_sequence_files():
    """Gộp các file sequence từ ADP và T1 thành một file duy nhất."""

    # Xây dựng đường dẫn tương đối một cách an toàn dựa trên vị trí file script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pretraining_dir = os.path.dirname(script_dir)  # d:/VHAS/data/pretraining

    ADP_FILE = os.path.join(pretraining_dir, 'adp', 'data', 'pretraining_sequences_adp.json')
    T1_FILE = os.path.join(pretraining_dir, 'T1', 'pretraining_sequences_t1.json')
    # Lưu file output vào thư mục pretraining/data như ảnh minh họa
    OUTPUT_FILE = os.path.join(pretraining_dir, 'data', 'pretraining_sequences_combined.json')
    
    print("--- Starting: Merging sequence files ---")
    
    all_sequences = []
    
    # Tải dữ liệu từ ADP
    if os.path.exists(ADP_FILE):
        with open(ADP_FILE, 'r', encoding='utf-8') as f:
            adp_data = json.load(f)
            all_sequences.extend(adp_data)
            print(f"Loaded {len(adp_data)} sequences from ADP.")
    else:
        print(f"WARNING: ADP file not found at {ADP_FILE}")

    # Tải dữ liệu từ T1
    if os.path.exists(T1_FILE):
        with open(T1_FILE, 'r', encoding='utf-8') as f:
            t1_data = json.load(f)
            all_sequences.extend(t1_data)
            print(f"Loaded {len(t1_data)} sequences from T1.")
    else:
        print(f"WARNING: T1 file not found at {T1_FILE}")
        
    # Lưu file đã gộp
    print(f"\nSaving a total of {len(all_sequences)} combined sequences to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_sequences, f)
        
    print("--- Merging Complete! ---")

if __name__ == "__main__":
    merge_sequence_files()