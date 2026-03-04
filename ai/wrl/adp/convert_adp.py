# convert_adp.py
import json
import os
import re
from tqdm import tqdm
from itertools import groupby

def parse_sql_action(sql_content: str) -> str:
    """Trích xuất loại hành động chính từ một câu lệnh SQL."""
    sql_content = sql_content.strip().upper()
    if sql_content.startswith("SELECT"):
        return "SELECT_JOIN" if "JOIN" in sql_content else "SELECT"
    for keyword in ["UPDATE", "INSERT", "DELETE", "CREATE", "DROP", "ALTER"]:
        if sql_content.startswith(keyword):
            return keyword
    return "EXECUTE_SQL"

def parse_bash_action(bash_content: str) -> str:
    """Trích xuất lệnh chính đầu tiên từ một chuỗi lệnh bash."""
    cleaned_content = re.sub(r'^\s*cd .*? &&\s*', '', bash_content.strip())
    parts = cleaned_content.split()
    return parts[0] if parts else "execute_bash"

def parse_generic_code_action(code_content: str) -> str:
    """Heuristic để tìm tên hàm/class được định nghĩa hoặc gọi."""
    # Ưu tiên tìm định nghĩa hàm/class
    def_match = re.search(r'(?:def|fn|function|class)\s+([\w_]+)', code_content)
    if def_match:
        return f"define:{def_match.group(1)}"
    
    # Nếu không, tìm lệnh gọi hàm
    call_match = re.search(r'([\w_]+)\(', code_content)
    if call_match:
        return call_match.group(1)
        
    return "execute_code"

def extract_action_sequences(adp_std_file_path: str) -> list[list[str]]:
    """
    Đọc file full_std.jsonl và trích xuất chuỗi hành động, xử lý đa ngôn ngữ và các loại action.
    """
    action_sequences = []
    print(f"Processing file: {os.path.basename(adp_std_file_path)}")
    
    try:
        with open(adp_std_file_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for line in f)
    except Exception as e:
        print(f"  Error counting lines: {e}")
        return []

    with open(adp_std_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc="Extracting sequences"):
            try:
                data = json.loads(line)
                raw_sequence = []
                for step in data.get("content", []):
                    action_name = None
                    step_class = step.get("class_")
                    
                    if step_class == "api_action":
                        func_name = step.get("function")
                        if not func_name: continue
                        
                        kwargs = step.get("kwargs", {})
                        command = kwargs.get("command") if isinstance(kwargs, dict) else None
                        
                        # Bỏ qua hành động submit cuối cùng
                        if func_name == "submit": continue

                        action_name = f"{func_name}:{command}" if command else func_name
                    
                    elif step_class == "code_action":
                        lang = step.get("language", "code")
                        content = step.get("content", "")
                        if not content: continue

                        parsed_func = ""
                        if lang == "mysql":
                            parsed_func = parse_sql_action(content)
                        elif lang == "bash":
                            parsed_func = parse_bash_action(content)
                        else:
                            parsed_func = parse_generic_code_action(content)
                            
                        action_name = f"{lang}:{parsed_func}"
                    
                    # Lọc bỏ các message_action không mang thông tin
                    elif step_class == "message_action":
                        content = step.get("content", "")
                        if content and "<finish>" in content:
                            continue # Bỏ qua các message kết thúc

                    if action_name:
                        raw_sequence.append(action_name)
                
                # --- Clean up the sequence ---
                # 1. Loại bỏ các hành động lặp lại liên tiếp
                cleaned_sequence = [k for k, g in groupby(raw_sequence)]
                
                # 2. Chỉ thêm vào nếu chuỗi không rỗng
                if cleaned_sequence:
                    action_sequences.append(cleaned_sequence)

            except (json.JSONDecodeError, AttributeError):
                continue
                
    return action_sequences

if __name__ == "__main__":
    ADP_DATA_DIR = "data/adp_pretraining"
    TARGET_SUBSETS = [
        "orca_agentinstruct",
        "agenttuning_db",
        "swe-smith",
    ]
    
    all_sequences = []
    
    print("--- Starting Stage 1A: Extracting Action Sequences from ADP (Final Version) ---")
    for subset in TARGET_SUBSETS:
        file_path = os.path.join(ADP_DATA_DIR, subset, "full_std.jsonl")
        
        if os.path.exists(file_path):
            sequences = extract_action_sequences(file_path)
            all_sequences.extend(sequences)
            print(f"  Extracted {len(sequences)} sequences from '{subset}'. Total sequences so far: {len(all_sequences)}")
        else:
            print(f"WARNING: File not found for subset '{subset}' at: {file_path}")

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pretraining_sequences_adp.json")
    
    print(f"\nSaving a total of {len(all_sequences)} action sequences to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_sequences, f, indent=2)
        
    print("--- Stage 1A Complete! ---")
    print("You now have the high-quality 'fuel' to start Stage 1B: Training the Encoder.")