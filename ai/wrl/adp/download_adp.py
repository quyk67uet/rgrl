"""Utility to download selected ADP dataset subsets from Hugging Face."""

from huggingface_hub import hf_hub_download
import os

TARGET_SUBSETS = [
    "orca_agentinstruct",
    "agenttuning_db",
    "swe-smith",
]

LOCAL_DATA_DIR = "data/adp_pretraining"
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

print("Starting ADP data download...")
for subset in TARGET_SUBSETS:
    print(f"Downloading subset {subset}...")
    try:
        # Download the standard-format file `full_std.jsonl`
        hf_hub_download(
            repo_id="neulab/agent-data-collection",
            repo_type="dataset",
            filename=f"{subset}/full_std.jsonl",
            local_dir=LOCAL_DATA_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded: {subset}")
    except Exception as e:
        print(f"Error downloading {subset}: {e}")

print("ADP data download complete.")