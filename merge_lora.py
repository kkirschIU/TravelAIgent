import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LORA_DIR = "outputs/travelaigent-qlora"
MERGED_DIR = "merged_travelaigent"

def main():
    Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

    # WICHTIG: device_map=None
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)

    model = PeftModel.from_pretrained(
        base_model,
        LORA_DIR,
        device_map=None,  # <- auch hier: kein dispatch/offload
    )

    model = model.merge_and_unload()
    model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"Merged model saved to {MERGED_DIR}")

if __name__ == "__main__":
    main()

