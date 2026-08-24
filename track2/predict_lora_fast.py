import os, json, argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration
from peft import PeftModel

DATA_DIR  = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
OUT_DIR   = "/data/multimer/ACMMM2026/results/track2"
VAL_DIR   = "/data/multimer/ACMMM2026/track2/pseudo_val"
TRAIN_DIR = "/data/multimer/ACMMM2026/track2/train_extract"
MODEL_ID  = "/home/multimer/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

SYSTEM = (
    "You are an emotion recognition expert. "
    "Output ONLY 3-5 comma-separated English emotion words. "
    "No markdown, no explanation — just the words."
)

LORA_CONFIGS = {
    "qwen3omni_lora_full":     f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch1",
    "qwen3omni_lora_human":    f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_human_epoch1",
    "qwen3omni_lora_full_ep2": f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch2",
    "qwen3omni_lora_full_ep3": f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch3",
    "qwen3omni_lora_full_ep4": f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch4",
}

def clean_output(text):
    import re
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    skip = ("based on", "the audio", "the subtitle", "emotion", "predict", "output", "in the")
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if any(p in low for p in skip):
            continue
        line = re.sub(r'^emotions?\s*:', '', line, flags=re.I).strip()
        words = [w.strip().lower() for w in line.split(",")
                 if w.strip() and 2 < len(w.strip()) < 25]
        if len(words) >= 2:
            return ",".join(words[:5])
    words = [w.strip().lower() for w in re.split(r'[,\n]+', text)
             if w.strip() and 2 < len(w.strip()) < 25
             and not any(p in w.lower() for p in skip)]
    return ",".join(words[:5]) or "neutral"

def build_prompt(name, sub_df, processor):
    subtitle = ""
    if name in sub_df.index:
        s = str(sub_df.loc[name, "chinese"])
        if s != "nan":
            subtitle = s[:300]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f'Subtitle: "{subtitle}"\nEmotions:'},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus",       type=str, default="1,0",
                   help="comma-separated GPU ids; first is primary")
    p.add_argument("--adapter",    default="qwen3omni_lora_human",
                   choices=list(LORA_CONFIGS.keys()))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--holdout",    action="store_true")
    args = p.parse_args()

    suffix   = args.adapter
    out_path = (f"{OUT_DIR}/predictions_{suffix}.jsonl" if not args.holdout
                else f"{VAL_DIR}/holdout_predictions_{suffix}.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["name"])
                except Exception:
                    pass

    if args.holdout:
        split = pd.read_csv(f"{VAL_DIR}/holdout_split.csv")
        names = split[split.is_holdout]["name"].tolist()
    else:
        df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
        names = df_cand["name"].tolist()

    sub_df = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    todo   = [n for n in names if n not in done]
    print(f"{suffix}: {len(done)} done, {len(todo)} remaining")
    if not todo:
        print("All done!"); return

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    primary_device = torch.device(f"cuda:{gpu_ids[0]}")
    lora_dir = LORA_CONFIGS[args.adapter]

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    if len(gpu_ids) == 1:
        device_map = {"": primary_device}
        max_mem = None
        print(f"Loading Qwen3-Omni-30B on GPU {gpu_ids[0]} ...")
    else:

        max_mem = {gpu_ids[0]: "49500MiB", gpu_ids[1]: "15GiB", "cpu": "20GiB"}
        device_map = "auto"
        print(f"Loading Qwen3-Omni-30B across GPUs {gpu_ids} with max_memory={max_mem} ...")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_mem if len(gpu_ids) > 1 else None,
        trust_remote_code=True,
    )
    model.disable_talker()
    print(f"Applying LoRA from {lora_dir} ...")
    model.thinker = PeftModel.from_pretrained(model.thinker, lora_dir)
    model.eval()
    print(f"Ready. Running {len(todo)} samples with batch_size={args.batch_size} ...")

    errors = 0
    with open(out_path, "a") as fout:
        for i in tqdm(range(0, len(todo), args.batch_size), desc=suffix):
            batch_names   = todo[i : i + args.batch_size]
            batch_prompts = [build_prompt(n, sub_df, processor) for n in batch_names]

            try:
                enc = processor.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(primary_device)

                with torch.no_grad():
                    out_ids = model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        return_audio=False,
                        thinker_max_new_tokens=25,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                    )

                in_len = enc["input_ids"].shape[1]
                for j, name in enumerate(batch_names):
                    new_ids  = out_ids[j][in_len:]
                    text_out = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
                    pred     = clean_output(text_out)
                    fout.write(json.dumps({"name": name, "prediction": pred, "status": "ok"}) + "\n")
                fout.flush()

            except Exception as e:
                errors += len(batch_names)
                print(f"Batch error at i={i}: {e}", flush=True)
                for name in batch_names:
                    fout.write(json.dumps({"name": name, "prediction": "neutral", "status": f"error"}) + "\n")
                fout.flush()

    print(f"Done. Errors: {errors}/{len(todo)}")

if __name__ == "__main__":
    main()
