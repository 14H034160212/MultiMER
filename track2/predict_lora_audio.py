import os, json, argparse
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

DATA_DIR  = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
AUDIO_DIR = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/audio"
OUT_DIR   = "/data/multimer/ACMMM2026/results/track2"
VAL_DIR   = "/data/multimer/ACMMM2026/track2/pseudo_val"
TRAIN_DIR = "/data/multimer/ACMMM2026/track2/train_extract"
MODEL_ID  = "/home/multimer/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

FIXED_LEN = 8 * 16000
SR = 16000

SYSTEM = (
    "You are an emotion recognition expert. "
    "Output ONLY 3-5 comma-separated English emotion words. "
    "No markdown, no explanation — just the words."
)

LORA_CONFIGS = {
    "qwen3omni_lora_full":    f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch1",
    "qwen3omni_lora_human":   f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_human_epoch1",
    "qwen3omni_lora_full_ep2": f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch2",
    "qwen3omni_lora_full_ep2_fp16": f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch2",
    "qwen3omni_lora_full_ep3": f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch3",
    "qwen3omni_lora_full_ep4": f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch4",
    "qwen3omni_lora_full_ep5": f"{TRAIN_DIR}/qwen_lora_stage3_qwen3omni_r32_epoch5",
}

AUDIO_SUFFIX = {
    "qwen3omni_lora_full":    "qwen3omni_lora_full_audio",
    "qwen3omni_lora_human":   "qwen3omni_lora_human_audio",
    "qwen3omni_lora_full_ep2": "qwen3omni_lora_full_ep2",
    "qwen3omni_lora_full_ep2_fp16": "qwen3omni_lora_full_ep2_fp16",
    "qwen3omni_lora_full_ep3": "qwen3omni_lora_full_ep3",
    "qwen3omni_lora_full_ep4": "qwen3omni_lora_full_ep4",
    "qwen3omni_lora_full_ep5": "qwen3omni_lora_full_ep5",
}

def load_audio_fixed(name):
    path = f"{AUDIO_DIR}/{name}.wav"
    if not os.path.exists(path):
        return np.zeros(FIXED_LEN, dtype=np.float32)
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SR:
        n = int(len(data) * SR / sr)
        data = np.interp(np.linspace(0, len(data)-1, n), np.arange(len(data)), data).astype(np.float32)
    if len(data) >= FIXED_LEN:
        return data[:FIXED_LEN]
    return np.pad(data, (0, FIXED_LEN - len(data)))

def clean_output(text):
    import re
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    skip = ("based on", "the audio", "the subtitle", "emotion", "predict", "output", "in the")
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(p in line.lower() for p in skip):
            continue
        line = re.sub(r'^emotions?\s*:', '', line, flags=re.I).strip()
        words = [w.strip().lower() for w in line.split(",")
                 if w.strip() and 2 < len(w.strip()) < 25]
        if len(words) >= 2:
            return ",".join(words[:5])
    words = [w.strip().lower() for w in __import__('re').split(r'[,\n]+', text)
             if w.strip() and 2 < len(w.strip()) < 25
             and not any(p in w.lower() for p in skip)]
    return ",".join(words[:5]) or "neutral"

def build_messages(name, sub_df, audio):
    subtitle = ""
    if name in sub_df.index:
        s = str(sub_df.loc[name, "chinese"])
        if s != "nan":
            subtitle = s[:300]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio, "sample_rate": SR},
            {"type": "text", "text": f'Subtitle: "{subtitle}"\nEmotions:'},
        ]},
    ]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",        type=int, default=5)
    p.add_argument("--adapter",    default="qwen3omni_lora_human",
                   choices=list(LORA_CONFIGS.keys()))
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--holdout",    action="store_true")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load model in 4-bit NF4 quant (~15GB instead of 60GB)")
    args = p.parse_args()

    suffix    = AUDIO_SUFFIX[args.adapter]
    out_path  = (f"{OUT_DIR}/predictions_{suffix}.jsonl" if not args.holdout
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

    device = torch.device(f"cuda:{args.gpu}")
    lora_dir = LORA_CONFIGS[args.adapter]
    quant_str = " [4-bit NF4]" if args.load_in_4bit else " [BF16]"
    print(f"Loading Qwen3-Omni-30B on GPU {args.gpu}{quant_str} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.disable_talker()
    print(f"Applying LoRA from {lora_dir} ...")
    model.thinker = PeftModel.from_pretrained(model.thinker, lora_dir)
    model.eval()
    print(f"Ready. batch_size={args.batch_size}, fixed audio={FIXED_LEN/SR:.0f}s")

    errors = 0
    with open(out_path, "a") as fout:
        for i in tqdm(range(0, len(todo), args.batch_size), desc=suffix):
            batch_names  = todo[i : i + args.batch_size]
            batch_audios = [load_audio_fixed(n) for n in batch_names]
            batch_msgs   = [build_messages(n, sub_df, a) for n, a in zip(batch_names, batch_audios)]
            batch_texts  = [
                processor.apply_chat_template(m, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
                for m in batch_msgs
            ]

            try:
                inputs = processor(
                    text=batch_texts,
                    audio=batch_audios,
                    sampling_rate=SR,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {
                    k: v.to(device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(device)
                    for k, v in inputs.items()
                }

                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        return_audio=False,
                        thinker_max_new_tokens=25,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                    )

                if isinstance(out_ids, tuple):
                    out_ids = out_ids[0]

                in_len = inputs["input_ids"].shape[1]
                for j, name in enumerate(batch_names):
                    text_out = processor.tokenizer.decode(out_ids[j][in_len:], skip_special_tokens=True)
                    pred = clean_output(text_out)
                    fout.write(json.dumps({"name": name, "prediction": pred, "status": "ok"}) + "\n")
                fout.flush()

            except Exception as e:
                errors += len(batch_names)
                print(f"Batch error i={i}: {e}", flush=True)
                for name in batch_names:
                    fout.write(json.dumps({"name": name, "prediction": "neutral", "status": "error"}) + "\n")
                fout.flush()

    print(f"Done. Errors: {errors}/{len(todo)}")

if __name__ == "__main__":
    main()
