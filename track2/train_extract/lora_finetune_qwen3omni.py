import json, os, random, argparse

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor
from tqdm import tqdm

DATA_DIR  = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
AUDIO_DIR = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/audio"
OUT_DIR   = "/data/multimer/ACMMM2026/track2/train_extract"
VAL_DIR   = "/data/multimer/ACMMM2026/track2/pseudo_val"
WHEEL_NPZ = "/data/multimer/ACMMM2026/MERTools/MER2026/MER2026_Track2/emotion_wheel/wheel_mapping.npz"
MODEL_ID  = "/home/multimer/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

GRAD_ACCUM   = 8
LR           = 5e-5
HUMAN_WEIGHT = 1.0

SYSTEM = (
    "You are an emotion recognition expert. "
    "Output ONLY 3-5 comma-separated English emotion words. "
    "No markdown, no explanation — just the words."
)

def parse(s):
    s = str(s).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [w.strip().lower() for w in s.split(",") if w.strip()]

def load_audio(name, target_sr=16000):
    path = f"{AUDIO_DIR}/{name}.wav"
    if not os.path.exists(path):
        return None
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != target_sr:
        n_out = int(len(data) * target_sr / sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, n_out),
            np.arange(len(data)), data
        ).astype(np.float32)
    return data

def build_examples(n_target, mercap_weight):
    d = np.load(WHEEL_NPZ, allow_pickle=True)
    format_mapping = d["format_mapping"].tolist()

    split = pd.read_csv(f"{VAL_DIR}/holdout_split.csv")
    holdout_names = set(split[split.is_holdout]["name"])
    print(f"Excluding held-out pool: {len(holdout_names)} names")

    th = pd.read_csv(f"{DATA_DIR}/track2_train_human.csv")
    tm = pd.read_csv(f"{DATA_DIR}/track2_train_mercaptionplus.csv")
    sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")

    human_rows  = [(r["name"], "human",        parse(r["openset"])) for _, r in th.iterrows() if r["name"] not in holdout_names]
    mercap_rows = [(r["name"], "mercaptionplus", parse(r["openset"])) for _, r in tm.iterrows() if r["name"] not in holdout_names]

    random.Random(42).shuffle(human_rows)
    random.Random(42).shuffle(mercap_rows)
    all_rows = human_rows + mercap_rows

    examples = []
    for name, src, words in all_rows:
        if len(examples) >= n_target:
            break
        audio_path = f"{AUDIO_DIR}/{name}.wav"
        if not os.path.exists(audio_path):
            continue
        text = str(sub.loc[name, "chinese"]) if name in sub.index else ""
        if text == "nan" or not text:
            continue
        vocab_words = [w for w in words if w in format_mapping]
        target_words = vocab_words[:5] if vocab_words else words[:5]
        if not target_words:
            continue
        weight = HUMAN_WEIGHT if src == "human" else mercap_weight
        examples.append({
            "name": name, "subtitle": text[:300],
            "target": ", ".join(target_words),
            "weight": weight, "source": src,
        })

    n_human = sum(1 for e in examples if e["source"] == "human")
    print(f"Built {len(examples)} examples ({n_human} human, {len(examples)-n_human} mercaptionplus)")
    return examples

def build_input_ids(processor, subtitle, audio, target, device):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio, "sample_rate": 16000},
            {"type": "text",  "text": f'Subtitle: "{subtitle}"\nEmotions:'},
        ]},
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = prompt_text + " " + target

    prompt_inputs = processor(
        text=[prompt_text], audio=[audio], sampling_rate=16000, return_tensors="pt")
    full_inputs = processor(
        text=[full_text], audio=[audio], sampling_rate=16000, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    full_inputs = {
        k: v.to(device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(device)
        for k, v in full_inputs.items()
    }
    labels = labels.to(device)
    return full_inputs, labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_target",       type=int,   default=999999)
    p.add_argument("--epochs",         type=int,   default=1)
    p.add_argument("--gpu",            type=int,   default=5)
    p.add_argument("--lora_r",         type=int,   default=32)
    p.add_argument("--mercap_weight",  type=float, default=0.2)
    p.add_argument("--ckpt_every",     type=int,   default=500)
    p.add_argument("--tag",            default="stage3_qwen3omni_r32")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Loading Qwen3-Omni-30B on GPU {args.gpu} (single-card)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.disable_talker()

    thinker = model.thinker
    thinker.gradient_checkpointing_enable()
    thinker.enable_input_require_grads()

    ckpt_dir      = f"{OUT_DIR}/qwen_lora_{args.tag}_ckpt"
    epoch1_dir    = f"{OUT_DIR}/qwen_lora_{args.tag}_epoch1"
    progress_path = f"{ckpt_dir}/progress.json"
    resume_epoch, resume_step = 0, 0

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        resume_epoch = progress["epoch"] - 1
        resume_step  = progress["step"]
        print(f"[resume] epoch {progress['epoch']} step {resume_step}/{progress['total']}")
        thinker = PeftModel.from_pretrained(thinker, ckpt_dir, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        thinker = get_peft_model(thinker, lora_config)
    thinker.print_trainable_parameters()

    examples  = build_examples(args.n_target, args.mercap_weight)
    optimizer = torch.optim.AdamW(
        [p for p in thinker.parameters() if p.requires_grad], lr=LR)
    if os.path.exists(progress_path) and os.path.exists(f"{ckpt_dir}/optimizer.pt"):
        optimizer.load_state_dict(
            torch.load(f"{ckpt_dir}/optimizer.pt", map_location=device))
        print("[resume] restored optimizer")

    thinker.train()
    for epoch in range(resume_epoch, args.epochs):
        random.Random(epoch).shuffle(examples)
        start_i = resume_step if epoch == resume_epoch else 0
        optimizer.zero_grad()
        total_loss, n_loss = 0.0, 0

        for i, ex in enumerate(tqdm(examples, desc=f"epoch {epoch+1}")):
            if i < start_i:
                continue

            audio = load_audio(ex["name"])
            if audio is None:
                continue
            try:
                inputs, labels = build_input_ids(
                    processor, ex["subtitle"], audio, ex["target"], device)

                out  = thinker(**inputs, labels=labels)
                loss = out.loss * ex["weight"] / GRAD_ACCUM
                loss.backward()
                total_loss += out.loss.item()
                n_loss += 1
            except Exception as e:
                print(f"skip {ex['name']}: {e}", flush=True)
                optimizer.zero_grad()
                continue

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(thinker.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % 200 == 0:
                avg = total_loss / max(n_loss, 1)
                print(f"epoch {epoch+1} [{i+1}/{len(examples)}] avg_loss={avg:.4f}", flush=True)

            if (i + 1) % args.ckpt_every == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                thinker.save_pretrained(ckpt_dir)
                torch.save(optimizer.state_dict(), f"{ckpt_dir}/optimizer.pt")
                with open(progress_path, "w") as f:
                    json.dump({"epoch": epoch + 1, "step": i + 1,
                               "total": len(examples)}, f)
                print(f"[checkpoint] saved {i+1}/{len(examples)} to {ckpt_dir}", flush=True)

        torch.nn.utils.clip_grad_norm_(thinker.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        avg = total_loss / max(n_loss, 1)
        print(f"=== epoch {epoch+1} done, avg_loss={avg:.4f} ===", flush=True)

        save_dir = f"{OUT_DIR}/qwen_lora_{args.tag}_epoch{epoch+1}"
        thinker.save_pretrained(save_dir)
        print(f"saved LoRA adapter (thinker) to {save_dir}", flush=True)

        with open(progress_path, "w") as pf:
            json.dump({"epoch": epoch + 1, "step": len(examples),
                       "total": len(examples)}, pf)

if __name__ == "__main__":
    main()
