import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

DATA_DIR = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VIDEO_ROOT = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/video"
OUT_DIR = "/data/multimer/ACMMM2026/track2/train_extract"
MODEL_ID = "Qwen/Qwen3.5-9B"
N_FRAMES = 2
N_TRAIN_SAMPLES = 3000
EPOCHS = 1
GRAD_ACCUM = 8
LR = 1e-4

SYSTEM = (
    "You are an expert in fine-grained multimodal emotion recognition. "
    "Given a short video clip showing a listener's reaction and their subtitle (what the speaker said), "
    "predict the listener's emotions as 3-6 comma-separated English emotion words. "
    "Be specific and nuanced. No numbers, no explanation."
)

def parse(s):
    s = str(s).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [w.strip().lower() for w in s.split(",") if w.strip()]

def sample_frames_pil(video_path, n=N_FRAMES, size=448):
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        scale = min(size / max(h, w), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames

def build_examples():
    th = pd.read_csv(f"{DATA_DIR}/track2_train_human.csv")
    tm = pd.read_csv(f"{DATA_DIR}/track2_train_mercaptionplus.csv")
    sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")

    rows = []
    for _, r in th.iterrows():
        rows.append((r["name"], "human", parse(r["openset"])))
    for _, r in tm.iterrows():
        rows.append((r["name"], "mercaptionplus", parse(r["openset"])))

    random.Random(42).shuffle(rows)

    examples = []
    for name, src, words in rows:
        if len(examples) >= N_TRAIN_SAMPLES:
            break
        video_path = f"{VIDEO_ROOT}/track2_train_{src}/video/{name}.mp4"
        if not os.path.exists(video_path):
            continue
        text = str(sub.loc[name, "chinese"]) if name in sub.index else ""
        if text == "nan" or not text:
            continue
        if not words:
            continue
        examples.append({"name": name, "video_path": video_path, "subtitle": text[:300], "target": ", ".join(words[:6])})
    print(f"Built {len(examples)} training examples")
    return examples

def build_input_ids(processor, subtitle, frames, target, device):
    text_prompt = f'Subtitle: "{subtitle}"\nEmotions:'
    user_content = [{"type": "image", "image": img} for img in frames]
    user_content.append({"type": "text", "text": text_prompt})
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full_text = prompt_text + " " + target

    prompt_inputs = processor(text=[prompt_text], images=frames if frames else None, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=frames if frames else None, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
    labels = labels.to(device)
    return full_inputs, labels

def main():
    device = torch.device("cuda:5")
    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    examples = build_examples()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    model.train()
    step = 0
    for epoch in range(EPOCHS):
        random.Random(epoch).shuffle(examples)
        optimizer.zero_grad()
        total_loss, n_loss = 0.0, 0
        for i, ex in enumerate(examples):
            frames = sample_frames_pil(ex["video_path"])
            try:
                inputs, labels = build_input_ids(processor, ex["subtitle"], frames, ex["target"], device)
                out = model(**inputs, labels=labels)
                loss = out.loss / GRAD_ACCUM
                loss.backward()
                total_loss += out.loss.item()
                n_loss += 1
            except Exception as e:
                print(f"skip {ex['name']}: {e}", flush=True)
                continue

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            if (i + 1) % 200 == 0:
                print(f"epoch {epoch+1} [{i+1}/{len(examples)}] avg_loss={total_loss/max(n_loss,1):.4f}", flush=True)

        print(f"=== epoch {epoch+1} done, avg_loss={total_loss/max(n_loss,1):.4f} ===", flush=True)
        save_dir = f"{OUT_DIR}/qwen_lora_epoch{epoch+1}"
        model.save_pretrained(save_dir)
        print(f"saved LoRA adapter to {save_dir}")

if __name__ == "__main__":
    main()
