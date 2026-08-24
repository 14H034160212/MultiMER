import os, json, argparse, traceback
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, AutoProcessor

DATA_DIR  = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
AUDIO_DIR = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/audio"
OUT_DIR   = "/data/multimer/ACMMM2026/results/track2"
MODEL_ID  = "/home/multimer/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

SYSTEM = (
    "You are an emotion recognition expert. "
    "Output ONLY 3-5 comma-separated English emotion words. "
    "No markdown, no explanation, no prefix — just the words."
)

FEW_SHOT_PAIRS = [
    ("你给我出去！现在！马上给我滚！", "angry,furious,irritated,hostile"),
    ("哈哈真的假的！太棒了！",           "happy,excited,joyful,thrilled"),
    ("他走了，就这么走了，再也回不来了",  "sad,heartbroken,grieving,sorrowful"),
    ("万一出错了怎么办，我心里很不安",    "anxious,worried,nervous,uneasy"),
]

def load_audio(path, target_sr=16000):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != target_sr:

        n_out = int(len(data) * target_sr / sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, n_out),
            np.arange(len(data)), data
        ).astype(np.float32)
    return data

def clean_output(text):
    import re
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    blocks = re.findall(r'```(?:\w*\n)?(.*?)```', text, re.DOTALL)
    if blocks:
        text = blocks[0].strip()

    text = re.sub(r'^(?:assistant|user)\s*:', '', text.strip(), flags=re.I).strip()

    skip_phrases = ("based on", "the audio", "the subtitle", "in the", "the speaker",
                    "emotion", "predict", "output")
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if any(p in low for p in skip_phrases):
            continue

        line = re.sub(r'^emotions?\s*:', '', line, flags=re.I).strip()
        words = [w.strip().lower().replace(" ", "_").replace("-", "_")
                 for w in line.split(",") if w.strip() and 2 < len(w.strip()) < 25]
        if len(words) >= 2:
            return ",".join(words[:6])

    words = [w.strip().lower().replace(" ", "_").replace("-", "_")
             for w in re.split(r'[,\n]+', text) if w.strip() and 2 < len(w.strip()) < 25
             and not any(p in w.lower() for p in skip_phrases)]
    return ",".join(words[:6]) or "neutral"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus",   type=str, default="0,6",
                   help="comma-separated GPU ids for device_map=auto")
    p.add_argument("--shard",  type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--out_suffix", default="qwen3omni")
    p.add_argument("--test", action="store_true", help="Run on first 5 samples only")
    args = p.parse_args()

    out_path = f"{OUT_DIR}/predictions_{args.out_suffix}.jsonl"
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try: done.add(json.loads(line)["name"])
                except: pass
    print(f"Already done: {len(done)}")

    df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    all_names = df_cand["name"].tolist()
    if args.test:
        all_names = all_names[:5]

    all_names = [n for i, n in enumerate(all_names)
                 if i % args.n_shards == args.shard and n not in done]
    print(f"Shard {args.shard}/{args.n_shards}: {len(all_names)} to process")
    if not all_names:
        print("All done!"); return

    sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")

    gpu_ids = [int(g) for g in args.gpus.split(",")]

    max_memory = {gid: "54GiB" for gid in gpu_ids}
    max_memory["cpu"] = "20GiB"
    print(f"Loading Qwen3-Omni-30B across GPUs {gpu_ids} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    device_str = f"cuda:{gpu_ids[0]}"
    model.eval()

    model.disable_talker()
    print("Model loaded (talker disabled).")

    errors = 0
    with open(out_path, "a") as fout:
        for name in tqdm(all_names, desc=f"shard{args.shard}"):
            audio_path = f"{AUDIO_DIR}/{name}.wav"
            if not os.path.exists(audio_path):
                record = {"name": name, "prediction": "neutral", "status": "no_audio"}
                fout.write(json.dumps(record) + "\n"); fout.flush()
                continue

            try:
                audio_arr = load_audio(audio_path)
                subtitle = str(sub.loc[name, "chinese"]) if name in sub.index else ""
                if subtitle == "nan": subtitle = ""

                messages = [{"role": "system", "content": SYSTEM}]
                for sub_ex, emo_ex in FEW_SHOT_PAIRS:
                    messages.append({"role": "user",      "content": f'Subtitle: "{sub_ex}"'})
                    messages.append({"role": "assistant", "content": emo_ex})
                messages.append({"role": "user", "content": [
                    {"type": "audio", "audio": audio_arr},
                    {"type": "text",  "text": f'Subtitle: "{subtitle[:300]}"'},
                ]})

                text_in = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = processor(
                    text=text_in, audio=audio_arr, sampling_rate=16000,
                    return_tensors="pt",
                )

                inputs = {k: v.to(device_str) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_ids = model.generate(
                        **inputs,
                        return_audio=False,
                        thinker_max_new_tokens=40,
                        do_sample=False,
                    )
                new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                text_out = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
                prediction = clean_output(text_out)

                record = {"name": name, "prediction": prediction, "status": "ok"}
                fout.write(json.dumps(record) + "\n"); fout.flush()

                if args.test:
                    print(f"  {name}: {subtitle[:60]!r} → raw={text_out!r} → {prediction}")

            except Exception as e:
                errors += 1
                record = {"name": name, "prediction": "neutral", "status": f"error:{e}"}
                fout.write(json.dumps(record) + "\n"); fout.flush()
                if args.test:
                    print(f"  ERROR {name}: {e}")
                    traceback.print_exc()

    print(f"Done. Errors: {errors}")

if __name__ == "__main__":
    main()
