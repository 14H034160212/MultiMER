import os, json, argparse
import numpy as np, pandas as pd, torch, soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

DATA_DIR   = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
AUDIO_DIR  = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/audio"
OUT_DIR    = "/data/multimer/ACMMM2026/results/track2"
VAL_DIR    = "/data/multimer/ACMMM2026/track2/pseudo_val"
MODEL_PATH = "/home/multimer/.cache/huggingface/hub/models--google--gemma-4-12b-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205"

SYSTEM = (
    "You are an expert in fine-grained emotion recognition. "
    "Analyze the audio clip and subtitle to predict emotions. "
    "Output ONLY 3-5 comma-separated English emotion words. "
    "Use standard emotion vocabulary. No explanation, no markdown."
)

TARGET_SR = 16000
MAX_AUDIO_SEC = 30

def load_audio(name):
    path = f"{AUDIO_DIR}/{name}.wav"
    if not os.path.exists(path):
        return None
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != TARGET_SR:
        n_out = int(len(data) * TARGET_SR / sr)
        data = np.interp(np.linspace(0, len(data)-1, n_out),
                         np.arange(len(data)), data).astype(np.float32)
    max_samples = MAX_AUDIO_SEC * TARGET_SR
    if len(data) > max_samples:
        data = data[:max_samples]
    return data

def clean_output(text):
    lines = text.strip().split("\n")
    first_line = lines[0] if lines else text
    words = [w.strip().lower() for w in first_line.split(",")
             if w.strip() and 2 < len(w.strip()) < 25]
    return ",".join(words[:5]) if words else "neutral"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",    type=int, default=6)
    p.add_argument("--suffix", default="gemma4audio")
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--part",   type=int, default=0, help="0-indexed part when splitting across GPUs")
    p.add_argument("--n_parts",type=int, default=1, help="Total number of parts")
    args = p.parse_args()

    out_path = (f"{OUT_DIR}/predictions_{args.suffix}.jsonl" if not args.holdout
                else f"{VAL_DIR}/holdout_predictions_{args.suffix}.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["name"])
                except:
                    pass

    if args.holdout:
        split = pd.read_csv(f"{VAL_DIR}/holdout_split.csv")
        names = split[split.is_holdout]["name"].tolist()
    else:
        df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
        names = df_cand["name"].tolist()

    if args.n_parts > 1:
        names = [n for i, n in enumerate(names) if i % args.n_parts == args.part]

    sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    todo = [n for n in names if n not in done]
    print(f"{args.suffix}: {len(done)} done, {len(todo)} remaining", flush=True)
    if not todo:
        print("All done!"); return

    device = f"cuda:{args.gpu}"
    print(f"Loading Gemma4-12B Unified on {device} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    print(f"Loaded. Running inference on {len(todo)} samples ...", flush=True)

    errors = 0
    with open(out_path, "a") as fout:
        for name in tqdm(todo, desc=args.suffix):
            audio = load_audio(name)
            subtitle = str(sub.loc[name, "chinese"]) if name in sub.index else ""
            if subtitle == "nan":
                subtitle = ""

            try:
                if audio is not None:
                    user_text = f'Subtitle: "{subtitle[:300]}"\nEmotions:'
                    messages = [
                        {"role": "user", "content": [
                            {"type": "audio", "audio": audio},
                            {"type": "text",  "text": user_text},
                        ]},
                    ]
                    inputs = processor(
                        text=processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True),
                        audio=[audio],
                        sampling_rate=TARGET_SR,
                        return_tensors="pt",
                    ).to(device)
                else:
                    user_text = f'Subtitle: "{subtitle[:300]}"\nEmotions:'
                    messages = [
                        {"role": "user", "content": user_text},
                    ]
                    inputs = processor(
                        text=processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True),
                        return_tensors="pt",
                    ).to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=40,
                        do_sample=False,
                    )
                text_out = processor.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred = clean_output(text_out)

            except Exception as e:
                errors += 1
                pred = "neutral"
                print(f"Error {name}: {e}", flush=True)

            fout.write(json.dumps({"name": name, "prediction": pred}) + "\n")
            fout.flush()

    print(f"Done. Errors: {errors}/{len(todo)}", flush=True)

if __name__ == "__main__":
    main()
