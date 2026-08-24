import os, json, argparse
import cv2, numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DATA_DIR   = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VIDEO_ROOT = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/video/track1_track2_candidate/video"
OUT_DIR    = "/data/multimer/ACMMM2026/results/track2"
VAL_DIR    = "/data/multimer/ACMMM2026/track2/pseudo_val"
MODEL_ID   = "/home/multimer/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb"

SYSTEM = (
    "You are an expert in fine-grained multimodal emotion recognition. "
    "Given a short video clip showing a listener's reaction and their subtitle (what the speaker said), "
    "predict the listener's emotions as 3-5 comma-separated English emotion words. "
    "Use specific, standard emotion-wheel vocabulary. No numbers, no explanation."
)

def sample_frames_pil(video_path, n=4, size=448):
    if not video_path or not os.path.exists(video_path):
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
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def clean_output(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    lines = text.strip().split("\n")
    first_line = lines[0] if lines else text
    words = [w.strip().lower() for w in first_line.split(",")
             if w.strip() and 2 < len(w.strip()) < 25]
    return ",".join(words[:5]) if words else "neutral"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",      type=int, default=7)
    p.add_argument("--suffix",   default="qwen3vl32b")
    p.add_argument("--holdout",  action="store_true")
    p.add_argument("--n_frames", type=int, default=4)
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

    sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    todo = [n for n in names if n not in done]
    print(f"{args.suffix}: {len(done)} done, {len(todo)} remaining")
    if not todo:
        print("All done!"); return

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Loading Qwen3-VL-32B on GPU {args.gpu} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded. Running inference on {len(todo)} samples ...")

    errors = 0
    with open(out_path, "a") as fout:
        for name in tqdm(todo, desc=args.suffix):
            video_path = f"{VIDEO_ROOT}/{name}.mp4"
            frames = sample_frames_pil(video_path, n=args.n_frames)
            subtitle = str(sub.loc[name, "chinese"]) if name in sub.index else ""
            if subtitle == "nan":
                subtitle = ""

            try:
                if frames:
                    user_content = [{"type": "image", "image": img} for img in frames]
                    user_content.append(
                        {"type": "text", "text": f'Subtitle: "{subtitle[:300]}"\nEmotions:'})
                else:
                    user_content = [
                        {"type": "text", "text": f'Subtitle: "{subtitle[:300]}"\nEmotions:'}]

                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": user_content},
                ]
                text_in = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                inputs = processor(
                    text=[text_in],
                    images=frames if frames else None,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=40,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                text_out = processor.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred = clean_output(text_out)

            except Exception as e:
                errors += 1
                pred = "neutral"
                print(f"Error {name}: {e}", flush=True)

            fout.write(json.dumps({"name": name, "prediction": pred, "status": "ok"}) + "\n")
            fout.flush()

    print(f"Done. Errors: {errors}/{len(todo)}")

if __name__ == "__main__":
    main()
