import os, json, argparse
import cv2, numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
import re

DATA_DIR   = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VIDEO_ROOT = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/video/track1_track2_candidate/video"
OUT_DIR    = "/data/multimer/ACMMM2026/results/track2"

snap_dir = "/home/multimer/.cache/huggingface/hub/models--rhymes-ai--Aria/snapshots"
snap = os.listdir(snap_dir)[0] if os.path.isdir(snap_dir) else None
MODEL_PATH = f"{snap_dir}/{snap}" if snap else None

SYSTEM = (
    "You are an expert in fine-grained multimodal emotion recognition. "
    "Given a short video clip showing a listener's reaction and their Chinese subtitle, "
    "predict the listener's emotions as 3-5 comma-separated English emotion words. "
    "Use specific, standard emotion-wheel vocabulary. No numbers, no explanation. "
    "Output ONLY the emotion words separated by commas."
)

def sample_frames(video_path, n=6):
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
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def clean_output(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    for tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<eos>"]:
        if tok in text:
            text = text.split(tok)[0].strip()
    lines = text.strip().split("\n")
    first_line = lines[0] if lines else text
    words = [re.sub(r'[^a-z\-]', '', w.strip().lower())
             for w in first_line.split(",")]
    words = [w for w in words if 2 < len(w) < 25]
    return ",".join(words[:5]) if words else "neutral"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",      type=int, default=0)
    p.add_argument("--suffix",   default="aria")
    p.add_argument("--n_frames", type=int, default=6)
    args = p.parse_args()

    out_path = f"{OUT_DIR}/predictions_{args.suffix}.jsonl"
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try: done.add(json.loads(line)["name"])
                except: pass

    df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    names = df_cand["name"].tolist()
    sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    todo = [n for n in names if n not in done]
    print(f"{args.suffix}: {len(done)} done, {len(todo)} remaining", flush=True)
    if not todo:
        print("All done!"); return

    from transformers import AriaForConditionalGeneration, AriaProcessor
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Loading Aria on GPU {args.gpu}...", flush=True)
    processor = AriaProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AriaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    print("Loaded.", flush=True)

    errors = 0
    with open(out_path, "a") as fout:
        for name in tqdm(todo, desc=args.suffix):
            video_path = f"{VIDEO_ROOT}/{name}.mp4"
            frames = sample_frames(video_path, n=args.n_frames)
            subtitle = str(sub.loc[name, "chinese"]) if name in sub.index else ""
            if subtitle == "nan": subtitle = ""

            try:
                if frames:
                    content = [{"type": "image", "image": img} for img in frames]
                    content.append({
                        "type": "text",
                        "text": f'{SYSTEM}\n\nChinese subtitle: "{subtitle[:300]}"\nEmotion words:'
                    })
                else:
                    content = [{
                        "type": "text",
                        "text": f'{SYSTEM}\n\nChinese subtitle: "{subtitle[:300]}"\nEmotion words:'
                    }]

                messages = [{"role": "user", "content": content}]
                text_in = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)

                inputs = processor(
                    text=[text_in],
                    images=frames if frames else None,
                    return_tensors="pt",
                )

                inputs = {
                    k: (v.to(device=device, dtype=torch.bfloat16)
                        if isinstance(v, torch.Tensor) and v.dtype == torch.float32
                        else v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in inputs.items()
                }

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
                if errors <= 3:
                    import traceback; traceback.print_exc()
                print(f"Error {name}: {e}", flush=True)

            fout.write(json.dumps({"name": name, "prediction": pred}) + "\n")
            fout.flush()

    print(f"Done. Errors: {errors}/{len(todo)}", flush=True)

if __name__ == "__main__":
    main()
