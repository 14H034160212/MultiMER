import os
import base64, cv2, numpy as np, os, json, time, re, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm import tqdm
from collections import Counter

DATA_DIR   = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VIDEO_DIR  = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/video/track1_track2_candidate/video"
OUT_DIR    = "/data/multimer/ACMMM2026/results/track2"

ALIBABA_KEY  = os.environ["DASHSCOPE_API_KEY"]
ALIBABA_BASE = os.environ["DASHSCOPE_BASE_URL"]
MODEL = "qwen3.5-omni-flash"

SYSTEM = (
    "You are an expert in fine-grained multimodal emotion recognition. "
    "Given video frames of a listener's reaction and their Chinese subtitle, "
    "predict the listener's emotions as exactly 4 comma-separated English emotion words. "
    "Use specific emotion-wheel vocabulary. Output ONLY the 4 emotion words. No explanation."
)

ERROR_PHRASES = {"error", "sorry", "i cannot", "i can't", "unable to", "apologies"}

client = OpenAI(api_key=ALIBABA_KEY, base_url=ALIBABA_BASE)

def sample_frames(video_path, n=4):
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img).resize((224, 224))
        buf = BytesIO()
        pil.save(buf, format='JPEG', quality=75)
        frames.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    cap.release()
    return frames

def clean_output(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    words = [re.sub(r'[^a-z\-_]', '', w.strip().lower()) for w in text.split(",")]
    words = [w for w in words if 2 < len(w) < 25]
    return ",".join(words[:5]) if words else "neutral"

def predict_one(name, subtitle, frames, max_retries=3):
    content = []
    for f in frames:
        content.append({"type": "image_url", "image_url": {"url": f}})
    content.append({"type": "text", "text": f'Chinese subtitle: "{subtitle[:200]}"\nEmotion words:'})

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content}
                ],
                max_tokens=60,
                stream=False
            )
            text = resp.choices[0].message.content or ""
            if any(p in text.lower() for p in ERROR_PHRASES):
                return "neutral"
            return clean_output(text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"error:{type(e).__name__}"
    return "neutral"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n_frames", type=int, default=4)
    p.add_argument("--suffix", default="omniflash")
    args = p.parse_args()

    out_path = f"{OUT_DIR}/predictions_{args.suffix}.jsonl"
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["name"])
                except:
                    pass

    df = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    names = df["name"].tolist()
    todo = [n for n in names if n not in done]
    print(f"{args.suffix}: {len(done)} done, {len(todo)} remaining", flush=True)
    if not todo:
        print("All done!")
        return

    results = {}

    def process(name):
        video_path = f"{VIDEO_DIR}/{name}.mp4"
        subtitle = str(sub.loc[name, "chinese"]) if name in sub.index else ""
        if subtitle == "nan":
            subtitle = ""
        frames = sample_frames(video_path, n=args.n_frames)
        pred = predict_one(name, subtitle, frames)
        return name, pred

    errors = 0
    with open(out_path, "a", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, n): n for n in todo}
            for fut in tqdm(as_completed(futures), total=len(todo), desc=args.suffix):
                name, pred = fut.result()
                if "error:" in pred:
                    errors += 1
                fout.write(json.dumps({"name": name, "prediction": pred}) + "\n")

    print(f"Done. Errors: {errors}/{len(todo)}", flush=True)

    wc = Counter()
    n = 0
    with open(out_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                n += 1
                for w in d["prediction"].split(","):
                    w = w.strip().lower()
                    if w:
                        wc[w] += 1
            except:
                pass
    tot = n or 1
    anx = (wc.get("anxious", 0) + wc.get("anxiety", 0)) / tot
    cur = (wc.get("curious", 0) + wc.get("curiosity", 0)) / tot
    print(f"Bias: anxious={anx*100:.1f}%, curious={cur*100:.1f}%, worried={wc.get('worried',0)/tot*100:.1f}%")
    print(f"Top 10: {wc.most_common(10)}")

if __name__ == "__main__":
    main()
