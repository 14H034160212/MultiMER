import os
import base64, cv2, numpy as np, os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm import tqdm

DATA_DIR   = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VIDEO_DIR  = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/video/track1_track2_candidate/video"
OUT_DIR    = "/data/multimer/ACMMM2026/results/track2"
VAL_DIR    = "/data/multimer/ACMMM2026/track2/pseudo_val"

ALIBABA_KEY = os.environ["DASHSCOPE_API_KEY"]
ALIBABA_BASE = os.environ["DASHSCOPE_BASE_URL"]
MODEL = "qwen3.5-omni-plus"

SYSTEM = (
    "You are an expert in fine-grained multimodal emotion recognition. "
    "Given video frames of a listener's reaction and their Chinese subtitle, "
    "predict the listener's emotions as exactly 4 comma-separated English emotion words. "
    "Use specific emotion-wheel vocabulary. Output ONLY the 4 emotion words. No explanation."
)

ERROR_PHRASES = {"error","sorry","i cannot","i can't","unable to","apologies"}

client = OpenAI(api_key=ALIBABA_KEY, base_url=ALIBABA_BASE)

def sample_frames(video_path, n=8):
    if not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: cap.release(); return []
    idxs = np.linspace(0, max(total-1,0), n).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok: continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img).resize((224,224))
        buf = BytesIO()
        pil.save(buf, format='JPEG', quality=75)
        frames.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    cap.release()
    return frames

def predict_one(name, subtitle):
    video_path = f"{VIDEO_DIR}/{name}.mp4"
    frames = sample_frames(video_path, n=8)
    if not frames:
        return name, "neutral", "no_video"

    content = [{"type":"image_url","image_url":{"url":f}} for f in frames]
    content.append({"type":"text","text":f'Subtitle: "{subtitle}"\nEmotions (4 words):'})

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role":"system","content":SYSTEM},
                    {"role":"user","content":content}
                ],
                max_tokens=60,
                extra_body={"enable_thinking": False}
            )
            text = resp.choices[0].message.content.strip().lower()
            for ep in ERROR_PHRASES:
                if ep in text: return name, f"error:content_{ep[:20]}", "error"
            words = [w.strip() for w in text.replace(";",",").replace("\n",",").split(",")
                     if w.strip() and 2 < len(w.strip()) < 25]
            pred = ",".join(words[:5]) or "neutral"
            return name, pred, "ok"
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429","503","rate","quota","RESOURCE_EXHAUSTED","overloaded"]):
                wait = 15*(attempt+1)
                print(f"  [rate-limit {attempt+1}/4] sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            return name, f"error:{err[:80]}", "error"
    return name, "error:max_retries", "error"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--holdout", action="store_true")
    args = p.parse_args()

    out_path = (f"{OUT_DIR}/predictions_omniplus.jsonl" if not args.holdout
                else f"{VAL_DIR}/holdout_predictions_omniplus.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try: done.add(json.loads(line)["name"])
                except: pass

    if args.holdout:
        split = pd.read_csv(f"{VAL_DIR}/holdout_split.csv")
        names = split[split.is_holdout]["name"].tolist()
    else:
        df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
        names = df_cand["name"].tolist()

    sub_df = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    todo = [(n, str(sub_df.loc[n,"chinese"]) if n in sub_df.index else "") for n in names if n not in done]
    todo = [(n, "" if s=="nan" else s[:200]) for n, s in todo]
    print(f"omniplus: {len(done)} done, {len(todo)} remaining")
    if not todo: print("All done!"); return

    errors = 0
    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(predict_one, n, s): (n,s) for n,s in todo}
            for fut in tqdm(as_completed(futures), total=len(todo), desc="omniplus"):
                name, pred, status = fut.result()
                if status == "error": errors += 1
                fout.write(json.dumps({"name":name,"prediction":pred,"status":status}) + "\n")
                fout.flush()

    print(f"Done. Errors: {errors}/{len(todo)}")

if __name__ == "__main__":
    main()
