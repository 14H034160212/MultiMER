import os
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DATA_DIR = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VIDEO_ROOT = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/video/track1_track2_candidate/video"
OUTPUT_DIR = "/data/multimer/ACMMM2026/results/track2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEM_MSG = "You are an emotion recognition expert. Return only comma-separated emotion words."

PROMPT = """You are predicting fine-grained emotions from short video frames and subtitle.

Subtitle (Chinese): {subtitle}

Output ONLY 3-6 lowercase emotion words, comma-separated, no explanation."""

def sample_frames(video_path, n_frames=8):
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int).tolist()
    outs = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        scale = min(640 / max(h, w), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok2:
            continue
        outs.append(buf.tobytes())
    cap.release()
    return outs

def predict_one(name, subtitle, model, client):
    video_path = os.path.join(VIDEO_ROOT, f"{name}.mp4")
    frames = sample_frames(video_path, n_frames=8)
    if not frames:
        return name, None, "missing_video"

    content = [{"type": "text", "text": PROMPT.format(subtitle=subtitle)}]
    for f in frames:
        b64 = base64.b64encode(f).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": content},
            ],
            max_completion_tokens=150,
            timeout=60,
        )
        text = resp.choices[0].message.content.strip().lower()
        emotions = [e.strip() for e in text.replace("\n", ",").split(",") if e.strip()]
        emotions = [e for e in emotions if e and len(e) < 30][:6]
        return name, ",".join(emotions) if emotions else "neutral", "ok"
    except Exception as e:
        return name, "neutral", f"error:{str(e)[:120]}"

def run(max_workers=8, resume=True, model="gpt-5.5", out_suffix="gpt55_video", limit=None):
    client = OpenAI()

    df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    df_sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    if limit:
        df_cand = df_cand.head(limit)

    raw_path = f"{OUTPUT_DIR}/predictions_{out_suffix}.jsonl"
    done = {}
    if resume and os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["name"]] = r["prediction"]
                except Exception:
                    pass
        print(f"Resuming: {len(done)} done")

    todo = df_cand[~df_cand["name"].isin(done)]
    print(f"Predicting {len(todo)} samples with model={model}, workers={max_workers}...")

    missing = 0
    with open(raw_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {}
            for _, row in todo.iterrows():
                name = row["name"]
                subtitle = str(df_sub.loc[name, "chinese"]) if name in df_sub.index else ""
                if subtitle == "nan":
                    subtitle = ""
                fut = exe.submit(predict_one, name, subtitle, model, client)
                futures[fut] = name

            for fut in tqdm(as_completed(futures), total=len(futures)):
                name, pred, status = fut.result()
                if pred is None:
                    missing += 1
                    continue
                done[name] = pred
                fout.write(json.dumps({"name": name, "prediction": pred, "status": status}, ensure_ascii=False) + "\n")
                fout.flush()

    df_cand["prediction"] = df_cand["name"].map(done).fillna("neutral")
    out_csv = f"{OUTPUT_DIR}/answer_{out_suffix}.csv"
    df_cand[["name", "prediction"]].to_csv(out_csv, index=False)

    print(f"Saved {len(done)}/{len(df_cand)} predictions -> {out_csv}")
    if missing > 0:
        print(f"Missing video for {missing} samples this run.")

    import zipfile

    with zipfile.ZipFile(out_csv.replace(".csv", ".zip"), "w") as z:
        z.write(out_csv, "answer.csv")
    print(f"Submission: {out_csv.replace('.csv', '.zip')}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", type=str, default="gpt-5.5")
    p.add_argument("--out_suffix", type=str, default="gpt55_video")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="only run on first N candidates (pilot)")
    args = p.parse_args()
    run(args.workers, not args.no_resume, args.model, args.out_suffix, args.limit)
