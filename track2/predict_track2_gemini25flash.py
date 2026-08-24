import os, sys, json, re, time, base64, threading
import urllib.request
import cv2, numpy as np, pandas as pd
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DATA_DIR   = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VIDEO_ROOT = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/video/track1_track2_candidate/video"
OUT_DIR    = "/data/multimer/ACMMM2026/results/track2"
OUT_PATH   = f"{OUT_DIR}/predictions_gemini25flash.jsonl"

API_KEY = os.environ["GOOGLE_API_KEY"]
MODEL   = "models/gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent"

WORKERS = 8
RETRY   = 3

PROMPT_TPLT = (
    'Chinese subtitle: "{subtitle}"\n'
    "Based on the video frames and the subtitle, identify the LISTENER's emotional state. "
    "Output ONLY 3-5 comma-separated English emotion words.\n"
    "Examples: calm,worried,sad  |  anxious,tense,fearful  |  serious,concerned,determined\n"
    "No explanation, no numbers."
)

write_lock = threading.Lock()

def load_frames_b64(video_path, n=4):
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
    results = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok and fr is not None:
            pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=60)
            results.append(base64.b64encode(buf.getvalue()).decode())
    cap.release()
    return results

def clean_output(text):
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    first = text.split("\n")[0].strip()

    first = re.sub(r'\*+', '', first)
    words = [re.sub(r"[^a-z\-]", "", w.strip().lower()) for w in re.split(r"[,;|/\s]+", first)]
    words = [w for w in words if 2 < len(w) < 25]
    return ",".join(words[:5]) if words else "calm"

def infer_one(name, subtitle, frames_b64):
    prompt = PROMPT_TPLT.format(subtitle=subtitle[:300])
    parts = [{"inlineData": {"mimeType": "image/jpeg", "data": b64}} for b64 in frames_b64]
    parts.append({"text": prompt})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.0},
    }
    data_bytes = json.dumps(payload).encode()

    for attempt in range(RETRY):
        try:
            req = urllib.request.Request(
                API_URL, data=data_bytes,
                headers={"Content-Type": "application/json", "x-goog-api-key": API_KEY}
            )
            resp = urllib.request.urlopen(req, timeout=90)
            data = json.loads(resp.read())
            cand = data["candidates"][0]
            parts_out = cand.get("content", {}).get("parts", [])
            text = next((p["text"] for p in parts_out if "text" in p), "")
            if text.strip():
                return name, clean_output(text)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            if "RATE" in body or e.code == 429:
                time.sleep(min(30, 5 * (2 ** attempt)))
            elif attempt >= RETRY - 1:
                print(f"HTTPError {name}: {e.code} {body}", flush=True)
        except Exception as e:
            if attempt >= RETRY - 1:
                print(f"Error {name}: {e}", flush=True)
            time.sleep(2)
    return name, "calm"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                try: done.add(json.loads(line)["name"])
                except: pass

    df    = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    sub_df = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    names  = df["name"].tolist()
    todo   = [n for n in names if n not in done]
    print(f"gemini25flash: {len(done)} done, {len(todo)} remaining", flush=True)
    if not todo:
        print("All done!"); return

    def get_sub(name):
        v = str(sub_df.loc[name, "chinese"]) if name in sub_df.index else ""
        return "" if v == "nan" else v

    fout = open(OUT_PATH, "a")

    print("Bias test (30 samples)...", flush=True)
    bias_names = todo[:30]
    bias_results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(infer_one, n, get_sub(n), load_frames_b64(f"{VIDEO_ROOT}/{n}.mp4")): n
                for n in bias_names}
        for f in as_completed(futs):
            bias_results.append(f.result())

    words_flat = [w for _, p in bias_results for w in p.split(",")]
    total_w = len(words_flat) or 1
    anxious = words_flat.count("anxious") / total_w
    curious = words_flat.count("curious") / total_w
    calm    = words_flat.count("calm") / total_w
    print(f"Bias: anxious={anxious*100:.1f}% curious={curious*100:.1f}% calm={calm*100:.1f}%", flush=True)
    if anxious > 0.15 or curious > 0.15:
        print("BIAS FAIL!", flush=True)
        for n, p in bias_results:
            fout.write(json.dumps({"name": n, "prediction": p}) + "\n")
        fout.flush()
        fout.close()
        return

    print("BIAS PASS - launching full run...", flush=True)
    for n, p in bias_results:
        fout.write(json.dumps({"name": n, "prediction": p}) + "\n")
        done.add(n)
    fout.flush()

    remaining = [n for n in todo if n not in done]
    print(f"{len(remaining)} remaining", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(infer_one, n, get_sub(n), load_frames_b64(f"{VIDEO_ROOT}/{n}.mp4")): n
            for n in remaining
        }
        pbar = tqdm(total=len(remaining), desc="gemini25flash")
        for f in as_completed(futs):
            name, pred = f.result()
            with write_lock:
                fout.write(json.dumps({"name": name, "prediction": pred}) + "\n")
                fout.flush()
            pbar.update(1)
        pbar.close()

    fout.close()
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
