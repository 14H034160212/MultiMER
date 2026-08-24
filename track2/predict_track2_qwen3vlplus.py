import os, json, base64, time, re
import pandas as pd
from openai import OpenAI
import cv2, numpy as np
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor

API_KEY = os.environ["DASHSCOPE_API_KEY"]
BASE_URL = os.environ["DASHSCOPE_BASE_URL"]
DATA_DIR = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VIDEO_ROOT = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/video/track1_track2_candidate/video"
OUT_DIR = "/data/multimer/ACMMM2026/results/track2"

MODEL_ID = "qwen3-vl-plus"
SUFFIX = "qwen3vlplus"
WORKERS = 6

PROMPT = (
    'Chinese subtitle: "{subtitle}"\n'
    "Look at the video frames and subtitle. Identify the listener's emotional state. "
    "Output ONLY 3-5 comma-separated English emotion words such as: "
    "calm, worried, anxious, sad, angry, surprised, frustrated, confident, serious, "
    "confused, disappointed, hopeful, tense, nervous, concerned, determined, stressed, "
    "contemplative, resigned, attentive, thoughtful, or similar precise emotion words. "
    "No explanation. No sentences. Just comma-separated words."
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_frames_b64(video_path, n=4):
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
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((336, 189))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        frames.append(base64.b64encode(buf.getvalue()).decode())
    cap.release()
    return frames

def clean_pred(text):
    if not text: return "calm"
    if "</think>" in text: text = text.split("</think>")[-1]
    text = text.strip().split("\n")[0]
    words = []
    for w in re.split(r'[,;\s]+', text):
        w = re.sub(r'[^a-z\-]', '', w.lower().strip())
        if 2 < len(w) < 25:
            words.append(w)
    return ",".join(words[:5]) if words else "calm"

def infer_one(name, subtitle, frames):
    content = []
    for b64 in frames:
        content.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
    content.append({"type":"text","text":PROMPT.format(subtitle=subtitle[:250])})
    if not frames:
        content = PROMPT.format(subtitle=subtitle[:250])
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role":"user","content":content}],
                max_tokens=80, temperature=0,
                extra_body={"enable_thinking": False}
            )
            raw = (resp.choices[0].message.content or "").strip()
            return clean_pred(raw)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(3 * (attempt + 1))
            elif attempt == 2:
                return "calm"
            else:
                time.sleep(1)
    return "calm"

def main():
    out_path = f"{OUT_DIR}/predictions_{SUFFIX}.jsonl"
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try: done.add(json.loads(line)["name"])
                except: pass

    df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    df_sub  = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    todo    = [n for n in df_cand["name"].tolist() if n not in done]
    print(f"{SUFFIX}: {len(done)} done, {len(todo)} remaining", flush=True)
    if not todo: print("All done!"); return

    fout = open(out_path, "a")
    done_count = len(done)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        idx = 0
        while idx < len(todo) or futures:
            while len(futures) < WORKERS * 3 and idx < len(todo):
                name = todo[idx]; idx += 1
                subtitle = str(df_sub.loc[name, "chinese"]) if name in df_sub.index else ""
                if subtitle == "nan": subtitle = ""
                frames = extract_frames_b64(f"{VIDEO_ROOT}/{name}.mp4", n=4)
                fut = pool.submit(infer_one, name, subtitle, frames)
                futures[fut] = name
            completed = [f for f in futures if f.done()]
            for fut in completed:
                name = futures.pop(fut)
                try: pred = fut.result()
                except: pred = "calm"
                fout.write(json.dumps({"name": name, "prediction": pred}) + "\n")
                done_count += 1
                if done_count % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed
                    eta = (len(todo)+len(done)-done_count) / rate / 3600
                    print(f"  {done_count}/{len(todo)+len(done)}  {rate:.2f}/s  ETA={eta:.2f}h", flush=True)
            fout.flush()
            if not completed: time.sleep(0.05)

    fout.close()
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
