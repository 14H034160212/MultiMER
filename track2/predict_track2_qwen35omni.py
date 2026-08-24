import os, json, base64, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR  = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
AUDIO_DIR = "/data/multimer/ACMMM2026/dataset/mer2026-dataset-process/audio"
OUT_DIR   = "/data/multimer/ACMMM2026/results/track2"
VAL_DIR   = "/data/multimer/ACMMM2026/track2/pseudo_val"

ALIBABA_KEY = os.environ["DASHSCOPE_API_KEY"]
API_URL     = os.environ["DASHSCOPE_API_URL"]
MODEL       = "qwen3.5-omni-plus"

HEADERS = {
    "Authorization": f"Bearer {ALIBABA_KEY}",
    "Content-Type": "application/json",
}

def predict_one(name, subtitle, audio_b64):
    prompt = (
        f'Chinese subtitle: "{subtitle[:200]}"\n\n'
        "List exactly 4 emotion words comma-separated. Example: calm,neutral,composed,serene\n"
        "Your 4 words:"
    )
    payload = {
        "model": MODEL,
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"audio": f"data:audio/wav;base64,{audio_b64}"},
                    {"text": prompt},
                ],
            }]
        },
        "parameters": {"max_tokens": 80},
    }
    for attempt in range(4):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            if resp.status_code == 429 or resp.status_code == 503:
                time.sleep(10 * (attempt + 1))
                continue
            if resp.status_code != 200:
                if attempt < 3:
                    time.sleep(5)
                    continue
                return f"error:http{resp.status_code}"
            data = resp.json()
            text = data["output"]["choices"][0]["message"]["content"][0]["text"]
            text = text.strip().lower().replace("\n", ",").replace(";", ",")
            words = [w.strip() for w in text.split(",") if w.strip() and 2 < len(w.strip()) < 25]
            return ",".join(words[:6]) or "neutral"
        except Exception as e:
            if attempt < 3:
                time.sleep(5)
                continue
            return f"error:{str(e)[:80]}"
    return "error:max_retries"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--suffix",  default="qwen35omni")
    p.add_argument("--holdout", action="store_true")
    args = p.parse_args()

    out_path = (f"{OUT_DIR}/predictions_{args.suffix}.jsonl" if not args.holdout
                else f"{VAL_DIR}/holdout_predictions_{args.suffix}.jsonl")

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

    sub = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")
    todo = [n for n in names if n not in done]
    print(f"{args.suffix}: {len(done)} done, {len(todo)} remaining")
    if not todo:
        print("All done!"); return

    def load_b64(name):
        path = f"{AUDIO_DIR}/{name}.wav"
        if not os.path.exists(path): return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    errors = 0
    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {}
            for name in todo:
                b64 = load_b64(name)
                subtitle = str(sub.loc[name, "chinese"]) if name in sub.index else ""
                if subtitle == "nan": subtitle = ""
                if b64 is None:
                    fout.write(json.dumps({"name": name, "prediction": "neutral", "status": "no_audio"}) + "\n")
                    fout.flush()
                    continue
                fut = exe.submit(predict_one, name, subtitle, b64)
                futures[fut] = name

            for fut in tqdm(as_completed(futures), total=len(futures), desc=args.suffix):
                name = futures[fut]
                pred = fut.result()
                if pred.startswith("error:"):
                    errors += 1
                fout.write(json.dumps({"name": name, "prediction": pred, "status": "ok"}) + "\n")
                fout.flush()

    print(f"Done. Errors: {errors}/{len(todo)}")

if __name__ == "__main__":
    main()
