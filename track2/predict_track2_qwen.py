import os, json, argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

DATA_DIR   = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
OUTPUT_DIR = "/data/multimer/ACMMM2026/results/track2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEW_SHOT = [
    ("一点都不担心，放心好了，没事的",       "relaxed,calm,reassuring,confident"),
    ("你给我出去！现在！马上给我滚！",         "angry,furious,irritated,hostile"),
    ("哈哈真的假的！太棒了！我早就知道！",     "happy,excited,joyful,thrilled"),
    ("他走了，就这么走了，再也回不来了",       "sad,heartbroken,grieving,sorrowful"),
    ("万一出错了怎么办，我心里很不安",         "anxious,worried,nervous,uneasy"),
    ("这……这怎么可能，完全没想到",            "surprised,shocked,astonished,disbelief"),
    ("谢谢你，真的，你帮了我大忙了",           "grateful,touched,appreciative,warm"),
    ("对不起，是我的错，我真的很后悔",         "guilty,remorseful,apologetic,regretful"),
]

SYSTEM = (
    "You are an expert in fine-grained emotion recognition. "
    "Given a Chinese subtitle, predict the listener's emotions. "
    "Output 3-6 comma-separated English emotion words only. No explanations."
)

def build_user_msg(subtitle: str) -> str:
    shots = "\n".join(f'Subtitle: "{s}"\nEmotions: {e}' for s, e in FEW_SHOT)
    return f"{shots}\nSubtitle: \"{subtitle[:300]}\"\nEmotions:"

def load_model(model_id: str, gpu: int):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_id} on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True
    )
    mdl.eval()
    print("Model loaded.")
    return tok, mdl, device

def predict_batch(tok, mdl, device, subtitles):
    texts = []
    for sub in subtitles:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": build_user_msg(sub)},
        ]
        texts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        ))

    enc = tok(texts, return_tensors="pt", padding=True,
              truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = mdl.generate(
            **enc, max_new_tokens=40, do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    results = []
    for i, seq in enumerate(out):
        new  = seq[enc["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True).strip()

        emotions = ",".join(
            w.strip().lower().replace(" ", "_")
            for w in text.split(",")
            if w.strip() and len(w.strip()) < 25
        )[:200]
        results.append(emotions or "neutral")
    return results

def run(model_id, batch_size, gpu, out_suffix, resume):
    tok, mdl, device = load_model(model_id, gpu)

    df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    df_sub  = pd.read_csv(f"{DATA_DIR}/subtitle_chieng.csv").set_index("name")

    out_path = f"{OUTPUT_DIR}/predictions_{out_suffix}.jsonl"
    done = {}
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try: r=json.loads(line); done[r["name"]]=r["prediction"]
                except: pass
        print(f"Resuming: {len(done)} done")

    todo  = df_cand[~df_cand["name"].isin(done)]
    names = todo["name"].tolist()
    subs  = [(str(df_sub.loc[n,"chinese"]) if n in df_sub.index else "").replace("nan","")
             for n in names]

    print(f"Predicting {len(names)} samples, batch={batch_size} ...")
    with open(out_path, "a") as fout:
        for i in tqdm(range(0, len(names), batch_size)):
            bn, bs = names[i:i+batch_size], subs[i:i+batch_size]
            preds  = predict_batch(tok, mdl, device, bs)
            for name, pred in zip(bn, preds):
                done[name] = pred
                fout.write(json.dumps({"name": name, "prediction": pred}) + "\n")
            fout.flush()

    df_cand["prediction"] = df_cand["name"].map(done).fillna("neutral")
    out_csv = f"{OUTPUT_DIR}/answer_{out_suffix}.csv"
    df_cand[["name","prediction"]].to_csv(out_csv, index=False)

    import zipfile
    with zipfile.ZipFile(out_csv.replace(".csv",".zip"), "w") as z:
        z.write(out_csv, "answer.csv")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="Qwen/Qwen3-8B")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gpu",        type=int, default=5)
    p.add_argument("--out_suffix", default="qwen3_8b")
    p.add_argument("--no_resume",  action="store_true")
    args = p.parse_args()
    run(args.model, args.batch_size, args.gpu, args.out_suffix, not args.no_resume)
