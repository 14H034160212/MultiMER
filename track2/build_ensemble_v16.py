import json, os, re, zipfile
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

DATA_DIR   = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
RESULTS    = "/data/multimer/ACMMM2026/results/track2"
SUB_DIR    = "/data/multimer/ACMMM2026/submissions/round2_20260708v16"
os.makedirs(SUB_DIR, exist_ok=True)

NORM_MAP = {

    "anger": "angry", "furious": "angry", "infuriated": "angry",
    "irate": "angry", "outraged": "angry", "enraged": "angry",
    "irritated": "irritated", "annoyed": "annoyed", "frustrated": "frustrated",
    "indignant": "angry", "hatred": "angry", "hate": "angry",
    "vexed": "annoyed",

    "sadness": "sad", "sorrow": "sad", "sorrowful": "sad",
    "grief": "grief", "grieving": "grief", "grief-stricken": "grief",
    "grief_stricken": "grief",
    "devastated": "devastated", "miserable": "miserable",
    "depressed": "depressed", "melancholic": "sad", "melancholy": "sad",
    "mournful": "sad", "longing": "nostalgic",
    "nostalgia": "sad", "nostalgic": "sad",

    "anxious": "anxious", "anxiety": "anxious",
    "fearful": "fearful", "fear": "fearful", "afraid": "fearful",
    "terrified": "terrified", "terror": "terrified",
    "nervous": "nervous", "uneasy": "uneasy", "apprehensive": "worried",
    "distressed": "distressed", "distress": "distressed",
    "panicked": "panicked", "panic": "panicked",
    "apprehension": "worried", "urgency": "worried", "worry": "worried",

    "surprised": "surprised", "astonished": "astonished",
    "astonishment": "astonished", "shocked": "shocked", "shock": "shocked",
    "stunned": "stunned", "startled": "surprised", "aghast": "shocked",

    "disgust": "disgusted", "disgusting": "disgusted",
    "repulsed": "repulsed", "repulsion": "repulsed",

    "dislike": "disgusted",

    "happy": "happy", "happiness": "happy", "joyful": "joyful",
    "joy": "joyful", "elated": "elated", "ecstatic": "ecstatic",
    "delighted": "delighted", "excited": "excited", "excitement": "excited",
    "thrilled": "thrilled", "pleased": "pleased",
    "cheerful": "cheerful", "gleeful": "gleeful", "amused": "pleased",
    "amusement": "pleased",

    "calm": "calm", "peaceful": "peaceful", "serene": "serene",
    "neutral": "calm",
    "stoic": "calm",

    "guilty": "guilty", "guilt": "guilty", "remorseful": "regret",
    "remorse": "regret", "apologetic": "apologetic",
    "ashamed": "ashamed", "shame": "ashamed", "embarrassed": "embarrassed",

    "loving": "loving", "love": "loving", "affectionate": "affectionate",
    "tender": "tender", "caring": "caring",
    "empathetic": "sympathetic", "compassionate": "compassionate",

    "confused": "confused", "confusion": "confused",
    "bewildered": "confused", "perplexed": "confused", "puzzled": "confused",

    "jealous": "jealous", "jealousy": "jealous", "envious": "envious",
    "envy": "envious",

    "lonely": "lonely", "loneliness": "lonely", "isolated": "isolated",

    "hopeful": "hope", "hope": "hope", "optimistic": "optimistic",
    "anticipatory": "anticipatory",

    "proud": "proud", "pride": "proud",

    "determined": "confident", "determination": "confident",
    "resolve": "confident", "focused": "confident",

    "regret": "regret", "regretful": "regret",
    "bitter": "bitter", "bitterness": "bitter",

    "suspicious": "suspicious", "suspicion": "suspicious",

    "conflicted": "conflicted", "torn": "conflicted",

    "serious": "serious", "solemn": "serious", "grave": "serious",
    "seriousness": "serious",
    "somber": "serious", "solemnity": "serious",

    "curiosity": "curious",

    "cautious": "cautious",

    "dissatisfied": "dissatisfied", "displeased": "dissatisfied",
    "dissatisfaction": "dissatisfied",

    "pensive": "contemplative", "thoughtful": "thoughtful",
    "reflective": "contemplative", "contemplative": "contemplative",
    "contemplation": "contemplative", "contemplate": "contemplative",

    "grateful": "grateful", "gratitude": "grateful",

    "disappointed": "disappointed", "disappointment": "disappointed",
    "dismayed": "disappointed", "disheartened": "disappointed",
    "disapproval": "disappointed",

    "helpless": "helpless", "hopeless": "helpless",
    "desperate": "desperate", "despair": "desperate",
    "overwhelmed": "helpless", "resigned": "helpless",
    "distraught": "helpless", "wretched": "helpless",

    "relieved": "relieved", "relief": "relieved",
    "warm": "warm", "warmth": "warm",

    "skeptical": "doubtful", "skepticism": "doubtful",

    "content": "satisfied", "respectful": "respect",
    "uncomfortable": "uneasy", "tense": "nervous",
    "anguished": "painful",
    "agitated": "nervous", "flustered": "nervous",
    "alarmed": "alarmed", "unsettled": "uncertain",
    "exasperated": "frustrated",
    "perturbed": "worried",
    "repentant": "regret",
    "concern": "concerned",
    "tension": "nervous",
    "resignation": "helpless",
    "confidence": "confident",
    "unease": "uneasy",
    "reluctance": "hesitant",
    "burdened": "stressed", "stressed": "stressed",
    "blame": "resentful",
    "responsibility": "responsible", "responsible": "responsible",
    "uncertain": "uncertain", "uncertainty": "uncertain",
    "discontented": "discontented",
    "touched": "moved", "moved": "emotional",

    "indifferent": "calm",
    "sincere": "calm",
    "humble": "calm",
    "urgent": "worried",
    "pleading": "worried",
    "authoritative": "confident",
    "polite": "pleased",
    "resolute": "determined",

    "defiant": "angry",

    "firm": "serious",
    "friendly": "pleased",
    "stern": "serious",

    "relaxed": "calm",
    "composed": "calm",

    "frustration": "frustrated",
    "calmness": "calm",
    "surprise": "surprised",
    "concern": "concerned",
    "contemplation": "contemplative",

    "pensive": "contemplative",
    "wistful": "melancholic",

    "informative": "attentive",
    "unemotional": "calm",

    "attentive": "calm",

}

def load_train_vocab():
    try:
        df = pd.read_csv(f"{DATA_DIR}/track2_train_human.csv")
        vocab = Counter()
        for row in df['openset']:
            for w in str(row).split(','):
                w = w.strip().lower()
                if w:
                    w = normalize_raw(w)
                    if w:
                        vocab[w] += 1
        return vocab
    except:
        return Counter()

def normalize_raw(word):
    w = word.strip().lower()
    if ' ' in w:
        return None
    w = re.sub(r'[^a-z_-]', '', w)
    if len(w) < 2 or len(w) > 20:
        return None
    return w

def normalize(word):
    w = word.strip().lower()
    if ' ' in w:
        return None
    if w in NORM_MAP:
        return NORM_MAP[w]
    w_clean = re.sub(r'[^a-z_-]', '', w)
    if not w_clean:
        return None
    if w_clean in NORM_MAP:
        return NORM_MAP[w_clean]
    w_under = w_clean.replace('-', '_')
    if w_under in NORM_MAP:
        return NORM_MAP[w_under]
    w_nohyph = w_clean.replace('-', '')
    if w_nohyph in NORM_MAP:
        return NORM_MAP[w_nohyph]
    result = w_under
    if len(result) < 2 or len(result) > 20:
        return None
    return result

TRAIN_VOCAB = load_train_vocab()
print(f"Training vocab: {len(TRAIN_VOCAB)} unique words")

MODEL_CONFIGS = [

    ("gpt_audio",              3.5),
    ("gemma4_26b",             3.0),
    ("qwen3omni",              3.0),
    ("qwen35omni",             3.0),
    ("sonnet5_vision",         2.5),
    ("gpt55_video",            2.5),
    ("qwen3vl32b",             2.5),

    ("gemma4audio",            2.5),
    ("qwen3omni_lora_human",         2.5),
    ("qwen3omni_lora_full",          2.5),
    ("qwen3omni_lora_human_audio",   3.0),
    ("qwen3omni_lora_full_audio",    3.0),
    ("qwen35_9b",              2.0),
    ("qwen35_9b_video_t2",     2.0),
    ("gemma4_31b",             2.5),
    ("qwen35lora",             2.0),

    ("gemini25flash_v2",       2.0),

    ("qwen3omni_lora_full_ep3",   3.0),
    ("qwen3omni_lora_full_ep2",   2.5),

    ("qwen3omni_lora_full_ep2_fp16_worried_only", 1.5),

    ("aria_worried_only",     1.5),
    ("omniflash",     0.5),
    ("omniplus",     0.5),

    ("qwen3omni_lora_full_ep4_worried_only", 0.75),

    ("qwen3omni_lora_full_ep5_worried_only", 1.0),

    ("qwen3vlplus",        1.5),

]

ERROR_PHRASES = {"credit balance is too low", "error", "api error", "rate limit",
                 "i cannot", "i can't", "sorry", "unable to", "apologies"}

def parse_emotions(text):
    text = str(text).strip().lower()
    for ep in ERROR_PHRASES:
        if ep in text:
            return []
    words = [normalize(w) for w in text.split(',')]
    return [w for w in words if w]

def load_preds(suffix, min_count=19000):
    path = f"{RESULTS}/predictions_{suffix}.jsonl"
    if not os.path.exists(path):
        return None
    done = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                pred = r.get("prediction") or r.get("openset") or ""
                done[r["name"]] = pred
            except:
                pass
    return done if len(done) >= min_count else None

def build_ensemble(out_suffix, min_score_ratio=0.28, max_words=6, min_words=3, calm_ratio=None, worried_ratio=None, rank_decay=0.5, tv_boost_factor=1.2):
    df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    all_names = df_cand["name"].tolist()

    loaded = []
    for suffix, weight in MODEL_CONFIGS:
        preds = load_preds(suffix)
        if preds:
            loaded.append((suffix, weight, preds))
            print(f"  Loaded {suffix}: {len(preds)} preds (w={weight})")
        else:
            print(f"  SKIP {suffix}")

    if calm_ratio is None:
        calm_ratio = min_score_ratio
    if worried_ratio is None:
        worried_ratio = min_score_ratio
    print(f"\nEnsembling {len(loaded)} models (ratio={min_score_ratio}, calm_ratio={calm_ratio}, worried_ratio={worried_ratio})")

    results = []
    for name in all_names:
        word_scores = defaultdict(float)
        for suffix, weight, preds in loaded:
            if name not in preds:
                continue
            emotions = parse_emotions(preds[name])
            for rank, word in enumerate(emotions[:8]):
                pos_weight = 1.0 / (rank * rank_decay + 1)
                tv_boost = tv_boost_factor if TRAIN_VOCAB.get(word, 0) > 5 else 1.0
                word_scores[word] += weight * pos_weight * tv_boost

        if word_scores:
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            best_score = sorted_words[0][1]
            threshold = best_score * min_score_ratio
            calm_threshold = best_score * calm_ratio
            worried_threshold = best_score * worried_ratio
            top_words = []
            for word, score in sorted_words:
                if word == "calm":
                    effective_threshold = calm_threshold
                elif word in ("worried", "worry"):
                    effective_threshold = worried_threshold
                else:
                    effective_threshold = threshold
                if len(top_words) < min_words:
                    top_words.append(word)
                elif score >= effective_threshold and len(top_words) < max_words:
                    top_words.append(word)
                else:
                    break
        else:
            top_words = ["neutral"]

        results.append({"name": name, "openset": ",".join(top_words)})

    df_out = pd.DataFrame(results)
    out_csv = f"{RESULTS}/answer_{out_suffix}.csv"
    df_out[["name", "openset"]].to_csv(out_csv, index=False)

    out_zip = f"{SUB_DIR}/track2_{out_suffix}.zip"
    with zipfile.ZipFile(out_zip, "w") as z:
        z.write(out_csv, "answer.csv")
    print(f"Saved: {out_zip}")

    avg_words = df_out["openset"].apply(lambda x: len(str(x).split(","))).mean()
    all_words = Counter()
    for row in df_out["openset"]:
        for w in str(row).split(","):
            all_words[w.strip()] += 1
    print(f"Avg words/sample: {avg_words:.2f}")
    print(f"Top 15 words: {all_words.most_common(15)}")
    return out_zip

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="ensemble_v16")
    p.add_argument("--ratio", type=float, default=0.28)
    p.add_argument("--calm_ratio", type=float, default=None)
    p.add_argument("--worried_ratio", type=float, default=None)
    p.add_argument("--min_words", type=int, default=3)
    p.add_argument("--max_words", type=int, default=6)
    p.add_argument("--rank_decay", type=float, default=0.5, help="Rank decay factor: pos_weight = 1/(rank*decay+1). Default=0.5.")
    p.add_argument("--tv_boost", type=float, default=1.2, help="TV boost factor for train-vocab words. Default=1.2, set 1.0 to disable.")
    args = p.parse_args()
    build_ensemble(args.suffix, min_score_ratio=args.ratio, calm_ratio=args.calm_ratio,
                   worried_ratio=args.worried_ratio, min_words=args.min_words, max_words=args.max_words,
                   rank_decay=args.rank_decay, tv_boost_factor=args.tv_boost)
