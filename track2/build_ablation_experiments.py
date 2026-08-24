import json, os, zipfile
import pandas as pd
from collections import Counter, defaultdict

import build_ensemble_v16 as base

DATA_DIR = base.DATA_DIR
RESULTS  = base.RESULTS
SUB_DIR  = "/data/multimer/ACMMM2026/submissions/ablation_20260718"
os.makedirs(SUB_DIR, exist_ok=True)

FULL_MODEL_CONFIGS = base.MODEL_CONFIGS

VOTERS_17 = FULL_MODEL_CONFIGS[:17]

def build(out_suffix, model_configs, norm_map, min_score_ratio=0.29,
          rank_decay=0.7, tv_boost_factor=1.2, min_words=3, max_words=6):
    def normalize(word):
        w = word.strip().lower()
        if ' ' in w:
            return None
        if w in norm_map:
            return norm_map[w]
        import re
        w_clean = re.sub(r'[^a-z_-]', '', w)
        if not w_clean:
            return None
        if w_clean in norm_map:
            return norm_map[w_clean]
        w_under = w_clean.replace('-', '_')
        if w_under in norm_map:
            return norm_map[w_under]
        w_nohyph = w_clean.replace('-', '')
        if w_nohyph in norm_map:
            return norm_map[w_nohyph]
        result = w_under
        if len(result) < 2 or len(result) > 20:
            return None
        return result

    def parse_emotions(text):
        text = str(text).strip().lower()
        for ep in base.ERROR_PHRASES:
            if ep in text:
                return []
        words = [normalize(w) for w in text.split(',')]
        return [w for w in words if w]

    df_cand = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")
    all_names = df_cand["name"].tolist()

    loaded = []
    for suffix, weight in model_configs:
        preds = base.load_preds(suffix)
        if preds:
            loaded.append((suffix, weight, preds))
        else:
            print(f"  SKIP {suffix} (file missing/incomplete)")

    print(f"\n[{out_suffix}] Ensembling {len(loaded)}/{len(model_configs)} models "
          f"(ratio={min_score_ratio}, rank_decay={rank_decay}, tv_boost={tv_boost_factor}, "
          f"norm_map_size={len(norm_map)})")

    results = []
    for name in all_names:
        word_scores = defaultdict(float)
        for suffix, weight, preds in loaded:
            if name not in preds:
                continue
            emotions = parse_emotions(preds[name])
            for rank, word in enumerate(emotions[:8]):
                pos_weight = 1.0 / (rank * rank_decay + 1)
                tv_boost = tv_boost_factor if base.TRAIN_VOCAB.get(word, 0) > 5 else 1.0
                word_scores[word] += weight * pos_weight * tv_boost

        if word_scores:
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            best_score = sorted_words[0][1]
            threshold = best_score * min_score_ratio
            top_words = []
            for word, score in sorted_words:
                if len(top_words) < min_words:
                    top_words.append(word)
                elif score >= threshold and len(top_words) < max_words:
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

    avg_words = df_out["openset"].apply(lambda x: len(str(x).split(","))).mean()
    all_words = Counter()
    for row in df_out["openset"]:
        for w in str(row).split(","):
            all_words[w.strip()] += 1
    print(f"Saved: {out_zip}")
    print(f"Avg words/sample: {avg_words:.3f}  |  Top 8: {all_words.most_common(8)}")
    return out_zip

if __name__ == "__main__":
    print("=" * 70)
    print("(a) THRESHOLD-ONLY ablation (26 voters, full 211-rule NORM_MAP, "
          "rank_decay=0.7 fixed) -- vary tau only")
    print("=" * 70)
    build("ablation_thresh_028", FULL_MODEL_CONFIGS, base.NORM_MAP, min_score_ratio=0.28)
    build("ablation_thresh_030", FULL_MODEL_CONFIGS, base.NORM_MAP, min_score_ratio=0.30)

    print("=" * 70)
    print("(b) NORM_MAP-ONLY ablation (26 voters, tau=0.29 fixed) -- vary "
          "NORM_MAP only")
    print("=" * 70)
    build("ablation_normmap_none", FULL_MODEL_CONFIGS, {}, min_score_ratio=0.29)
    build("ablation_normmap_neutralonly", FULL_MODEL_CONFIGS,
          {"neutral": "calm"}, min_score_ratio=0.29)

    print("=" * 70)
    print("(c) VOTER-COUNT-ONLY ablation (full 211-rule NORM_MAP, tau=0.29 "
          "fixed) -- vary voter set only (17 vs. 26; 26-voter point is the "
          "already-submitted 79.3928 anchor, not rebuilt here)")
    print("=" * 70)
    build("ablation_voters17_fullnormmap", VOTERS_17, base.NORM_MAP, min_score_ratio=0.29)

    print("=" * 70)
    print("(d) WEIGHT-ONLY ablation (26 voters, full 211-rule NORM_MAP, "
          "tau=0.29 fixed) -- vary voter weights only (uniform vs. tuned; "
          "tuned-weight point is the already-submitted 79.3928 anchor)")
    print("=" * 70)
    uniform_configs = [(suffix, 2.0) for suffix, _ in FULL_MODEL_CONFIGS]
    build("ablation_uniformweights", uniform_configs, base.NORM_MAP, min_score_ratio=0.29)

    print("\nAll ablation zips written to:", SUB_DIR)
