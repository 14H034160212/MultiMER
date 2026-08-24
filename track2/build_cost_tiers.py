import json, os, re, sys
import pandas as pd
from collections import defaultdict, Counter

import build_ensemble_v16 as base

DATA_DIR = base.DATA_DIR
RESULTS = base.RESULTS

FULL = base.MODEL_CONFIGS

LORA_NAMES = {
    "qwen3omni_lora_human", "qwen3omni_lora_full",
    "qwen3omni_lora_human_audio", "qwen3omni_lora_full_audio",
    "qwen3omni_lora_full_ep3", "qwen3omni_lora_full_ep2",
    "qwen3omni_lora_full_ep2_fp16_worried_only",
    "qwen3omni_lora_full_ep4_worried_only",
    "qwen3omni_lora_full_ep5_worried_only",
    "qwen35lora",
}
LOCAL_NAMES = {
    "gemma4_26b", "qwen3omni", "qwen3vl32b", "gemma4audio", "gemma4_31b",
    "qwen35_9b_video_t2", "qwen35_9b", "aria_worried_only",
}
API_NAMES = {
    "gpt_audio", "qwen35omni", "sonnet5_vision", "gpt55_video",
    "gemini25flash_v2", "qwen3vlplus", "omniflash", "omniplus",
}

def subset(names):
    return [(s, w) for s, w in FULL if s in names]

TIERS = {
    "full26":    FULL,
    "apifree18": subset(LORA_NAMES | LOCAL_NAMES),
    "local8":    subset(LOCAL_NAMES),
    "lora10":    subset(LORA_NAMES),
    "api8":      subset(API_NAMES),
}

def predict(model_configs, norm_map, all_names,
            min_score_ratio=0.29, rank_decay=0.7, tv_boost_factor=1.2,
            min_words=3, max_words=6):
    def normalize(word):
        w = word.strip().lower()
        if ' ' in w:
            return None
        if w in norm_map:
            return norm_map[w]
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
        return [w for w in (normalize(x) for x in text.split(',')) if w]

    loaded = []
    for suffix, weight in model_configs:
        preds = base.load_preds(suffix)
        if preds:
            loaded.append((suffix, weight, preds))
        else:
            print(f"  SKIP {suffix} (file missing/incomplete)")

    out = {}
    for name in all_names:
        word_scores = defaultdict(float)
        for suffix, weight, preds in loaded:
            if name not in preds:
                continue
            for rank, word in enumerate(parse_emotions(preds[name])[:8]):
                pos_weight = 1.0 / (rank * rank_decay + 1)
                tv = tv_boost_factor if base.TRAIN_VOCAB.get(word, 0) > 5 else 1.0
                word_scores[word] += weight * pos_weight * tv

        if word_scores:
            sw = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            thr = sw[0][1] * min_score_ratio
            top = []
            for word, score in sw:
                if len(top) < min_words:
                    top.append(word)
                elif score >= thr and len(top) < max_words:
                    top.append(word)
                else:
                    break
        else:
            top = ["neutral"]
        out[name] = top
    return len(loaded), out

def agreement(ref, hyp):
    jac, exact, inter, rank1 = 0.0, 0, 0.0, 0
    n = len(ref)
    for k, rw in ref.items():
        hw = hyp[k]
        R, H = set(rw), set(hw)
        u = len(R | H)
        jac += len(R & H) / u if u else 1.0
        exact += (rw == hw)
        inter += len(R & H) / len(R) if R else 1.0
        rank1 += (rw[0] == hw[0]) if rw and hw else 0
    return dict(jaccard=jac / n, exact=exact / n, recall_of_ref=inter / n,
                top1=rank1 / n)

if __name__ == "__main__":
    all_names = pd.read_csv(f"{DATA_DIR}/track1_track2_candidate.csv")["name"].tolist()

    preds = {}
    for tier, cfgs in TIERS.items():
        n, p = predict(cfgs, base.NORM_MAP, all_names)
        preds[tier] = p
        avg = sum(len(v) for v in p.values()) / len(p)
        print(f"[{tier}] {n} voters loaded, avg {avg:.3f} words/sample")

    ref = preds["full26"]
    rows = []
    for tier in TIERS:
        a = agreement(ref, preds[tier])
        cnt = Counter(w for v in preds[tier].values() for w in v)
        rows.append(dict(tier=tier, voters=len(TIERS[tier]),
                         avg_words=sum(len(v) for v in preds[tier].values()) / len(preds[tier]),
                         **a, top5=cnt.most_common(5)))

    print("\n" + "=" * 88)
    print(f"{'tier':<11}{'voters':>7}{'avg_w':>8}{'Jaccard':>10}{'exact':>9}"
          f"{'ref-recall':>12}{'top1':>8}")
    print("-" * 88)
    for r in rows:
        print(f"{r['tier']:<11}{r['voters']:>7}{r['avg_words']:>8.3f}"
              f"{r['jaccard']:>10.4f}{r['exact']:>9.4f}"
              f"{r['recall_of_ref']:>12.4f}{r['top1']:>8.4f}")
    print("=" * 88)
    for r in rows:
        print(f"{r['tier']:<11} top5: {r['top5']}")

    with open(f"{RESULTS}/cost_tier_agreement.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {RESULTS}/cost_tier_agreement.json")
