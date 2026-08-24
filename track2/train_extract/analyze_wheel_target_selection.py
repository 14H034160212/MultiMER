import numpy as np
import pandas as pd

DATA_DIR = "/data/multimer/ACMMM2026/dataset/mer2026-dataset"
VAL_DIR = "/data/multimer/ACMMM2026/track2/pseudo_val"
WHEEL_NPZ = "/data/multimer/ACMMM2026/MERTools/MER2026/MER2026_Track2/emotion_wheel/wheel_mapping.npz"

def parse(s):
    s = str(s).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [w.strip().lower() for w in s.split(",") if w.strip()]

def load_schemes():
    d = np.load(WHEEL_NPZ, allow_pickle=True)
    format_mapping = d["format_mapping"].tolist()
    raw_mapping = d["raw_mapping"].tolist()
    wheel_map_whole = d["wheel_map_whole"].tolist()
    schemes = [wheel_map_whole[w]["level1"] for w in ["wheel1", "wheel2", "wheel3", "wheel4", "wheel5"]]
    return format_mapping, raw_mapping, schemes

def word_bucket_signature(word, format_mapping, raw_mapping, schemes):
    if word not in format_mapping:
        return ()
    level1_whole = []
    for fmt in format_mapping[word]:
        for raw in raw_mapping.get(fmt, [fmt]):
            level1_whole.append(raw)
    sig = []
    for si, wheel_map in enumerate(schemes):
        for level1 in sorted(level1_whole):
            if level1 in wheel_map:
                sig.append((si, wheel_map[level1]))
                break
    return tuple(sig)

def select_v2(words, format_mapping):
    vocab_words = [w for w in words if w in format_mapping]
    return (vocab_words[:5] if vocab_words else words[:5])

def select_v3(words, format_mapping, raw_mapping, schemes, max_words=5):
    vocab_words = [w for w in words if w in format_mapping]
    if not vocab_words:
        return words[:5]
    sigs = {w: set(word_bucket_signature(w, format_mapping, raw_mapping, schemes)) for w in vocab_words}
    chosen, covered = [], set()
    remaining = list(vocab_words)
    while remaining and len(chosen) < max_words:
        best_w, best_gain = None, -1
        for w in remaining:
            gain = len(sigs[w] - covered)
            if gain > best_gain:
                best_w, best_gain = w, gain
        if best_gain <= 0:
            break
        chosen.append(best_w)
        covered |= sigs[best_w]
        remaining.remove(best_w)

    for w in vocab_words:
        if len(chosen) >= max_words:
            break
        if w not in chosen:
            chosen.append(w)
    return chosen

def coverage(words_list, format_mapping, raw_mapping, schemes):
    total_buckets = 0
    for words in words_list:
        buckets = set()
        for w in words:
            buckets |= set(word_bucket_signature(w, format_mapping, raw_mapping, schemes))
        total_buckets += len(buckets)
    return total_buckets

def main():
    format_mapping, raw_mapping, schemes = load_schemes()
    split = pd.read_csv(f"{VAL_DIR}/holdout_split.csv")
    holdout_names = set(split[split.is_holdout]["name"])

    th = pd.read_csv(f"{DATA_DIR}/track2_train_human.csv")
    tm = pd.read_csv(f"{DATA_DIR}/track2_train_mercaptionplus.csv")
    rows = []
    for df in [th, tm]:
        for _, r in df.iterrows():
            if r["name"] in holdout_names:
                continue
            rows.append(parse(r["openset"]))

    print(f"Analyzing {len(rows)} training rows...")

    v2_targets = [select_v2(w, format_mapping) for w in rows]
    v3_targets = [select_v3(w, format_mapping, raw_mapping, schemes) for w in rows]

    v2_cov = coverage(v2_targets, format_mapping, raw_mapping, schemes)
    v3_cov = coverage(v3_targets, format_mapping, raw_mapping, schemes)
    print(f"Total (scheme,bucket) coverage -- v2: {v2_cov}, v3: {v3_cov}  ({(v3_cov/v2_cov-1)*100:+.1f}%)")

    v2_lens = [len(t) for t in v2_targets]
    v3_lens = [len(t) for t in v3_targets]
    print(f"Mean target length -- v2: {np.mean(v2_lens):.2f}, v3: {np.mean(v3_lens):.2f}")

    n_diff = sum(1 for a, b in zip(v2_targets, v3_targets) if a != b)
    print(f"Rows where v2 and v3 target selection differ: {n_diff}/{len(rows)} ({n_diff/len(rows)*100:.1f}%)")

    print("\nSample differing examples:")
    shown = 0
    for orig, a, b in zip(rows, v2_targets, v3_targets):
        if a != b and shown < 12:
            print(f"  GT={orig}\n    v2={a}\n    v3={b}")
            shown += 1

if __name__ == "__main__":
    main()
