import numpy as np
import pandas as pd
from collections import Counter
import json, re

WHEEL_MAP_PATH = "/data/multimer/ACMMM2026/MERTools/MER2026/MER2026_Track2/emotion_wheel/wheel_mapping.npz"
RESULTS = "/data/multimer/ACMMM2026/results/track2"

data = np.load(WHEEL_MAP_PATH, allow_pickle=True)
format_mapping = data['format_mapping'].item()
raw_mapping    = data['raw_mapping'].item()
wheel_map_whole = data['wheel_map_whole'].item()

wheels = list(wheel_map_whole.keys())
n_schemes = len(wheels) * 2
print(f"Wheels: {wheels}")
print(f"Total schemes: {n_schemes}")

def get_coverage(word):
    w = word.strip().lower()
    if w not in format_mapping:
        return []
    results = []
    for fmt in format_mapping[w]:
        if fmt not in raw_mapping:
            continue
        level1_words = raw_mapping[fmt]
        for wname in wheels:
            wmap = wheel_map_whole[wname]
            for lvl in ['level1', 'level2']:
                if lvl not in wmap:
                    continue
                lvl_map = wmap[lvl]

                for lw in sorted(level1_words):
                    if lw in lvl_map:
                        results.append((wname, lvl, lvl_map[lw]))
                        break
    return results

def count_schemes(word):
    covered = set()
    for wname, lvl, cat in get_coverage(word):
        covered.add((wname, lvl))
    return len(covered)

answer_path = f"{RESULTS}/answer_ensemble_v16_normfix2c.csv"
if not __import__('os').path.exists(answer_path):
    answer_path = f"{RESULTS}/answer_ensemble_v16_normfix2b.csv"
    if not __import__('os').path.exists(answer_path):
        answer_path = f"{RESULTS}/answer_ensemble_v16.csv"

print(f"\nAnalyzing: {answer_path}")
df = pd.read_csv(answer_path)
all_words = Counter()
for row in df['openset']:
    for w in str(row).split(','):
        w = w.strip()
        if w:
            all_words[w] += 1

print(f"\nTop-30 ensemble words with wheel coverage:")
print(f"{'Word':<25} {'Count':>8} {'Schemes':>8}  {'Example categories'}")
print("-" * 70)
for word, cnt in all_words.most_common(30):
    n = count_schemes(word)
    covs = get_coverage(word)
    cats = set(cat for _, _, cat in covs)
    cats_str = ', '.join(sorted(cats)[:3])
    in_fmt = word in format_mapping
    oov_flag = "" if in_fmt else " [OOV]"
    partial = " [PARTIAL]" if 0 < n < 8 else ""
    print(f"  {word:<23}{oov_flag}{partial} {cnt:>8} {n:>8}/10  {cats_str}")

print("\n\nRaw prediction words (before normalization) - checking top OOV:")
raw_words = Counter()
suffix_list = ["gpt_audio", "gemma4_26b", "qwen3omni", "qwen35omni", "sonnet5_vision",
               "gpt55_video", "qwen3vl32b", "qwen3omni_lora_human", "qwen3omni_lora_full",
               "qwen3omni_lora_human_audio", "qwen3omni_lora_full_audio",
               "qwen35_9b", "qwen35_9b_video_t2", "gemma4_31b", "qwen35lora", "gemini25flash"]
for suffix in suffix_list:
    path = f"{RESULTS}/predictions_{suffix}.jsonl"
    if not __import__('os').path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                pred = r.get("prediction") or r.get("openset") or ""
                for w in str(pred).lower().split(','):
                    w = re.sub(r'[^a-z_-]', '', w.strip())
                    if 2 <= len(w) <= 20:
                        raw_words[w] += 1
            except:
                pass

print(f"\nTop-20 raw words with coverage (before NORM_MAP):")
print(f"{'Word':<25} {'Count':>8} {'Schemes':>8}  {'Example categories'}")
print("-" * 70)
for word, cnt in raw_words.most_common(50):
    n = count_schemes(word)
    if n < 10:
        covs = get_coverage(word)
        cats = set(cat for _, _, cat in covs)
        cats_str = ', '.join(sorted(cats)[:3])
        in_fmt = word in format_mapping
        oov_flag = " [OOV]" if not in_fmt else ""
        print(f"  {word:<23}{oov_flag} {cnt:>8} {n:>8}/10  {cats_str}")
