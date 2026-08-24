# Candidate voters tested and excluded

Companion to *MultiMER: Vocabulary-Aligned Multi-Model Ensemble for Fine-grained
Emotion Recognition* (ACM MM '26 Grand Challenge, DOI 10.1145/3767308.3838611).
Table 4 of the paper shows 8 representative rows; this file documents every candidate
for which we retained a prediction file.

## How this list was produced

Every voter that has a `results/track2/predictions_<name>.jsonl` file but does **not**
appear in `MODEL_CONFIGS` of `build_ensemble_v16.py` (the script that produced the
submitted 79.3928 system) was run and then excluded. From that set we removed:

- synthetic prediction files (`calm_prior`, `calm_worried_prior`, `raw`) — the
  distribution-matching prior discussed as a failure mode in Section 4.4;
- alternate runs of voters that *are* in the final ensemble (different epoch,
  precision, or worried-only filtering of an already-included checkpoint);
- probes with fewer than 1,000 samples (listed separately at the end).

`Top-1 share` is the fraction of samples in which the model's single most frequent
word appears. The paper's exclusion criterion (1) is *any single word > 15% of
outputs*; column `>15%` marks candidates that trip it.

## Large-sample candidates (23)

| Candidate | Samples | Top-1 word | Share | 2nd | 3rd | >15% | Note |
|---|---:|---|---:|---|---|:-:|---|
| `llava_next34b_BAD_anxious99pct` | 1,569 | anxious | 99.6% | melancholic 83.2% | relieved 81.3% | **yes** |  |
| `sonnet45` | 20,000 | neutral | 99.3% | concerned 0.1% | anxious 0.1% | **yes** |  |
| `fable5` | 6,154 | neutral | 96.4% | anxious 0.7% | surprise 0.7% | **yes** | Table 4 (paper): pre-emptive; 88.8% neutral at the ~2k checkpoint (89.0% verified here); subtitle-only, no video |
| `sonnet` | 20,000 | neutral | 95.8% | resigned 0.8% | concerned 0.7% | **yes** |  |
| `opus` | 20,000 | neutral | 88.8% | anxious 1.8% | serious 1.6% | **yes** |  |
| `gemini25pro` | 17,389 | errormax_retries | 86.3% | neutral 9.3% | calm 4.4% | **yes** | Table 4 (paper): unusable, 86% API error rate (quota exhausted) |
| `gemini31pro` | 1,344 | calm | 83.6% | serious 9.1% | attentive 6.8% | **yes** |  |
| `gemini35flash` | 6,783 | calm | 77.2% | serious 12.2% | concerned 11.0% | **yes** |  |
| `gemini31flashlite` | 20,000 | serious | 76.6% | concerned 41.6% | confident 33.3% | **yes** |  |
| `qwen35omniplus` | 20,000 | calm | 66.6% | serious 22.0% | concerned 11.4% | **yes** | Table 4 (paper, text-only variant): harmful; all combined builds < 79.0057 |
| `qwen3vl_30b` | 20,000 | concerned | 45.6% | thoughtful 38.5% | serious 35.5% | **yes** | Table 4 (paper): redundant, 79.6% per-sample overlap with Qwen3-VL-32B |
| `internvl3_14b` | 9,237 | concerned | 36.4% | serious 28.9% | confused 17.9% | **yes** | Table 4 (paper): pre-emptive on output-bias check |
| `qwen3vl235b` | 3,156 | concerned | 28.1% | anxious 14.0% | contemplative 13.9% | **yes** |  |
| `glm46vflash` | 20,000 | concerned | 27.3% | concern 27.1% | confused 16.1% | **yes** | Table 4 (paper): −0.47 EW-F1 |
| `qwenvlmax` | 20,000 | concerned | 26.8% | surprise 25.1% | concern 19.4% | **yes** |  |
| `gemma3_27b` | 20,000 | frustration | 25.7% | sadness 21.0% | resignation 18.1% | **yes** |  |
| `qwen3vl_32b_dense` | 20,000 | concerned | 25.6% | sadness 23.3% | concern 19.9% | **yes** |  |
| `qwen36_35b` | 19,196 | concerned | 25.2% | serious 17.1% | worried 14.8% | **yes** | Table 4 (paper): −0.48 EW-F1, 77.4725→76.9931 |
| `internvl3_38b` | 11,819 | concerned | 23.9% | calm 23.4% | contemplative 19.1% | **yes** |  |
| `glm51` | 20,000 | curious | 18.6% | concerned 15.8% | anxious 14.3% | **yes** | Table 4 (paper): −0.0755 EW-F1 at w=1.5; text-only, no vision path |
| `minicpmo` | 20,000 | frustration | 17.8% | disgust 17.7% | concern 16.7% | **yes** | Table 4 (paper): 79.0220 < 79.3928 baseline |
| `deepseek_v4pro` | 20,000 | amused | 14.6% | curious 14.0% | curiosity 11.0% | no |  |
| `glm52_audio` | 20,000 | frustrated | 10.4% | anxious 9.6% | resolute 7.9% | no |  |

21 of 23 trip criterion (1) directly.
The two that do not (`deepseek_v4pro`, `glm52_audio`) were excluded on negative
marginal EW-F1 rather than on output bias.

## Small-sample probes (<1,000 samples)

Launched and abandoned early, mostly on immediate output-quality or API failures;
we list them for completeness but they were never scored:

- `molmo7b` — 566 samples
- `llava_next34b` — 140 samples
- `kimivl` — 84 samples
- `llava_next34b_v2_BAD_constant32` — 62 samples
- `qwqplus` — 50 samples
- `gemini20flash` — 30 samples
- `phi4mm` — 22 samples
- `phi4mm_test` — 1 samples

## Reproducing this table

```bash
cd track2 && python build_cost_tiers.py   # also prints per-tier word distributions
```

The bias figures above come straight from the retained `predictions_*.jsonl` files;
no numbers here are estimated or reconstructed after the fact.
