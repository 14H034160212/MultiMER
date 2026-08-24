<div align="center">

# MultiMER

### Vocabulary-Aligned Multi-Model Ensemble for Fine-grained Emotion Recognition

[![ACM Multimedia 2026](https://img.shields.io/badge/ACM%20Multimedia-2026-5B3FD3)](https://doi.org/10.1145/3767308.3838611)
![MER2026 Track 2](https://img.shields.io/badge/MER2026%20Track%202-1st%20Place-FFD700)
![EW-F1](https://img.shields.io/badge/EW--F1-79.3928-2EA44F)
![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)

[📄 Paper](https://doi.org/10.1145/3767308.3838611) ·
[🚀 Reproduction](#reproduction) ·
[🔐 Privacy](#privacy) ·
[✒️ Citation](#citation)

</div>

## ✨ Overview

This repository contains the code and best submission for *MultiMER:
Vocabulary-Aligned Multi-Model Ensemble for Fine-grained Emotion Recognition*,
presented at the ACM Multimedia 2026 Grand Challenge.

> 🏆 **MultiMER achieved 79.3928 EW-F1 and ranked 1st on MER2026 Track 2
> (MER-FG).**

MultiMER combines 26 voters spanning 10 base model families, including LoRA
fine-tuned Qwen3-Omni variants, locally hosted open-weight models, and commercial
APIs. Their predictions are integrated through cascade weighted voting after
applying NORM_MAP, a manually curated 211-entry vocabulary normalization
dictionary.

### 🌟 Highlights

- 🥇 **Top-ranked result:** 79.3928 EW-F1 on MER2026 Track 2.
- 🤝 **Diverse ensemble:** 26 voters from 10 base model families.
- 🧭 **Vocabulary alignment:** 211 normalization rules applied before voting.
- ⚡ **Flexible deployment:** full, cost-reduced, and API-free voter subsets.

## 📦 Repository contents

### 🏆 The submitted system

The 79.3928 submission is the output of `track2/build_ensemble_v16.py` at
τ=0.29, rank decay 0.7, tv_boost 1.2 and the full 211-rule NORM_MAP. The
prediction file itself is not redistributed here: it is keyed to MER2026 sample
identifiers, and the dataset licence forbids redistributing the dataset,
its annotations, or derived files without written permission. Running the script
against your own copy of the dataset reproduces it exactly.

### 🧩 What produced it

| Stage | Script |
|---|---|
| NORM_MAP (211 rules), the 26-voter `MODEL_CONFIGS`, and the cascade weighted voting that emits the submission | `track2/build_ensemble_v16.py` |
| Coverage of each NORM_MAP target, measured against the official `wheel_mapping.npz` rather than assumed | `track2/train_extract/check_wheel_coverage.py`, `analyze_wheel_target_selection.py` |
| LoRA fine-tuning of the Qwen3-Omni voters | `track2/train_extract/lora_finetune_qwen3omni.py`, `lora_finetune_qwen.py` |

Also included, because the paper points at them: `track2/EXCLUDED_VOTERS.md` (every
candidate voter that was tested and excluded, with word-bias evidence recomputed from
the retained prediction files) and `track2/build_cost_tiers.py` /
`build_ablation_experiments.py` (the cost-reduced voter subsets and the six
single-variable ablation configurations).

### 🤖 Voter inference

One script per model family, covering all 26 voters in the final ensemble:

| Voters | Script |
|---|---|
| 10 LoRA fine-tuned Qwen3-Omni checkpoints (ep2–ep5, full/human, audio/full-modality) | `predict_lora_audio.py`, `predict_lora_fast.py` |
| Qwen3-Omni, Qwen3.5-Omni, Qwen3.5-Omni-Flash / -Plus | `predict_track2_qwen3omni.py`, `predict_track2_qwen35omni.py`, `predict_track2_omniflash.py`, `predict_track2_omniplus.py` |
| Qwen3-VL-32B, Qwen3-VL-Plus, Qwen3.5-9B (text and video) | `predict_track2_qwen3vl.py`, `predict_track2_qwen3vlplus.py`, `predict_track2_qwen.py` |
| Gemma-4-26B / 31B, Gemma-4-26B audio | `predict_track2_gemma4.py`, `predict_track2_gemma4audio.py` |
| Aria | `predict_track2_aria.py` |
| GPT-4o Audio, GPT-5.5 | `predict_track2_audio_gpt.py`, `predict_track2_video_gpt55.py` |
| Claude Sonnet 5 | `predict_track2_video_claude.py` |
| Gemini 2.5 Flash | `predict_track2_gemini25flash.py` |

<a id="reproduction"></a>

## 🛠️ Environment

Python 3.9 or newer. Install the PyTorch wheel matching your CUDA version first
(<https://pytorch.org/get-started/locally/>), then the rest:

```bash
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml && conda activate multimer
```

The pinned versions are the ones the submitted system was run with, so they are
lower bounds rather than exhaustively tested minima. Only the local voters need
`torch`, `transformers`, `peft`, `accelerate` and `bitsandbytes`; running the
API voters alone needs just `openai`, `anthropic`, `requests`, `pandas` and
`tqdm`.

## 🚀 Reproducing the submitted system

```bash
cd track2
python build_ensemble_v16.py --suffix repro --ratio 0.29 --rank_decay 0.7
```

This consumes the per-voter prediction files
`results/track2/predictions_<voter>.jsonl`, one per entry in `MODEL_CONFIGS`.
They are not redistributed here: regenerate them with the inference scripts above,
which require the MER2026 dataset from the challenge organisers.

```bash
python build_ablation_experiments.py   # the six single-variable ablation configurations
python build_cost_tiers.py             # cost-tier agreement against the submitted file
```

`build_cost_tiers.py` first rebuilds the submitted prediction file, then measures
how closely each cheaper voter subset matches it. The API-free 18-voter subset
recovers 87.2% of the submitted system's predicted words.

<a id="privacy"></a>

## 🔐 Data privacy and permitted use

MER2026 is a gated dataset provided for approved academic research and challenge
participation. Its dataset card prohibits redistributing the dataset, annotations,
or derived files without written permission. This repository therefore does not
include the dataset, per-voter prediction files, or the submitted prediction file.

The API-backed inference scripts transmit task data to third-party services:

| Provider | Data sent |
|---|---|
| OpenAI | WAV audio or sampled video frames, plus Chinese subtitles |
| Anthropic | Sampled video frames and Chinese subtitles |
| Google | Sampled video frames and Chinese subtitles |
| Alibaba Cloud | WAV audio or sampled video frames, plus Chinese subtitles |

> [!IMPORTANT]
> Before running an API-backed script, obtain any permission required by the
> dataset licence and confirm that the provider's terms, retention controls,
> processing region, and account settings are appropriate for the data.

Do not run these scripts with restricted or personal data unless that transfer is
explicitly authorised. The local open-weight inference scripts do not send task
data to an external API.

Keep credentials in environment variables only. Do not commit credentials, endpoint
identifiers, prediction outputs, media, subtitles, logs containing request URLs, or
other dataset-derived artifacts. Revoke and rotate a credential immediately if it is
accidentally exposed.

## 🔑 Credentials

Every API-backed voter reads its key from an environment variable — no key is stored in
this repository:

```bash
export ANTHROPIC_API_KEY=...   # Claude Sonnet 5
export GOOGLE_API_KEY=...      # Gemini 2.5 Flash
export DASHSCOPE_API_KEY=...   # Alibaba Cloud: Qwen3.5-Omni, -Flash, -Plus, Qwen3-VL-Plus
export DASHSCOPE_BASE_URL=...  # Alibaba Cloud OpenAI-compatible workspace endpoint
export DASHSCOPE_API_URL=...   # Alibaba Cloud multimodal-generation workspace endpoint
export OPENAI_API_KEY=...      # GPT-4o Audio, GPT-5.5 (read implicitly by the OpenAI SDK)
```

The locally hosted voters need no credentials: Gemma-4-26B/31B, Gemma-4-26B audio,
Qwen3-Omni, Qwen3-VL-32B, the two Qwen3.5-9B variants, Aria, and the ten LoRA
checkpoints all run from local weights.

<a id="citation"></a>

## ✒️ Citation

```bibtex
@inproceedings{bao2026multimer,
  author    = {Bao, Qiming and Zhang, Chenyuan and Qin, Libo and Zhang, Min},
  title     = {MultiMER: Vocabulary-Aligned Multi-Model Ensemble for Fine-grained Emotion Recognition},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3767308.3838611}
}
```
