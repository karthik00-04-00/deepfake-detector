# Post-Freeze Tooling Fixes

This document records non-research, post-freeze maintenance fixes.
All changes listed here do NOT alter training, evaluation, datasets,
or reported experimental results.

---

## infer_frames.py
- Removed duplicate `no_grad` context and unused forward pass
- Pure refactor; no semantic change

---

## extract_embeddings.py
- Made head removal backbone-agnostic (ResNet / EfficientNet / ConvNeXt)
- Tooling-only change; no impact on prior embeddings

---

## data.py
- Added image extension filtering with logging
- Defensive-only; semantics unchanged for clean datasets
- Note: data.py is shared with research code

---

## video_data_video.py
- Guarded last_50 against empty slices
- Added explicit normal temporal mode for clarity

---

## Model artifact alias
- `best_finetuned.pth` is the true research artifact (saved by `src.research.train`)
- `best_baseline.pth` may be added as an alias for inference tooling:  
  `cp outputs/models/best_finetuned.pth outputs/models/best_baseline.pth`
- No research code modified

---

**State:** `post-freeze-tooling-stable`  
No research files (`src/research/train.py`, `src/research/eval.py`, `scripts/research/sanity_run.py`) were edited.
