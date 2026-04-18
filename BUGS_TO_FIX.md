# Bugs Found in Deepfake Detector Project

## 🔴 Critical Bugs

### 1. **eval.py - Inconsistent Model API Usage**
**File:** `src/eval.py` (lines 58-61)
**Issue:** Uses old API instead of config-driven API, inconsistent with `train.py`

```python
# ❌ CURRENT (WRONG):
model = get_model(
    num_classes=cfg["model"]["num_classes"],
    pretrained=cfg["model"]["pretrained"]
).to(device)

# ✅ SHOULD BE:
model = get_model(cfg).to(device)
```

**Impact:** Model may not respect all config settings (e.g., `finetune_layer4`, `finetune_top`, `finetune_stage`)

---

### 2. **infer_frames.py - Duplicate Model Calls & Dead Code**
**File:** `scripts/infer_frames.py` (lines 45-49)
**Issue:** Calls model twice, has duplicate `torch.no_grad()` blocks, and unused variable

```python
# ❌ CURRENT (WRONG):
with torch.no_grad():
    logit = model(x).squeeze()  # ❌ Unused variable
    with torch.no_grad():        # ❌ Duplicate context manager
        logits = model(x)        # ❌ Model called twice!
        prob = torch.softmax(logits, dim=1)[0, 1].item()

# ✅ SHOULD BE:
with torch.no_grad():
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0, 1].item()
```

**Impact:** Wastes computation, inefficient, confusing code

---

### 3. **Model Filename Mismatch**
**File:** `src/train.py` (line 162) vs `src/eval.py` (line 66)
**Issue:** Train saves as `best_finetuned.pth` but eval looks for `best_baseline.pth`

```python
# train.py saves:
"best_finetuned.pth"

# eval.py looks for:
"best_baseline.pth"
```

**Impact:** Evaluation will fail with FileNotFoundError

---

### 4. **sanity_run.py - Wrong Dataset Arguments**
**File:** `scripts/sanity_run.py` (line 51)
**Issue:** `DeepfakeDataset` expects `(root_dir, split, transform)` but called with wrong signature

```python
# ❌ CURRENT (WRONG):
ds = DeepfakeDataset(SANITY_ROOT, transform=transform)

# ✅ SHOULD BE:
ds = DeepfakeDataset(SANITY_ROOT, "train", transform)
# OR if SANITY_ROOT already points to train/:
ds = DeepfakeDataset(os.path.dirname(SANITY_ROOT), "train", transform)
```

**Impact:** Will raise TypeError or incorrect dataset structure

---

## ⚠️ Potential Issues

### 5. **extract_embeddings.py - Model Architecture Assumption**
**File:** `scripts/extract_embeddings.py` (line 39)
**Issue:** Assumes model has `.fc` attribute, but EfficientNet uses `.classifier` and ConvNeXt uses `.classifier[2]`

```python
# ❌ CURRENT (ASSUMES RESNET):
model.fc = torch.nn.Identity()

# ✅ SHOULD HANDLE ALL MODELS:
if hasattr(model, 'fc'):
    model.fc = torch.nn.Identity()
elif hasattr(model, 'classifier'):
    if isinstance(model.classifier, nn.Sequential):
        # EfficientNet or ConvNeXt
        model.classifier[-1] = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()
```

**Impact:** Will fail for EfficientNet or ConvNeXt models

---

### 6. **data.py - No Image File Filtering**
**File:** `src/data.py` (lines 39, 44)
**Issue:** `os.listdir()` includes all files, not just images

```python
# ❌ CURRENT (INCLUDES ALL FILES):
for img in os.listdir(real_dir):
    self.samples.append((os.path.join(real_dir, img), 0))

# ✅ SHOULD FILTER:
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
for img in os.listdir(real_dir):
    if any(img.lower().endswith(ext) for ext in image_extensions):
        self.samples.append((os.path.join(real_dir, img), 0))
```

**Impact:** May try to load non-image files, causing crashes

---

### 7. **video_data_video.py - Potential Empty Slice**
**File:** `src/video_data_video.py` (line 80)
**Issue:** `last_50` mode could result in empty array if T is very small

```python
# ❌ CURRENT (POTENTIAL BUG):
elif self.temporal_mode == "last_50":
    embeddings = embeddings[int(0.50 * T):]  # Could be empty if T < 2

# ✅ SHOULD BE:
elif self.temporal_mode == "last_50":
    start_idx = max(0, int(0.50 * T))
    embeddings = embeddings[start_idx:]
```

**Impact:** Could cause downstream errors if embeddings become empty

---

### 8. **video_data_video.py - Missing "normal" Mode Handling**
**File:** `src/video_data_video.py` (lines 68-81)
**Issue:** No explicit handling for `temporal_mode == "normal"`, though it works by default

```python
# ⚠️ CURRENT (IMPLICIT):
if self.temporal_mode == "shuffle":
    ...
elif self.temporal_mode == "reverse":
    ...
# No explicit "normal" case - works but unclear

# ✅ BETTER (EXPLICIT):
if self.temporal_mode == "normal":
    pass  # Keep original order
elif self.temporal_mode == "shuffle":
    ...
```

**Impact:** Code clarity issue, not a bug but could be confusing

---

## 📋 Summary

**Total Issues Found:** 8
- **Critical Bugs:** 4
- **Potential Issues:** 4

**Files Needing Fixes:**
1. `src/eval.py` - Model API inconsistency
2. `scripts/infer_frames.py` - Duplicate calls and dead code
3. `src/train.py` - Filename mismatch
4. `scripts/sanity_run.py` - Wrong function arguments
5. `scripts/extract_embeddings.py` - Model architecture assumption
6. `src/data.py` - No file filtering
7. `src/video_data_video.py` - Edge case handling

---

## 🔧 Recommended Fix Order

1. **Fix eval.py** (prevents evaluation from working)
2. **Fix train.py filename** (prevents model loading)
3. **Fix infer_frames.py** (performance issue)
4. **Fix sanity_run.py** (prevents sanity check from running)
5. **Fix extract_embeddings.py** (will break with non-ResNet models)
6. **Fix data.py** (defensive programming)
7. **Fix video_data_video.py** (edge case handling)
