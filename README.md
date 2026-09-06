# Ophthalmic Referral CNN

A deep learning system that classifies eye fundus photos into three categories — **Cataract**, **Diabetic Retinopathy**, and **Normal** — built as a course project and framed around a concrete use case: helping a hospital decide which exams need a specialist and which don't.

This README is a an account of what was built, what the numbers say, and where the project's weak points are. No claim below is aspirational — everything is backed by a saved notebook output or the results table.

## The problem it's framed around

The scenario is a **hospital referral filter**: a queue of ophthalmic exams comes in, and the system's job is to flag the ones that likely need a specialist (Cataract or Diabetic Retinopathy) and let the clearly Normal ones through without adding to the specialist's workload.

In that setting, the two error types are not equally bad:vv
- **False negative** (a diseased eye labeled Normal) — the serious one. The patient doesn't get referred and the condition goes undiagnosed.
- **False positive** (a normal eye sent for review) — a wasted specialist slot, but not a harm to the patient.

That asymmetry drives most of the design decisions below: class weighting, choosing **F1-macro** (not raw accuracy) as the model-selection criterion during grid search, and tracking **recall on Cataract** specifically, since it's the minority class and the one a naive model is most likely to miss.

## Dataset

- 2,570 eye fundus images, RGB, split across three folders: `cataract` (398, ~15.5%), `diabetic_retinopathy` (1,098, ~42.7%), `normal` (1,074, ~41.8%). Clearly imbalanced toward the two majority classes.
- ~1,454 unique patients (filenames follow `{id}_{left|right}.jpg`), so most patients contribute one or two images (left/right eye).
- Integrity was checked before anything else: 0 corrupted files, 0 duplicate images, 0 duplicates that cross class boundaries. Most images are 512×512; 100 (all in the `cataract` folder, likely a different capture device) are larger. Everything is padded — not stretched — to 224×224 to avoid distorting the eye shape.
- **Split is at the patient level, not the image level**, using `StratifiedGroupKFold` (~60/20/20, train/val/test). why this matters? if a random image-level split were used, the two images of the same patient could land in different sets, and since both eyes of one patient look very similar, that would leak information into validation/test and inflate every metric. The split was verified to have zero patient overlap between train, validation, and test.
- Test set: 507 images (82 Cataract / 218 Diabetic Retinopathy / 207 Normal). All numbers in the results section below are computed on this held-out set.
- A second, synthetic data source is also used in the multimodal notebook: a short clinical-report sentence per image (`ocular_reports.csv`), generated from the image's own label. More on why that matters below.
- Raw images and reports aren't in this repo (see `.gitignore`)— only code and the derived, non-sensitive artifacts (`artifacts/results.csv`, split indices, prediction arrays) are versioned.

## Project structure

```
notebooks/
  01_eda.ipynb                 EDA, integrity checks, patient-grouped split
  02_cnn_from_scratch.ipynb    Baseline + regularized CNN
  03_transfer_learning.ipynb   EfficientNetB0 / ResNet50, feature extraction + fine-tuning
  04_multimodal.ipynb          Image + text fusion, text-only baseline
  05_clinical_evaluation.ipynb Confusion matrices, ROC/PR, comparison table (no GPU needed)
  06_explainability.ipynb      Grad-CAM
src/
  bootstrap.py   Colab vs local environment/path handling
  data.py        Dataset loading, patient split, results-table helpers
  viz.py         Plotting and Grad-CAM
artifacts/
  results.csv    Committed comparison table (the source of the numbers above)
```

The project is organized as six sequential Jupyter notebooks under `notebooks/`, developed and run on **Google Colab** (GPU runtime, Google Drive for persistence). Notebooks 02–04 and 06 need a GPU and the raw dataset; notebook 05 deliberately doesn't — it only reads the small, committed CSV/`.npy` artifacts the other notebooks produce, so it can run on a laptop with no GPU and no dataset.

1. **`01_eda.ipynb`** — Loads and validates the dataset, checks for corruption/duplicates, inspects the per-patient structure, and builds the patient-grouped train/val/test split. Persists the split indices so every later notebook reconstructs the exact same three sets without re-running this logic.
2. **`02_cnn_from_scratch.ipynb`** — A plain 3-block Conv2D+MaxPooling CNN trained from scratch, as the reference point everything else is measured against. Trained with and without class weighting, then a 9-combination grid search over L2 regularization × Dropout (selected by validation F1-macro) to address the gap between training and validation performance.
3. **`03_transfer_learning.ipynb`** — EfficientNetB0 and ResNet50, both pretrained on ImageNet, each run through feature extraction (frozen backbone) and then fine-tuning. Fine-tuning used a 9-combination grid search (learning rate × number of unfrozen layers), also selected by validation F1-macro.
4. **`04_multimodal.ipynb`** — Combines a frozen ResNet50 image branch with text embeddings from **BERTimbau** (a Portuguese BERT model) over the synthetic clinical reports, fused through dense layers. Also includes a **text-only baseline** (TF-IDF + Logistic Regression) built specifically to check how much of the answer the synthetic report text gives away on its own — see the caveat section below.
5. **`05_clinical_evaluation.ipynb`** — The GPU-free evaluation layer. Rebuilds confusion matrices, ROC/precision-recall curves, and sensitivity-vs-specificity-by-threshold plots for every model, purely from committed prediction files, and assembles the comparison table.
6. **`06_explainability.ipynb`** — Grad-CAM visualizations for the regularized CNN, fine-tuned EfficientNetB0, and fine-tuned ResNet50, to sanity-check that the models are looking at plausible regions of the eye rather than artifacts.

A small shared module (`src/`) factors out logic used by more than one notebook: dataset loading and the patient-grouped split (`data.py`), Colab-vs-local path handling (`bootstrap.py`), and the plotting/Grad-CAM functions (`viz.py`). Every notebook fixes random seeds (Python, NumPy, TensorFlow) for reproducibility.


## Results (test set, 507 images)

All numbers below are directly from `artifacts/results.csv`, the table each producer notebook appends its row to.

| Model | Accuracy | F1-macro | AUC-ROC (macro) | Recall Cataract | Recall Diabetic Retinopathy | Recall Normal |
|---|---|---|---|---|---|---|
| Baseline CNN (no class weight) | 94.3% | 0.919 | 0.986 | 76.8% | 100.0% | 95.2% |
| Baseline CNN (class weight) | 93.3% | 0.909 | 0.990 | 82.9% | 100.0% | 90.3% |
| Regularized CNN | 93.7% | 0.914 | 0.989 | 79.3% | 100.0% | 92.8% |
| ResNet50 (feature extraction) | 93.9% | 0.925 | 0.989 | 90.2% | 95.4% | 93.7% |
| ResNet50 (fine-tuned) | 96.8% | 0.958 | 0.991 | 89.0% | 99.5% | 97.1% |
| EfficientNetB0 (feature extraction) | 94.1% | 0.926 | 0.988 | 89.0% | 96.3% | 93.7% |
| **EfficientNetB0 (fine-tuned)** | **97.6%** | **0.965** | **0.992** | **90.2%** | **100.0%** | **98.1%** |
| Text-only (TF-IDF + Logistic Regression) | 92.9% | 0.917 | 0.995 | 93.9% | 90.8% | 94.7% |
| Multimodal (ResNet50 + BERTimbau) ⚠️ | 99.2% | 0.990 | 1.000 | 96.3% | 99.5% | 100.0% |

**Reading it honestly:**
- The from-scratch CNNs (baseline, class-weighted, regularized) plateau around 93–94% accuracy but visibly struggle on Cataract recall (77–83%), the minority class — exactly the failure mode the class-weighting and F1-macro selection were meant to catch, and exactly the error type that matters most for this use case.
- Transfer learning is a clear, consistent step up. Fine-tuning helps both backbones (ResNet50: 93.9% → 96.8% accuracy; EfficientNetB0: 94.1% → 97.6%).
- **EfficientNetB0 (fine-tuned) is the best image-only model on every single metric** in the table, including a higher AUC and Cataract recall than fine-tuned ResNet50. ResNet50 was still the backbone carried into the multimodal branch — that choice was made based on a quick visual check with Grad-CAM (its heatmaps were easier to read and explain) and familiarity, not on this comparison table. 
- The multimodal row (99.2% accuracy) looks like the best model by a wide margin. So we investigated below.

## The multimodal result comes with a caveat

The "clinical reports" fed into the multimodal model and the text-only baseline are **synthetic**: they were generated from each image's own label, not written by a clinician looking at the image. That means the report text can leak the diagnosis almost directly, independent of anything the model learns about the image.

The text-only baseline (TF-IDF + Logistic Regression, no image input at all) makes that concrete: **92.9% accuracy and 93.9% Cataract recall using text alone**, close to the fine-tuned image models and actually *higher* on Cataract recall than any of them. If a model with zero access to pixels can nearly match the best image model, a chunk of the multimodal model's 99.2% is very likely coming from the text branch reading a paraphrased label rather than from genuine image-text reasoning.

We explored the text-only baseline notebook cell so we could expose this risk. But it does mean **the multimodal number shouldn't be quoted as "the model"** — it's a result that needs real (clinician-written) reports to mean what it appears to mean. The image-only, fine-tuned EfficientNetB0 (97.6% accuracy, 90.2% Cataract recall) is the more trustworthy number to lead with. And since we want to reduce the amount of strain put on the clinical team we can't consider this model the best one of the test.

## Clinical impact, quantified

Accuracy percentages are hard to picture, so here's what each model's recall actually means over the 507-image test set (82 Cataract, 218 Diabetic Retinopathy, 207 Normal), computed by multiplying each model's recall by the number of test images in that class:

| Model | Pathology cases missed (of 300) | Normal exams unnecessarily referred (of 207) |
|---|---|---|
| Baseline CNN (no class weight) | 19 (6.3%) | 10 (4.8%) |
| ResNet50 (fine-tuned) | 10 (3.3%) | 6 (2.9%) |
| **EfficientNetB0 (fine-tuned)** | **8 (2.7%)** | **4 (1.9%)** |
| Multimodal ⚠️ (see caveat above) | 4 (1.3%) | 0 (0.0%) |

Going from the untuned baseline to the fine-tuned EfficientNetB0 more than halves the number of diseased patients who'd slip through as "Normal" — 19 down to 8 — while also cutting unnecessary specialist referrals from 10 to 4. On the workload side: 207 of the 507 test exams (41%) are truly Normal, so a model that correctly clears most of them is directly cutting the specialist queue, not just improving a scoreboard metric.

Two honest remarks on these numbers: (1) they're arithmetic on the existing test-set recall values, not a prospective or clinical trial measurement; (2) the test set is 507 images, so each single Cataract case is worth about 1.2 percentage points of recall — these are directionally solid but not precise to a decimal place.

## Explainability

`06_explainability.ipynb` runs Grad-CAM on the regularized CNN, fine-tuned EfficientNetB0, and fine-tuned ResNet50, for one example of each class, to check whether the models are attending to plausible anatomical regions (e.g. the lens for Cataract, exudates for Diabetic Retinopathy) instead of artifacts or borders. This is currently a qualitative spot-check, not a systematic audit — a planned next step (noted in the notebook itself) is running it specifically over the false positives/negatives and low-confidence predictions, which is where it would actually be diagnostic of a real problem.

## Tech stack

TensorFlow/Keras (CNNs, EfficientNetB0, ResNet50), PyTorch + Hugging Face Transformers (BERTimbau), scikit-learn (metrics, TF-IDF baseline, patient-grouped stratified split), pandas/NumPy, Matplotlib. Developed on Google Colab (GPU) with Google Drive as the artifact store; `05_clinical_evaluation.ipynb` is the one notebook designed to also run locally, CPU-only, off committed artifacts.


## Current limitations / honest state of the repo

- The dataset and trained model weights aren't in the repo (size and — for the images — unclear redistribution rights), so the notebooks can't be re-run end-to-end without separately providing `ocular/` and `ocular_reports.csv`.
- The repo is mid-refactor (see commit history): shared logic is being pulled out of a single monolithic notebook into `src/`. Results are still being interpreted. 

