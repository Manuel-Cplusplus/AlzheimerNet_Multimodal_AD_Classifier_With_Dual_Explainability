# AlzheimerNet - MULTIMODAL ALZHEIMER DISEASE CLASSIFIER WITH DUAL EXPLAINABILITY

**Case study - Computer Vision course**

**Authors:**
- Manuel Carlucci - Student ID 855237 - m.carlucci69@studenti.uniba.it
- Vittorio Nitti - Student ID 856698 - v.nitti19@studenti.uniba.it


## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Results](#results)
- [Dataset](#dataset)
- [License](#license)

## Overview

AlzheimerNet is a deep learning pipeline for classifying Alzheimer's Disease stages (**CN** – cognitively normal, **MCI** – mild cognitive impairment, **AD** – Alzheimer's disease) from multimodal brain MRI (**T1, T2, FLAIR**), using the **ADNI4** dataset.

The notebook documents the full research process end to end, across three progressively refined classification approaches, combined with a dual explainability framework:

1. **Autoencoder-based reconstruction classifier**: 9 group/modality-specific UNet autoencoders (one per class × modality: CN, MCI, AD × T1, T2, FLAIR), whose embeddings are fused per class via channel-wise concatenation and decoded by 3 class-specific decoders. Sessions are classified by the lowest (bias-calibrated) reconstruction error across the three class-specific decoders, with an additional MCI fallback rule for anatomically ambiguous cases. This reached a weighted F1 of 0.52, but reconstruction loss proved a weak, noise-sensitive discriminative signal (inter-decoder MSE differences of only 0.001–0.003).
2. **Hybrid CNN-based classifier**: built on top of the *frozen* pretrained multimodal-autoencoder encoders. Each modality embedding (256,16,16) is reduced via Global Average Pooling to a 256-d vector, projected into a shared latent space, and fused across modalities using **attention pooling** (rather than assuming equal contribution per modality), then passed through a Dropout + MLP head. Despite this design, the representations proved insufficiently discriminative, and the model **collapsed to majority-class prediction across all tasks (AUC = 0.500)** - indicating that reconstruction-trained embeddings did not carry enough task-specific structure for classification.
3. **2.5D ResNet-18 classifier** (final, best-performing approach): for each session, representative axial slices from T1, T2, and FLAIR are stacked as 3 input channels and fed to a **ResNet-18** (initialized via ImageNet transfer learning, fine-tuned end-to-end with a Dropout + Linear classification head) for 4 classification tasks: `CN_MCI_AD` (3-class), `CN_MCI`, `MCI_AD`, `CN_AD` (binary). Class imbalance is handled via `WeightedRandomSampler` + inverse-frequency label-smoothed cross-entropy. Slice-level softmax probabilities are averaged per session to produce patient-level predictions.
4. **Dual explainability**: **Grad-CAM** heatmaps from the ResNet-18's last convolutional layer, fused per patient via the same Gaussian-weighted, central-slice-favoring aggregation used for slice selection, with a sanity check on the fraction of saliency energy falling outside the brain mask; **reconstruction-based anomaly maps** from the class-specific autoencoders (approach 1), showing where a subject's anatomy deviates from each diagnostic group's expected pattern. The two mechanisms provide independent, cross-checkable evidence for each prediction.

## Pipeline

- **Dataset analysis & splitting** - inspection of the ADNI4 metadata/NIfTI files, patient- and session-level statistics, and a session-aware train/eval/test split (to avoid patient leakage across slices of the same subject).
- **Preprocessing** - clipping, brain cropping, resampling to (128,128,128), intensity normalization, and CLAHE contrast enhancement; selection of 16 representative axial slices per volume (evenly spaced from a fixed offset, skipping low slices dominated by skull).
- **Data augmentation** - online flip + small rotation, applied only to the minority AD class to counter class imbalance (1 original + 3 augmented copies per AD volume).
- **Autoencoder stage** - 9 UNet-based encoders (per modality × per class) and 3 class-specific fusion decoders; embeddings fused via channel-wise concatenation. Classification is done via bias-calibrated reconstruction loss + an MCI fallback rule (weighted F1 = 0.52).
- **Hybrid CNN classifier stage** - a classifier built on the *frozen* autoencoder encoders, with GAP-reduced modality embeddings fused via attention pooling and passed through an MLP head. This intermediate approach collapsed to majority-class prediction (AUC = 0.500) across all tasks, motivating the move to an end-to-end supervised backbone.
- **2.5D ResNet-18 classifier** - slice-level training (transfer learning + fine-tuning) with a `WeightedRandomSampler` and label-smoothed, inverse-frequency-weighted cross-entropy for class balance, followed by patient-level evaluation via mean-probability aggregation across slices. This is the best-performing and final classification stage.
- **Final retraining** - all four classification tasks retrained on the combined train+eval pool (session-aware, stratified internal validation split), then evaluated once on the held-out test set.
- **Grad-CAM & reporting** - three-panel visualizations (raw MRI, Grad-CAM heatmap, overlay) and a full per-patient report combining autoencoder anomaly scores, classifier predictions, and Grad-CAM explainability.

## Results

Patient-level performance of the 2.5D ResNet-18 classifier, reported for both evaluation settings described in the paper.

**Evaluation split** (models trained on train-only data; used for model selection/validation):

| Task | Accuracy | Balanced Acc. | Macro Prec. | Macro Rec. | Macro F1 | Macro Spec. | AUC (OvR) |
|---|---|---|---|---|---|---|---|
| CN–MCI–AD (3-class) | 0.560 | 0.468 | 0.461 | 0.468 | 0.463 | 0.746 | 0.724 |
| CN–MCI | 0.721 | 0.690 | 0.730 | 0.690 | 0.693 | 0.690 | 0.687 |
| MCI–AD | 0.545 | 0.593 | 0.555 | 0.593 | 0.499 | 0.593 | 0.552 |
| CN–AD | 0.759 | 0.650 | 0.597 | 0.650 | 0.607 | 0.650 | 0.687 |

**Test split** (final models, retrained on the combined train+eval pool, evaluated once on the held-out test set):

| Task | Accuracy | Balanced Acc. | Macro Prec. | Macro Rec. | Macro F1 | Macro Spec. | AUC (OvR) |
|---|---|---|---|---|---|---|---|
| CN–MCI–AD (3-class) | 0.578 | 0.415 | 0.386 | 0.415 | 0.400 | 0.743 | 0.666 |
| CN–MCI | 0.607 | 0.591 | 0.621 | 0.591 | 0.574 | 0.591 | 0.669 |
| MCI–AD | 0.827 | 0.492 | 0.419 | 0.492 | 0.453 | 0.492 | 0.692 |
| CN–AD | 0.869 | 0.542 | 0.934 | 0.542 | 0.541 | 0.542 | 0.828 |

Best separation is achieved on the CN–AD binary task; the full 3-class problem, and tasks involving MCI, remain the most challenging - consistent with MCI being a clinically transitional, harder-to-separate stage. On the evaluation split, class-balanced metrics show CN–MCI as the most robust task overall, despite CN–AD having the highest raw accuracy.

For reference, the reconstruction-based autoencoder classifier (evaluated on the same 141-session evaluation split, before the ResNet-18 was adopted) reached a weighted F1 of 0.52 (CN F1 0.660, MCI F1 0.380, AD F1 0.250), while the intermediate hybrid CNN classifier collapsed to majority-class prediction across all tasks (AUC = 0.500).


## Dataset

The notebook expects the **ADNI4** dataset (T1, T2, FLAIR MRI volumes in NIfTI format, with session/subject metadata), as sourced via Kaggle in the notebook's original environment. 

## License

This project is distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

You are free to:
- **Share** - copy and redistribute the material in any medium or format
- **Adapt** - remix, transform, and build upon the material

Under the following terms:
- **Attribution** - You must give appropriate credit to Manuel Carlucci and Vittorio Nitti
- **NonCommercial** - You may not use the material for commercial purposes
- **ShareAlike** - If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original

[View the full license](https://creativecommons.org/licenses/by-nc-sa/4.0/)

All files in this project are subject to the license indicated above.

**Authors**
- Manuel Carlucci - m.carlucci69@studenti.uniba.it
- Vittorio Nitti - v.nitti19@studenti.uniba.it

---
> *CV Case Study - Alzheimer Detection | Manuel Carlucci & Vittorio Nitti | Mtr: 855237 & Mtr: 856698*