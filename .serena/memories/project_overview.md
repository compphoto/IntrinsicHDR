# IntrinsicHDR — Project Overview

Code for the paper **"Intrinsic Single-Image HDR Reconstruction"** (Sebastian Dille, SFU).
Reconstructs an HDR image from a single LDR image by going through an *intrinsic decomposition*
(albedo + shading), hallucinating HDR detail per-component, then recombining and refining.

Primary language: **Python** (62 files, ~11k LOC). Core = PyTorch / PyTorch-Lightning.
A vendored TensorFlow 1.x baseline (SingleHDR) lives under `baselines/`. A scratch Rust crate
sits in `tmp.noindex/hdr_orchestrator/` (outside main tree).

## End-to-end inference flow (inference.py)
1. `load_models()` (intrinsic_decomposition/common/model_util.py) loads two decomposition nets:
   - `ordinal_model` = `MidasNet` (networks/altered_midas/midas_net.py)
   - `real_model`   = `MidasNet_small` (5-in,1-out) — maps ordinal→real inverse-shading
   Weights downloaded from GitHub release v1.0 via torch.hub (ord: vivid_bird_318_300.pt, mrg: fluent_eon_138_200.pt).
2. `load_reconstruction_models()` loads three Lightning modules:
   - shading `LitReconstructor(mode='shading')`  ← sh_weights.ckpt
   - albedo  `LitReconstructor(mode='albedo')`   ← alb_weights.ckpt
   - `LitRefiner()` ← ref_weights.ckpt (only keys starting with "refiner." kept)
3. `intrinsic_hdr()`: clip→resize to max_res & round_32 → `decompose_torch()` →
   `hdr_reconstruction()` → resize back → `blend_imgs()` blends new highlights over original.
4. Outputs `.hdr` (default) or `.exr` (half-float + ZIP). Loop has a `~/.hdr_pause` file
   feature that frees/reloads models. Device auto-selects cuda > mps > cpu.

## Key modules
- **src/decomposition_utils.py**: `decompose_torch` (ordinal→real intrinsic decomposition,
  double-estimation base+full at base_size=384), `equalize_predictions` (MPS-safe closed-form
  single-var least squares — avoids torch.linalg.lstsq which has no MPS kernel),
  `get_quantile`, `blend_albedo*`, `separate_regions`.
- **src/color_utils.py**: `rgb_to_lab` (kornia-based, no linearization; numpy/torch modes).
- **src/model.py**: VGG16 encoder/decoder U-Net (`Model`, `Encoder(_aa)`, `Decoder(_aa)`,
  `BlurPool_Transpose`). NOTE: appears legacy/alt-backbone; production uses MidasNet_small.
- **src/midas/** (midas_net.py, blocks.py, base_model.py): MidasNet_small backbone
  (efficientnet_lite3 encoder + RefineNet fusion). Used by LitReconstructor & LitRefiner.
- **src/msg_loss.py** `MSGLoss`: masked Multi-Scale Gradient loss (4 scales, erosion masking).
- **src/image_derivative.py** `ImageDerivative`: separable tap filters (3/5/7/9/13) for gradients.
- **src/utils.py**: tonemap, sRGB<->linear, OKLab, `apply_rf`/`interp_1d` (CRF apply), augments.

## Lightning modules
- **lit_reconstructor.py** `LitReconstructor`: backbone MidasNet_small('none').
  modes: albedo (7in/3out), shading (4in/1out), combined (7in/3out). `iteration` dispatches
  to albedo_iteration / shading_iteration. Uses inverse-shading repr `inv = 1/(shd+1)`.
  Losses: masked MSE (`dense_criterion`) + MSGLoss (`grad_criterion`). RAdam + CosineAnnealing.
  Has `decompose()` (own copy of decomposition) for non-preprocessed training.
- **lit_refiner.py** `LitRefiner`: 10-channel input refiner (rgb_ldr|alb_hdr|inv_sh_hdr|temp_hdr),
  MidasNet_small. Can run its own alb/sh reconstructors if `non_processed`.

## intrinsic_decomposition/ (vendored Intrinsic-Image-Decomposition, compphoto)
- common/pipeline.py `run_pipeline`: numpy-based decomposition orchestrator (CLI/eval path;
  decompose_torch is the torch inference equivalent).
- common/model_util.py: `load_models`. common/general.py: `round_32`, `get_brightness`,
  guided_filter, tiling/vis helpers. ordinal_util.py, methods.py (baseline IID methods),
  metrics.py (WHDR/LMSE etc.), sh_util.py (spherical harmonics), depth_util.py, normal_util.py,
  midas_loss.py, data_util.py, resolution_util.py (`optimal_resize`).
- networks/altered_midas/: forked MiDaS (midas_net MidasNet, midas_net_custom MidasNet_small,
  blocks, dpt_depth, vit, transforms, base_model).

## baselines/SingleHDR/ (TensorFlow 1.x, vendored)
SingleHDR pipeline: Dequantization_net → Linearization_net → hallucination → refinement.
Used by **dequantize_and_linearize.py** (tf.compat.v1) to produce linear EXR inputs from LDR
PNG/JPG. Checkpoints at baselines/SingleHDR/checkpoints/model.ckpt.

## Helper scripts
- **prepare_input.py**: convert sRGB PNG/JPG/TIFF (falls back to macOS `sips` for JXL/AVIF/HEIC)
  → linear half-float EXR. This is the simple input-prep path (no SingleHDR linearization).
- inference_testing.sh, linearization_testing.sh: bash test drivers.
- notebooks/intrinsicHDR.ipynb (Colab).

## Conventions / gotchas
- Shading is stored/processed in **inverse** form `inv = 1/(shd+1)` throughout.
- Albedo scale is ambiguous; rescaled to 95% quantile = 0.95 at inference (and lstsq-matched to GT in training).
- All processing sizes are rounded to multiples of 32 (`round_32`).
- MPS support is first-class; avoid torch.linalg.lstsq on MPS (falls back to CPU). Uncommitted
  changes in inference.py & src/decomposition_utils.py are MPS/perf + EXR-output tweaks.
- For offline use, many files have commented-out local-path alternatives to torch.hub downloads.
