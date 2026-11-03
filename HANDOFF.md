# Stereo confidence evaluation handoff

## Latest verified state · 2026-11-03

The public README and experiment artifacts are available from the [repository](https://github.com/zzXavier/stereo-confidence-evaluation).

Verified experiment revision: `f9e779d2508e3a55e0a44b61f10164faad658222`. A clean GitHub checkout reran the development demo and three held-out scenes with the original SceneFlow checkpoint. The complete per-scene CSV and coverage figure were regenerated.

Next local command: `python run.py evaluate`.

## Earlier implementation notes


Run `cpu-384w-12iter` completed. Weights are the authors' SceneFlow-only checkpoint. The Motorcycle demo and three held-out scenes ran in both directions. Two regression tests covered right-view warping and the Middlebury PFM scale convention.

Latest commands: `python run.py demo`, `python run.py reproduce`, `python run.py evaluate`. Next check: `python run.py evaluate` in the repository, or `python run.py reproduce --out scratch/fresh` for independent inference. Retain the input manifest before changing scene and resolution. The README documents the limits of all three region masks.
