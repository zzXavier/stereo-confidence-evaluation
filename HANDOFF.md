# P10 handoff

Run `cpu-384w-12iter` completed. Weights are the authors' SceneFlow-only checkpoint. The Motorcycle demo and three held-out scenes ran in both directions. Two regression tests covered right-view warping and the Middlebury PFM scale convention.

Latest commands: `python run.py demo`, `python run.py reproduce`, `python run.py evaluate`. Next check: `python run.py evaluate` in the repository, or `python run.py reproduce --out scratch/fresh` for independent inference. Retain the input manifest before changing scene and resolution. The README documents the limits of all three region masks.
