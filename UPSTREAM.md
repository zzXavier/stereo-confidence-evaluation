# Upstream and data

[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo/tree/6e93ed2169bd858dbb43033988563f3b0bb49506) is an MIT-licensed implementation from Princeton VL. We instantiate its original RAFTStereo and InputPadder classes and use the authors' official SceneFlow-trained weights. The separately fetched code and weights remain at the original distribution points. SHA-256 checks for each downloaded input are in configs/assets-sha256.json.

[Middlebury 2014 stereo](https://vision.middlebury.edu/stereo/data/scenes2014/) provides the public original image pairs and ground-truth disparity. Credit D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nesic, X. Wang and P. Westling, *High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth* (GCPR 2014). The dataset loader retains the original disparity-unit convention. Visualizations in this repository show disparity and error.

This repository provides the SGBM comparison, bidirectional consistency confidence, masks, saved-prediction evaluation and compact controlled subset; it does not retrain RAFT-Stereo.
