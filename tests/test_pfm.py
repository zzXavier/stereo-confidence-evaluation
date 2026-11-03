from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.benchmark import disparity

class StereoTruth(unittest.TestCase):
    def test_scale_is_not_a_disparity_multiplier(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "disp.pfm"
            f.write_bytes(b"Pf\n2 2\n-0.003922\n" + np.array([3,4,1,2],dtype="<f4").tobytes())
            np.testing.assert_array_equal(disparity(f), [[1,2],[3,4]])
