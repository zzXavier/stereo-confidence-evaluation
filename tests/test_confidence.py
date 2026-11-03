import unittest
import numpy as np
from src.benchmark import lr_error

class Confidence(unittest.TestCase):
    def test_geometry_and_unmatched_pixels(self):
        left=np.full((3,8),2,dtype="f")
        right=left.copy()
        right[0,2]=np.nan
        e=lr_error(left,right)
        self.assertTrue(np.isinf(e[:,:2]).all())
        self.assertTrue(np.isinf(e[0,4]))
        self.assertEqual(e[1,4],0)
