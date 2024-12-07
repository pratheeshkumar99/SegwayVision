# tests/test_models.py
import unittest
import torch
from segmentation import SegmentationConfig, FCN, LinkNet, UNet, DeepLabV3

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = SegmentationConfig()
        self.batch_size = 2
        self.input_shape = (3, 256, 256)
        
    def test_fcn(self):
        model = FCN(self.config)
        x = torch.randn(self.batch_size, *self.input_shape)
        output = model(x)
        self.assertEqual(output.shape[1], self.config.NUM_CLASSES)
        
    def test_linknet(self):
        model = LinkNet(self.config)
        x = torch.randn(self.batch_size, *self.input_shape)
        output = model(x)
        self.assertEqual(output.shape[1], self.config.NUM_CLASSES)
    
    # Add more tests...

if __name__ == '__main__':
    unittest.main()