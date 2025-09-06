import sys
root_dir = os.getcwd()
sys.path.append(os.path.join(root_dir, 'nn-SegMamba'))
sys.path.append(os.path.join(root_dir, 'nn-SegMamba/nnUNet'))
sys.path.append(os.path.join(root_dir, 'nn-SegMamba/mednext'))

import os
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class MySegmentation:
    def __init__(self, nnunet_model_dir='Task-1',
                 folds="all"
                 ):
        
        # network parameters
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        self.predictor.initialize_from_trained_model_folder(
            nnunet_model_dir,
            use_folds=folds,
            checkpoint_name='checkpoint_best.pth',
        )

    def process_image(self, image_np, properties):
        ret = self.predictor.predict_single_npy_array(
            image_np, properties, None, None, False)
        return ret
