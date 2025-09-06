# nnUNet/nnunetv2/test-SwinConvAE_DS.py

import torch 
from nnunetv2.model_swinmednext.swinmednext import SwinConvAE_DS

t1 = torch.rand(1, 4, 64, 64, 64).cuda()

model = SwinConvAE_DS(in_channels=4,
                      out_channels=4,
                      use_skip_connections=True,
                      do_deep_supervision=True).cuda()

out = model(t1)

for n in range(len(out)):
    print(out[n].shape)

print(model)