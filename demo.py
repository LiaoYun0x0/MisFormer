
import torch
from src.model import Model
import numpy as np
import cv2

model_config = {
    "dims":[64,128,192,256],
    "depths":[2,2,6,2],
    "window_size":[1,1,1,1],
    "ks":[1,1,1,1],
    "num_attn":5,
    "num_classes":2
}
device = "cuda:1"
weight_path = 'weights/mbformer-bs64-tm-22c/model_8000_0.0196.pth'
model = Model(model_config).to(device)
ckpts = torch.load(weight_path,map_location=device)
model.load_state_dict(ckpts['model'])

img22c = np.load('assets/1.npy') # 3rgb + 19gray
cv2.imwrite('assets/seed_rgb.jpg',img22c[...,:3])
cv2.imwrite('assets/gray.jpg',img22c[...,8])
x = (torch.from_numpy(img22c).float() - 127.5) / 127.5
x = x.unsqueeze(0).to(device)
x = { "seed_image" : x }
y = model(x)
print(y)