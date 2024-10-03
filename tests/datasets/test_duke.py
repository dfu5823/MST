
import torch 
from pathlib import Path 
from torchvision.utils import save_image
from mst.models.utils import tensor2image
from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D


ds = DUKE_Dataset3D(
    split='train',
    fraction=0.05,
    noise=True,
    random_center=True,
    random_rotate=True,
    flip=True,
    #image_resize=(224, 224, 32)
    image_crop=(224, 224, 32)
)

print("Dataset Length", len(ds))

item = ds[0]
uid = item["uid"]
img = item['source']
label = item['target']
print(img.min(), img.max(), img.mean(), img.std())
img_slice = img[:, 19]
print(img_slice.min(), img_slice.max(), img_slice.mean(), img_slice.std())

print("UID", uid, "Image Shape", list(img.shape), "Label", label)

path_out = Path.cwd()/'results/tests'
path_out.mkdir(parents=True, exist_ok=True)
# img = tensor2image(img[:, 19][None])
img = tensor2image(img[None])
save_image(img, path_out/'test.png', normalize=True)