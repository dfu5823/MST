import torch 
from pathlib import Path 
from torchvision.utils import save_image
from mst.models.utils import tensor2image, tensor_mask2image, one_hot, minmax_norm
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D


ds = LIDC_Dataset3D(
    # random_center=True,
    # split='val',
    # noise=True,
    # random_rotate=True,
    # flip=True,
    # image_resize=(224, 224, 32)
    # image_crop=(48, 48, 16)
    # image_crop = (224, 224, 32)
    image_crop = None 
)

print("Dataset Length", len(ds))

item = ds[0]
uid = item["uid"]
img = item['source']
mask = item['mask']
label = item['target']
print(img.min(), img.max(), img.mean(), img.std())
# img_slice = img[:, 19]
# print(img_slice.min(), img_slice.max(), img_slice.mean(), img_slice.std())

print("UID", uid, "Path", item['path'], "Image Shape", list(img.shape), "Label", label, "Mask", mask.sum())

path_out = Path.cwd()/'results/tests'
path_out.mkdir(parents=True, exist_ok=True)
# img = tensor2image(img[:, 19][None])
# .moveaxis(-1, -2)
save_image(tensor2image(img[None]), path_out/'test.png', normalize=True)
# save_image(tensor2image(mask), path_out/'mask.png', normalize=True)
save_image(tensor_mask2image(img[None], one_hot(mask, 2), alpha=0.25), path_out/'overlay.png', normalize=False)