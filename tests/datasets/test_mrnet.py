import torch 
from pathlib import Path 
from torchvision.utils import save_image
from mst.models.utils import tensor2image, tensor_mask2image, one_hot, minmax_norm
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D


ds = MRNet_Dataset3D(
    random_center=True,
    split=None,
    # noise=True,
    # random_rotate=True,
    # flip=True,
    # image_resize=(224, 224, 32)
    # image_crop=(48, 48, 16)
    # image_crop = (224, 224, 32)
    image_crop=None
)

print("Dataset Length", len(ds))
print(len(ds.df['ID'].unique())) # WARNING: ID != PatientID, PatientID is unknown 
print(ds.df['meniscus'].value_counts())

item = ds[9]
# item = ds.load_id(94)
uid = item["uid"]
img = item['source']
label = item['target']
print(img.min(), img.max(), img.mean(), img.std())

print("UID", uid, "Image Shape", list(img.shape), "Label", label)

path_out = Path.cwd()/'results/tests'
path_out.mkdir(parents=True, exist_ok=True)
# img = tensor2image(img[:, 19][None])

save_image(tensor2image(img[None]), path_out/'test.png', normalize=True)
