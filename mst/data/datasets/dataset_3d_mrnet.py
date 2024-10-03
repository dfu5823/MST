from pathlib import Path 
import pandas as pd 
import torchio as tio 
import torch.utils.data as data 

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, ZNormalization, CropOrPad

class MRNet_Dataset3D(data.Dataset):
    PATH_ROOT = Path('/home/gustav/Documents/datasets/MRNet/')
    LABEL = 'meniscus'
    # LABEL = 'acl'

    def __init__(
            self,
            path_root=None,
            fold = 0,
            split= None,
            fraction=None,
            transform = None,
            image_resize = None,
            resample=None,
            flip = False,
            random_rotate = False,
            image_crop = (224, 224, 32),
            random_center=False,
            noise=False, 
            to_tensor = True,
        ):
        super().__init__()
        self.path_root = self.PATH_ROOT if path_root is None else Path(path_root)

        if transform is None: 
            self.transform = tio.Compose([
                tio.Lambda(lambda x: x.transpose(-1, 1)),
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x),
                CropOrPad(image_crop, random_center=random_center) if image_crop is not None else tio.Lambda(lambda x: x),
                ZNormalization(per_channel=True, per_slice=False, percentiles=(0.5, 99.5)),
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True) if random_rotate else tio.Lambda(lambda x: x),
                # tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomNoise() if noise else tio.Lambda(lambda x: x),
                # tio.Lambda(lambda x: x/2, types_to_apply=tio.INTENSITY),
                # tio.Clamp((0-0.449)/0.226, (1-0.449)/0.226), #[-1.98, 2.43]
                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x) # [C, W, H, D] -> [C, D, H, W]
            ])
        else:
            self.transform = transform

        # Get split 
        path_csv = self.path_root/'preprocessed/splits/split.csv'
        path_or_stream = path_csv 
        self.df = self.load_split(path_or_stream, fold=fold, split=split, fraction=fraction).sort_values('meniscus', ascending=False).reset_index(drop=True)
        self.item_pointers = self.df.index.tolist()

    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)


    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        uid = int(item['ID'])
        folder = item['Folder']
        target =  int(item[self.LABEL])

    
        img = self.load_img(self.path_root/'preprocessed/data'/folder/'coronal'/f'{uid:04d}.nii.gz') # sagittal T2, coronal T1, and axial PD 
        img = self.transform(img)

        # img = torch.cat([
        #     self.transform(self.load_img(self.path_root/'preprocessed/data'/folder/plane/f'{uid:04d}.nii.gz'))
        #     for plane in ['sagittal', 'coronal', 'axial' ]   
        # ])

        # images = tio.Subject(**{
        #     plane:self.load_img(self.path_root/'preprocessed/data'/folder/plane/f'{uid:04d}.nii.gz')
        #     for plane in ['sagittal', 'coronal', 'axial']
        # })
        # images = self.transform(tio.Resample(images['coronal'])(images)) 
        # img = torch.cat(list(images.values()))
        

        return {'uid':uid, 'source': img, 'target':target} # 'axial':img_axial, 
    
    def load_id(self, id):
        index = self.df[self.df['ID'] == id].index[0]
        return self[index]

    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df