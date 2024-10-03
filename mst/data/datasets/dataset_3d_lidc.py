from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch  

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, ZNormalization, CropOrPad

class LIDC_Dataset3D(data.Dataset):
    PATH_ROOT = Path('/home/gustav/Coscine_Public/LIDC-IDRI/')
    LABEL = 'Malignant'

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
            random_rotate=False,
            image_crop = (224, 224, 32),
            random_center=False,
            noise=False, 
            to_tensor = True,
        ):
        self.path_root = self.PATH_ROOT if path_root is None else Path(path_root)
        self.path_root_data = self.path_root/'preprocessed_crop/data'
        self.split =  split

        if transform is None: 
            self.transform = tio.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                tio.Lambda(lambda x: x.moveaxis(1, 2)), # Just for viewing, otherwise upside down
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x),
                CropOrPad(image_crop, random_center=random_center, mask_name='mask') if image_crop is not None else tio.Lambda(lambda x: x),
                
                tio.Clamp(-1024, 800),
                RescaleIntensity((-1,1), in_min_max=(-1024, 800), per_channel=True),

                # tio.Clamp(-1000, 1000),
                # RescaleIntensity((-1,1), in_min_max=(-1000, 1000), per_channel=True),

                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True) if random_rotate else tio.Lambda(lambda x: x),
                tio.Lambda(lambda x: x+(torch.rand_like(x)-0.5)*0.25*torch.rand((1,),)[0], types_to_apply=tio.INTENSITY) if noise else tio.Lambda(lambda x: x),
                # tio.Lambda(lambda x: (x-0.449)/0.226, types_to_apply=tio.INTENSITY),
                # tio.Clamp((0-0.449)/0.226, (1-0.449)/0.226),
                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x) # [C, W, H, D] -> [C, D, H, W]
            ])
        else:
            self.transform = transform


        # Get split file 
        path_csv = self.path_root/'preprocessed/splits/split.csv'
        path_or_stream = path_csv 
        self.df = self.load_split(path_or_stream, fold=fold, split=split, fraction=fraction)#.set_index('scan_id', drop=True)
        self.item_pointers = self.df.index.tolist()

        
    def __len__(self):
        return len(self.item_pointers)
    
    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):   
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        item = self.df.loc[uid]
        target =  item[self.LABEL]
        nodule_idx = item['nodule_idx']
        rel_path = Path(item['patient_id'])/item['study_instance_uid']/item['series_instance_uid']
        path_dir = self.path_root_data/rel_path

        filename = f'img_{nodule_idx}.nii.gz'
        img = self.load_img(path_dir/filename)

        filename = f'seg_{nodule_idx}.nii.gz'
        mask = self.load_map(path_dir/filename)
        
        masks = {}
        if self.split == "test":
            for ann_idx in range(item['annotation_num']):
                masks[f'mask_{ann_idx}'] = self.load_map(path_dir/f"seg_{nodule_idx}_{ann_idx}.nii.gz" ) 
                    
        
        subj = tio.Subject(img=img, mask=mask, **masks)
        subj = self.transform(subj)
        
        if self.split == "test":
            masks = {key: subj[key] for key in masks.keys()}

        return {'uid':uid, 'source': subj['img'], 'mask':subj['mask'], **masks, 'target':target, 
                'affine':img.affine, 'path':str(rel_path), 'filename':filename}
    

    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df