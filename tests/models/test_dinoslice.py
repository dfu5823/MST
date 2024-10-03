import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch 
from mst.models import DinoV2ClassifierSlice, DinoV2ClassifierSliceMulti

input = torch.randn((1, 1, 32, 224, 224))

device=torch.device('cuda')

# model = ResNetSlice(in_ch=3, out_ch=2, pretrained=True)
model = DinoV2ClassifierSlice(in_ch=1, out_ch=2, pretrained=True, 
    rotary_positional_encoding='RoPE',
    # rotary_positional_encoding='LiRE',
    # use_bottleneck=True,
    # use_slice_pos_emb=True
    # enable_trans=False,
    use_bottleneck=True
)
# model = DinoV2ClassifierSliceMulti(in_ch=1, out_ch=2)
model.to(device)

pred = model(source=input.to(device), save_attn=False)
print(pred.shape)
print(pred)
