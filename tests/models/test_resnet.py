
import torch 
from mst.models import ResNet

input = torch.randn((1,3,224,224))
model = ResNet(in_ch=3, out_ch=2, spatial_dims=2, pretrained=False, model=34)

# input = torch.randn((1,1,32,224,224))
# model = ResNet(in_ch=1, out_ch=2, spatial_dims=3, pretrained=False, model=34)

num_34 = {n:p.numel() for n,p in model.named_parameters() if p.requires_grad}

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/10**6

pred = model(source=input)
print(pred.shape)
print(pred)
