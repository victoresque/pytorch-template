import sys,torch
sys.path.append('D:\software\Code\code-file\pytorch-model')
from einops import rearrange
from typing import OrderedDict
from .decoder.decoder import *
from .backbone.VIT import *
from .backbone.VIT_pretrain import *
from base import BaseModel

#非初始化的参数的setr
'''
class setr(BaseModel):
    def __init__(self, 
                        image_size=512,
                        patch_size=16, 
                        num_classes = 2, 
                        dim = 1024, 
                        depth = 12, 
                        heads = 16, 
                        hid_dim = 4096,
                        pretrain = False):
        super().__init__()
        self.pretrain = pretrain
        self.h = int(image_size/patch_size)
        self.w = int(image_size/patch_size)
        self.dim = dim
        self.encoder = ViT(
                image_size =image_size,
                patch_size = patch_size,
                dim=dim,
                depth= depth,
                heads=heads,
                hid_dim=hid_dim,
                dropout = 0.1,
                emb_dropout = 0.1)
        self.decoder = pointrend_setr_decoder(
            in_channels=dim,
            out_channels=num_classes,
            init_=False)
        self.neck = pointrend_middle_decoder(1024,512,init_=False)

    def forward(self, x):
        result = OrderedDict()
        features_in_hook= []
        features_out_hook = []
        def forward_hook(module,data_input,data_output):
            features_in_hook.append(data_input)
            features_out_hook.append(data_output)
        hook= self.encoder.transformer.layers[4][0].register_forward_hook(hook=forward_hook)

        x = self.encoder(x)
        x = rearrange(x, "b (h w) c -> b c h w", h = self.h , w = self.w , c = self.dim)
        x = self.decoder(x)
        hook.remove()

        result['res2']=self.neck(features_in_hook[0][0])
        result['coarse']=x
        return result
'''
#setr网络（预训练）
class setr_pretrain(BaseModel):
    def __init__(self, 
                        image_size=512,
                        patch_size=16, 
                        num_classes = 2, 
                        dim = 1024, 
                        depth = 12):
        super().__init__()
        self.h = int(image_size/patch_size)
        self.w = int(image_size/patch_size)
        self.dim = dim
        #backbone主干
        self.encoder = ViT_pretrain(
                image_size = image_size,
                patch_size = patch_size,
                dim = dim,
                depth = depth,
            )
        #decoder解码头
        self.decoder = pointrend_setr_decoder(
            in_channels=dim,
            out_channels=num_classes,
            init_=True)
        #中间层解码
        self.neck = pointrend_middle_decoder(1024,512,init_=True)

    def forward(self, x):
        result = OrderedDict()
        features_in_hook= []
        features_out_hook = []
        def forward_hook(module,data_input,data_output):
            features_in_hook.append(data_input)
            features_out_hook.append(data_output)
        hook= self.encoder.transformer[0].register_forward_hook(hook=forward_hook)

        x = self.encoder(x)
        x = rearrange(x, "b (h w) c -> b c h w", h = self.h , w = self.w , c = self.dim)
        x = self.decoder(x)
        hook.remove()

        result['res2']=self.neck(features_in_hook[0][0])
        result['coarse']=x
        return result



if __name__ =="__main__":
    a = torch.randn(1, 3, 512, 512)
    net = setr_pretrain()

    #打印模型层和名字
    # for (name, module) in net.named_modules():
    #     print(name)

    out = net(a)
    print(net)
    for k, v in out.items():
        print(k, v.shape)

    