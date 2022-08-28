import torch
from torch import nn
import torchvision.models as models
from einops.layers.torch import Rearrange

__all__ = [
    "ViT_pretrain"
]

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

#vit主干网络
class ViT_pretrain(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, pool = 'cls', channels = 3, emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = torch.nn.Sequential(*list(models.vision_transformer.vit_l_16(pretrained=True).encoder.layers.children())[:depth])
        self.layernorm = nn.LayerNorm(1024)
      
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.layernorm(x)
        return x

        
if __name__ =="__main__":
    v = ViT_pretrain(
    image_size = 256,
    patch_size = 16,
    dim = 1024,
    depth = 10,
    )

    # print(v)

    # img = torch.randn(2, 3, 256, 256)
    # preds = v(img) # (1, 1000)

    # print(preds.shape)
    for (name, module) in v.named_modules():
        print(name)
