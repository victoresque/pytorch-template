from torch import nn
import torch
from  einops import rearrange

__all__=[   "pointrend_setr_decoder",
            "pointrend_middle_decoder"
        ]

#setr解码头
# class SETR_up_decoder(nn.Module):
#     def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
#         super().__init__()
#         self.decoder_1 = nn.Sequential(
#                     nn.Conv2d(in_channels, features[0], 3, padding=1),
#                     nn.BatchNorm2d(features[0]),
#                     nn.ReLU(inplace=True),
#                     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#                 )
#         self.decoder_2 = nn.Sequential(
#                     nn.Conv2d(features[0], features[1], 3, padding=1),
#                     nn.BatchNorm2d(features[1]),
#                     nn.ReLU(inplace=True),
#                     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#                 )
#         self.decoder_3 = nn.Sequential(
#             nn.Conv2d(features[1], features[2], 3, padding=1),
#             nn.BatchNorm2d(features[2]),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.decoder_4 = nn.Sequential(
#             nn.Conv2d(features[2], features[3], 3, padding=1),
#             nn.BatchNorm2d(features[3]),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         )
#         self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)
#     def forward(self, x):
#         x = self.decoder_1(x)
#         x = self.decoder_2(x)
#         x = self.decoder_3(x)
#         x = self.decoder_4(x)
#         x = self.final_out(x)
#         return x

#pointrend_setr解码头
class pointrend_setr_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256],init_=False):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)
        #是否初始化
        if init_:
            self.decoder_1.apply(weigth_init)
            self.decoder_2.apply(weigth_init)
            self.final_out.apply(weigth_init)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.final_out(x)
        return x

#中间层解码
class pointrend_middle_decoder(nn.Module):
    def __init__(self, in_channels, out_channels,init_=False):
        super().__init__()
        self.neck_1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.neck_2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        #是否初始化
        if init_:
            self.neck_1.apply(weigth_init)
            self.neck_2.apply(weigth_init)

    def forward(self, x):
        x = rearrange(x, "b (h w) c -> b c h w", h = 32 , w = 32 , c = 1024)
        x = self.neck_1(x)
        x = self.neck_2(x)
        return x


#初始化方法一
def weigth_init(m):
    if isinstance(m, nn.Conv2d):    
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):  
        torch.nn.init.xavier_normal_(m.weight.data, 0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
#初始化方法二
'''
def weigth_init2(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):  
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
'''


if __name__=="__main__":
    # x = torch.rand(1,1024,1024)
    # decoder = neck(1024,512)
    # a = decoder(x)
    # print(a.shape)

    a = pointrend_setr_decoder(1024,512,init_=True)