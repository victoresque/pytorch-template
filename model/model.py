import sys
sys.path.append('D:\software\Code\code-file\pytorch-model')
import torch
from base import BaseModel
from .model_arc.my_model import setr_pretrain
from .model_arc.decoder import PointHead

#完整模型
class Pointrend_pre(BaseModel):
    def __init__(self,my_pretrain=False):
        super().__init__()
        self.model_net = setr_pretrain()
        self.pointhead = PointHead()
        if my_pretrain:
            pretrain_path = r'D:\software\Code\code-file\pytorch-model\saved\models\setr\0804_154240\checkpoint-epoch30.pth'
            checkpoint = torch.load(pretrain_path,map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            self.model_net.load_state_dict(state_dict)
    def forward(self,x):
        result = self.model_net(x)
        result.update(self.pointhead(x, result["res2"], result["coarse"]))
        return result

if __name__ == "__main__":
    a = torch.randn(1, 3, 512, 512)
    net = Pointrend_pre()

    # net.eval()
    out = net(a)
    for k, v in out.items():
        print(k, v.shape)