import sys
sys.path.append('D:\software\Code\code-file\pytorch-template')
from torchvision import datasets, transforms
from base import BaseDataLoader
from .my_dataset import root_Dataset

class MyDataLoader(BaseDataLoader):
    """
    根系数据集加载
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = root_Dataset(self.data_dir,train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class vocDataLoader(BaseDataLoader):
    """
    VOC数据加载
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = datasets.VOCSegmentation(self.data_dir, year='2012', image_set='train', download=False,transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ =="__main__":
    d_P = r'D:\software\Code\code-file\image\mydata\test_data'
    batch_size = 4
    Da = MyDataLoader(d_P,batch_size)
    a,b = next(iter(Da))
    print(a.shape)