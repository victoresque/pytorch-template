import torch,os,numpy
import matplotlib.pylab as plt
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image

class root_Dataset(Dataset):
    '''
    自定义加载数据集
    '''
    def __init__(self,data_dir,crop=True,train=True):
        self.data_dir = data_dir
        self.train = train
        self.crop = crop

        #建立数据列表
        images,labels = [],[]
        if self.train:
            for name in os.listdir(os.path.join(data_dir,'imgs','train')):
                images.append(os.path.join(data_dir,'imgs','train',name))
                labels.append(os.path.join(data_dir,'anno','train',name.split('.')[0]+'.png'))
        else:
            for name in os.listdir(os.path.join(data_dir,'imgs','test')):
                labels.append(os.path.join(data_dir,'imgs','test',name))
                labels.append(os.path.join(data_dir,'anno','train',name.split('.')[0]+'.png'))
        self.labels = labels 
        self.images = images
        # print(self.labels)

    def __getitem__(self, index):

        #读取图像
        img_path,label_path =  self.images[index],self.labels[index]
        imgs = Image.open(img_path).convert("RGB")
        lbls = Image.open(label_path).convert("L")
        # a =numpy.array(imgs)
        #图像变换
        trans = my_trans(imgs,lbls,is_crop=True)
        self.same_trans = trans
        imgs ,lbls = self.same_trans.process()

        return imgs,lbls

    def __len__(self):
        return len(self.images)

class My_Dataset2(Dataset):
    '''
    图像，标签，边缘数据集
    '''
    def __init__(self,data_dir,crop=True,train=True):
        self.data_dir = data_dir
        self.train = train
        self.crop = crop

        #建立数据列表
        images,labels,edges = [],[],[]
        if self.train:
            for name in os.listdir(os.path.join(data_dir,'imgs')):
                images.append(os.path.join(data_dir,'imgs',name))
                labels.append(os.path.join(data_dir,'label',name.split('.')[0]+'.png'))
                edges.append(os.path.join(data_dir,'edge',name.split('.')[0]+'.png'))
        self.labels = labels 
        self.images = images
        self.edges = edges
        # print(self.labels)

    def __getitem__(self, index):

        #读取图像
        img_path,label_path,edge_path =  self.images[index],self.labels[index],self.edges[index]
        imgs = Image.open(img_path).convert("RGB")
        lbls = Image.open(label_path).convert("L")
        edgs = Image.open(edge_path).convert("L")
        # a =numpy.array(imgs)
        #图像变换
        angle = transforms.RandomRotation.get_params([-180, 180])
        imgs = imgs.rotate(angle)
        lbls = lbls.rotate(angle)
        edgs = edgs.rotate(angle)
        to_tensor = transforms.ToTensor()#会归一化到（0，1）之间
        imgs = to_tensor(imgs)
        lbls = torch.LongTensor(numpy.array(lbls))
        edgs = torch.LongTensor(numpy.array(edgs))
        #归一化
        normal = transforms.Normalize(mean=[0.4754358, 0.35509014, 0.282971],std=[0.16318515, 0.15616792, 0.15164918])
        imgs = normal(imgs)

        return imgs,lbls,edgs

    def __len__(self):
        return len(self.images)

class my_trans():
    '''
    图像增强
    '''
    def __init__(self,image,label,is_crop=True,is_rotate=True):
        self.image = image
        self.label = label
        self.is_crop = is_crop
        self.is_rotate = is_rotate

    def process(self):
        image_p = self.image
        label_p = self.label
        #裁剪
        if self.is_crop:
            crop_pra = transforms.RandomCrop.get_params(image_p, (512, 512))
            image_p = transforms.functional.crop(image_p, *crop_pra)
            label_p = transforms.functional.crop(label_p, *crop_pra)
        #旋转
        if self.is_rotate:
            angle = transforms.RandomRotation.get_params([-180, 180])
            image_p = image_p.rotate(angle)
            label_p = label_p.rotate(angle)
        #转化为tensor
        to_tensor = transforms.ToTensor()#会归一化到（0，1）之间
        image_p = to_tensor(image_p)
        label_p = torch.LongTensor(numpy.array(label_p))
        #归一化
        normal = transforms.Normalize(mean=[0.4754358, 0.35509014, 0.282971],std=[0.16318515, 0.15616792, 0.15164918])
        image_p = normal(image_p)
        # normal = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normal = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return image_p, label_p  #image.shape(3,H,W)   label.shape(H,W)


def show(batch_size,img_tensor1,img_tensor2):
    '''
    画出dataset的图片和标签
    '''
    plt.figure()
    for i, img in enumerate(zip(img_tensor1,img_tensor2)):
        img1, img2=img[0],img[1] 
        img1 = img1.permute(1,2,0)
        # img2 = img2.permute(1,2,0) 
        if torch.is_tensor(img1):
            img1 = img1.numpy()
        if torch.is_tensor(img2):
            img2 = img2.numpy()
        plt.subplot(2,batch_size,i+1)
        plt.imshow(img1)
        plt.axis('off')
        plt.subplot(2,batch_size,(i+1)+batch_size)
        plt.imshow(img2, cmap='gray')
        plt.axis('off')
        #数据类型为（h,w,c）
    plt.show()



if __name__ == "__main__":

    data_dir = r'D:\software\Code\code-file\image\mydata\EGnet'
    batch_size = 4
    mydata = My_Dataset2(data_dir,train=True)
    mydataloader = DataLoader(mydata, batch_size, shuffle=True,num_workers=1)
    img_tensor1,img_tensor2,img_tensor3 = next(iter(mydataloader))
    # show(batch_size,img_tensor1,img_tensor2)
    print(img_tensor1.shape)
    print(img_tensor2.shape)
    print(img_tensor3.shape)

    # img_path = r'D:\software\code\codefile\image_root\mydata\root_data2\imgs\train\1_0_0.jpg'
    # imgs = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    # a = imgs

    
    
    