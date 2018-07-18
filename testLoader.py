import os, torch, torch.utils.data, scipy.misc, numpy as np, pdb
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='2'


def getInitLabel(allFileName):
    number=[]
    for i in range(len(allFileName)):
        index = allFileName[i].find('_')
        label = allFileName[i][:index]
        if label != 'sphere':
            number.append(int(label))
    InitLabel = list(set(number))
    return InitLabel


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path=path
        self.allFileName=os.listdir(path)
        self.InitLabel=getInitLabel(self.allFileName)

    def __read_image(self, path):
        img = scipy.misc.imread(path)
        img = scipy.misc.imresize(img, [256, 256], interp='nearest')
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        img = img.transpose(2, 0, 1) / 255.
        return img

    def __getitem__(self, idx):
        outputs = []
        albedo=self.__read_image(self.path+str(self.InitLabel[idx])+'_albedo.png')
        shading=self.__read_image(self.path+str(self.InitLabel[idx])+'_shading.png')
        mask=self.__read_image(self.path+str(self.InitLabel[idx])+'_albedo.png')
        input=albedo*shading
        outputs.append(input)
        outputs.append(albedo)
        outputs.append(shading)
        outputs.append(mask)
        return outputs

    def __len__(self):
            return len(self.InitLabel)


if __name__ == '__main__':
    directory = 'motorbike_val/'
    dataset=MyDataset(directory)
    loader=torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=2,shuffle=False)
    for ind,tensors in enumerate(loader):
        print(ind)
        tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
        inp,reflec,shading=tensors
        aa = transforms.ToPILImage()(inp[0].data.cpu()).convert('RGB')
        aa.save( str(ind) + '.jpg')
    print("ok")

