import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import os


class datasetTrainAndVal(Dataset):
    def __init__(self, root):
        self.trainImgPath = glob.glob(root+'imageDevided/*(2).tif')
        self.trainImgPath.sort()
        self.labelNpyPath = []
        for path in self.trainImgPath:
            labelName = path.split(' ')[0].split('/')[-1]
            labelNpyPath = root+'labelNpy/_16DimensionArray/'+labelName+'_label.npy'
            self.labelNpyPath.append(labelNpyPath)
        self.transforms = transforms.Compose([
            # 像素取值range(0,1)，参考：https://blog.csdn.net/xys430381_1/article/details/85724668
            transforms.ToTensor(),
            # transforms.Normalize([.485, .485, .456, .406],
            #                      [.229, .229, .224, .225]),
        ])

    def __getitem__(self, idx):
        trainImgData = Image.open(self.trainImgPath[idx])
        trainImgTensor = self.transforms(trainImgData)
        labelNpy = np.load(self.labelNpyPath[idx])
        # print(self.labelNpyPath[idx].split('/')[-1].split('.')[-2])
        labelTensor = torch.from_numpy(labelNpy).float()
        # 不使用NIR通道
        return trainImgTensor[1:], labelTensor

    def __len__(self):
        return len(self.trainImgPath)


class datasetTest(Dataset):
    def __init__(self, root):
        self.testImgPath = glob.glob(root+'test/*(2).tif')
        self.testImgPath.sort()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        testImgData = Image.open(self.testImgPath[idx])
        testImgName = self.testImgPath[idx].split(
            '/')[-1].split('.')[-2].split(' ')[0]
        testImgTensor = self.transforms(testImgData)
        return testImgName, testImgTensor[1:]

    def __len__(self):
        return len(self.testImgPath)


if __name__ == '__main__':
    # dataset = dataset('data/train/')
    dataset = datasetTest('data/test/')
    dataset.__getitem__(0)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # 构建迭代器
    iter = iter(dataloader)
    for i in range(4):
        img = next(iter)
    pass
