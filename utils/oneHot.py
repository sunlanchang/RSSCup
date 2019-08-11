import glob
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

labelIndexDict = {
    '[0, 200, 0]': 0,
    '[150, 250, 0]': 1,
    '[150, 200, 150]': 2,
    '[200, 0, 200]': 3,
    '[150, 0, 250]': 4,
    '[150, 150, 250]': 5,
    '[250, 200, 0]': 6,
    '[200, 200, 0]': 7,
    '[200, 0, 0]': 8,
    '[250, 0, 150]': 9,
    '[200, 150, 150]': 10,
    '[250, 150, 150]': 11,
    '[0, 0, 200]': 12,
    '[0, 150, 200]': 13,
    '[0, 200, 250]': 14,
    '[0, 0, 0]': 15,
}
classNum = len(labelIndexDict.keys())


def getMskShape(label: np.array)->tuple:
    mskShape = list(label.shape)
    mskShape = [e for e in mskShape if e != 3]
    mskShape = (16,)+tuple(mskShape)
    return mskShape


def to1DementionArray(dataType: str):
    labelPath = glob.glob('./data/{}/imageDevided/*label.tif'.format(dataType))
    for path in tqdm(labelPath):
        label = Image.open(path).convert('RGB')
        labelArray = np.array(label)
        mskShape = list(labelArray.shape)
        mskShape.remove(3)
        msk = np.zeros(tuple(mskShape), dtype=np.uint8)
        width, height, chanel = labelArray.shape
        for w in range(width):
            for h in range(height):
                key = str(labelArray[w, h, :].tolist())
                labelIndex = labelIndexDict[key]
                msk[w, h] = labelIndex
        imgLabelName = path.split('/')[-1].split('.')[0]
        np.save(
            './data/{}/labelNpy/_1DimensionArray/{}.npy'.format(dataType, imgLabelName), msk)


def to16DimensionArray(dataType: str):
    labelPaths = glob.glob(
        './data/{}/labelNpy/_1DimensionArray/*.npy'.format(dataType))
    labelSet = set()
    for labelPath in tqdm(labelPaths):
        label1Dim = np.load(labelPath)
        labelSet = labelSet | set(label1Dim.flatten().tolist())
        label16DimArrayShape = (classNum,)+label1Dim.shape
        label16DimArray = np.zeros(label16DimArrayShape, dtype=np.uint8)
        # 参考create one hot: https://github.com/pochih/FCN-pytorch/blob/master/python/CamVid_loader.py
        for c in range(classNum):
            label16DimArray[c][c == label1Dim] = 1
        imgname = labelPath.split('.npy')[-2].split('/')[-1]
        np.save(
            'data/{}/labelNpy/_16DimensionArray/{}.npy'.format(
                dataType, imgname),
            label16DimArray)
    # assert len(labelSet) == 16  # 共16个类别


if __name__ == '__main__':
    # to1DementionArray(dataType='train')
    # to16DimensionArray(dataType='train')
    to1DementionArray(dataType='val')
    to16DimensionArray(dataType='val')

'''
水      田 RGB: 0 200 0
水  浇 地 RGB: 150 250 0
旱  耕 地 RGB: 150 200 150

园      地 RGB: 200 0 200
乔木林地 RGB: 150 0 250
灌木林地 RGB: 150 150 250

天然草地 RGB: 250 200 0
人工草地 RGB: 200 200 0

工业用地 RGB: 200 0 0
城市住宅 RGB: 250 0 150
村镇住宅 RGB: 200 150 150
交通运输 RGB: 250 150 150

河      流 RGB: 0 0 200
湖      泊 RGB: 0 150 200
坑      塘 RGB: 0 200 250
其他类别 RGB: 0 0 0
'''
