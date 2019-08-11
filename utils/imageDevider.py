import glob
from PIL import Image
import numpy as np
import os


def devideImg(originImgs):
    # 参考：https://blog.csdn.net/qq_39938666/article/details/93011014
    for originImg in originImgs:
        img = Image.open(originImg)
        w = np.array(img).shape[0]
        h = np.array(img).shape[1]
        for x_idx in range(w // 100):
            for y_idx in range(h // 100):
                if x_idx * 100 + 256 < w and y_idx * 100 + 256 < h:
                    img1 = Image.fromarray(np.array(img)[
                        x_idx * 100:x_idx * 100 + 256, y_idx * 100:y_idx * 100 + 256, :], mode='CMYK')
                    imgName = originImg.split('.')[-2].split('/')[-1]
                    img1.save(os.path.join(
                        savedir, '{}_{}_{}.tif'.format(str(x_idx), str(y_idx), imgName)))
        print(originImg+' done!')


def devideLabel(originLabels):
    for originLabel in originLabels:
        label = Image.open(originLabel)
        w = np.array(label).shape[0]
        h = np.array(label).shape[1]
        for x_idx in range(w // 100):
            for y_idx in range(h // 100):
                if x_idx * 100 + 256 < w and y_idx * 100 + 256 < h:
                    img1 = Image.fromarray(np.array(label)[
                        x_idx * 100:x_idx * 100 + 256, y_idx * 100:y_idx * 100 + 256, :], mode='RGB')
                    imgName = originLabel.split('.')[-2].split('/')[-1]
                    img1.save(os.path.join(
                        savedir, '{}_{}_{}.tif'.format(str(x_idx), str(y_idx), imgName)))
        print(originLabel+' done!')


def testImgSize():
    import glob
    from PIL import Image
    imgPaths = glob.glob('data/train/imageDevided/*.tif')
    for imgPath in imgPaths:
        img = Image.open(imgPath)
        if img.size != (256, 256):
            print(imgPath)


'''
一个图片切割出的图片数量：
行切割文件数：（7200-256)/100 + 1后向下取整为70
列切割文件数：（6800-256)/100 + 1后向下取整为66
所以一个图片切割出：70*66=4620
'''

if __name__ == '__main__':
    # savedir = 'data/train/imageDevided/'
    # originImgs = glob.glob('data/train/train/*(2).tif')
    # originLabels = glob.glob('data/train/train/*_label.tif')
    savedir = 'data/val/imageDevided/'
    originImgs = glob.glob('data/val/val/*(2).tif')
    originLabels = glob.glob('data/val/val/*_label.tif')
    devideImg(originImgs)
    devideLabel(originLabels)
    pass
