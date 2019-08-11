# 参考：https://github.com/xiongzihua/pytorch-YOLO-v1
from utils.decode16ChanelOutput import decodeSegmap
import time
import os
import visdom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataGenerator import datasetTrainAndVal, datasetTest
from model.FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from utils.visualize import Visualizer, VisdomLinePlotter
from utils.accuracy import Accuracy

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size = 16
n_class = 16
height, width = 256, 256
epochs = 20
lr = 1e-5
momentum = 0
w_decay = 1e-5
step_size = 50
gamma = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg_model = VGGNet(requires_grad=True)

trainDataset = datasetTrainAndVal('data/train/')
valDataset = datasetTrainAndVal('data/val/')

trainDataloader = DataLoader(
    trainDataset,
    batch_size=batch_size,
    shuffle=True,
    # shuffle=False,
)
valDataloader = DataLoader(
    valDataset,
    batch_size=batch_size,
    shuffle=True,
    # shuffle=False,
)

model = FCN8s(
    pretrained_net=vgg_model,
    n_class=n_class,
).to(device)
# model = FCN16s(
#     pretrained_net=vgg_model,
#     n_class=n_class,
# ).to(device)
# model = FCN32s(
#     pretrained_net=vgg_model,
#     n_class=n_class,
# ).to(device)
# model = FCNs(
#     pretrained_net=vgg_model,
#     n_class=n_class,
# ).to(device)

# model.load_state_dict(torch.load(
#     'modelWeights/model-epoch-19-itera2309-trainLoss-0.017627024097021396.pth'))


def train():
    # BCEWithLogitsLoss、CrossEntropyLoss参考：
    # https://blog.csdn.net/qq_22210253/article/details/85222093
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()

    # optimizer = optim.RMSprop(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=momentum,
    #     weight_decay=w_decay,
    # )

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=0)

    # decay LR by a factor of 0.5 every 30 epochs
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)

    vis = Visualizer(env='main')
    vis2 = VisdomLinePlotter(env_name='lossAndAccuracy')
    bestLoss = np.inf
    for epoch in range(epochs):
        model.train()
        trainTotalLoss = 0.
        start = time.time()
        for trainIter, (testImgTensor, labelTensor) in enumerate(trainDataloader):
            testImgTensor = Variable(testImgTensor)
            labelTensor = Variable(labelTensor)
            testImgTensor = testImgTensor.to(device)
            labelTensor = labelTensor.to(device)

            optimizer.zero_grad()

            output = model(testImgTensor)
            acc = Accuracy(output, labelTensor)

            trainLoss = criterion(output, labelTensor)
            trainTotalLoss += trainLoss.data.item()

            optimizer.zero_grad()
            trainLoss.backward()
            optimizer.step()

            debug = False  # 调试使用

            if (trainIter+1) % 10 == 0:
                meanTotalLoss = trainTotalLoss/(trainIter+1)
                print('Epoch [%d/%d], Iter [%d/%d] iterLoss: %.4f, epochAccumulatedMeanLoss: %.4f, acc: %.4f%%'
                      % (epoch+1, epochs, trainIter+1, len(trainDataloader), trainLoss.data.item(), meanTotalLoss, acc*100))
                vis.plot_train_val(loss_train=meanTotalLoss)  # 每十个itera可视化一次
                vis2.plot(var_name='loss', split_name='train',
                          title_name='Loss', x=epoch*len(trainDataloader)+trainIter, y=meanTotalLoss)
                vis2.plot(var_name='acc', split_name='train',
                          title_name='Accuracy', x=epoch*len(trainDataloader)+trainIter, y=acc)

            if debug and bestLoss > meanTotalLoss and (trainIter+1) % 10 == 0:
                bestLoss = meanTotalLoss
                print('get best test trainLoss %.5f' % bestLoss)
                torch.save(model.state_dict(),
                           './modelWeights/model-epoch-{}-itera{}-trainLoss-{}.pth'.format(epoch, trainIter, meanTotalLoss))
        usedTime = time.time() - start
        print('This epoch uses %ds.' % usedTime)
        torch.save(model.state_dict(),
                   './modelWeights/model-epoch-{}-itera{}-trainLoss-{:.3f}.pth'.format(epoch, trainIter, meanTotalLoss))
        valTotalLoss = 0.0
        model.eval()  # dropout层及batch normalization层进入 evalution 模态
        with torch.no_grad():
            for valIter, (testImgTensor, labelTensor) in enumerate(valDataloader):
                testImgTensor = Variable(testImgTensor)
                labelTensor = Variable(labelTensor)
                testImgTensor = testImgTensor.to(device)
                labelTensor = labelTensor.to(device)

                output = model(testImgTensor)
                acc = Accuracy(output, labelTensor)
                valLoss = criterion(output, labelTensor)

                valTotalLoss += valLoss.data.item()

        meanValLoss = valTotalLoss / len(valDataloader)
        print('*'*40)
        print('Epoch [%d/%d], Iter [%d/%d] thisIterLoss: %.4f, EpochAccumulatedMeanLoss: %.4f'
              % (epoch+1, epochs, valIter+1, len(valDataloader), valLoss.data.item(), meanValLoss))
        print('*'*40)
        vis.plot_train_val(loss_val=meanValLoss)
        vis2.plot(var_name='loss', split_name='val',
                  title_name='Class Loss', x=epoch*len(trainDataloader)+trainIter, y=meanValLoss)
        vis2.plot(var_name='acc', split_name='val',
                  title_name='Accuracy', x=epoch*len(trainDataloader)+trainIter, y=acc)


def inference():
    batch_size = 1
    testDataset = datasetTest('data/test/')
    testDataloader = DataLoader(
        testDataset,
        batch_size=batch_size,
        # shuffle=True,
        shuffle=False,
    )

    model.load_state_dict(torch.load(
        'modelWeights/model-epoch-18-itera2309-trainLoss-0.029.pth',
        # map_location='cpu',
    ))
    for i, (imgName, ImgTensor) in enumerate(testDataloader):
        ImgTensor = Variable(ImgTensor)
        ImgTensor = ImgTensor.to(device)
        ImgArray = ImgTensor.cpu().numpy().squeeze()
        devideImgAndInference(ImgArray, model, imgName[0])


def devideImgAndInference(imgArray: np.array, model, imgName):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # Tensor通道维度下表为0
    # 保存为图片时
    stride = 256
    # 原始img大小：w=6800, h=7200
    chanel, width, height = imgArray.shape
    heightPadded = (height//stride+1)*stride
    widthPadded = (width//stride+1)*stride
    imgPadded = np.zeros((3, widthPadded, heightPadded))
    imgPadded[:, :width, :height] = imgArray
    # print('padded img shape {}'.format(imgPadded.shape))
    start = time.time()
    for i in range(heightPadded//stride):
        for j in range(widthPadded//stride):
            crop = imgPadded[:, j*stride:j*stride+256, i*stride:i*stride+256]
            # print('crop shape {}'.format(crop.shape))
            crop4Dim = np.expand_dims(crop, axis=0)
            cropTensor = torch.from_numpy(crop4Dim).float().to(device)
            outputTensor = model(cropTensor)
            rgbArray = visualize16ChanelOutput(outputTensor, visual=False)
            imgPadded[:, j*stride:j*stride+256,
                      i*stride:i*stride+256] = rgbArray
    print('use time {}s'.format(time.time()-start))
    imgPadded = imgPadded[:, :width, :height]
    R = imgPadded[0, :, :]
    G = imgPadded[1, :, :]
    B = imgPadded[2, :, :]
    rgbImgArray = np.stack([R, G, B], axis=2)
    # testRgbOrder(rgbImgArray)
    # plt.imsave('inferenceResult/{}_label.tif'.format(imgName), rgbImgArray)
    rgbImg = Image.fromarray(rgbImgArray.astype('uint8'), mode='RGB')
    rgbImg.save('inferenceResult/{}_label.tif'.format(imgName))
    pass


def testRgbOrder(rgb: np.array):
    # 测试预测的图片标签数量和顺序
    dic = {}
    width, height, chanel = rgb.shape
    for w in range(width):
        for h in range(height):
            tmp = []
            tmp.append(rgb[w, h, 0])
            tmp.append(rgb[w, h, 1])
            tmp.append(rgb[w, h, 2])
            key = str(tmp)
            if key in dic.keys():
                dic[key] += 1
            else:
                dic[key] = 1
    print(dic.keys())


def visualize16ChanelOutput(output: torch.Tensor, filename=None, visual=True):
    outputArgmax = torch.argmax(
        output.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decodeSegmap(outputArgmax)
    if visual:
        # plt保存图片时候图片通道下表为2即：M*N*chanel
        plt.imsave('inferenceResult/{}.jpg'.format(filename), rgb)
    return rgb


if __name__ == '__main__':
    # train()
    inference()
    pass
