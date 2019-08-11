# RSSCup
official website: http://rscup.bjxintong.com.cn/#/theme/3
1. dataGenerator.py is used to load data and feed data to model.
2. train.py includes train and inference process.
3. utils folder:
    1. accuracy.py: define accuracy function.
    2. imageDevider.py: devide image of size 7200*6800 to small image of size 256*256.
    3. oneHot.py: encode 3 chanels label image to 16 chanels label image, in which every chanel is a one hot encoding.
    4. decode16CanelOutput.py: decode model output which is a 16 chanels image to submmit result.