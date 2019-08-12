# RSSCup
official website: http://rscup.bjxintong.com.cn/#/theme/3

First of all, slice original image of size 7200*6800 to new image size of 256*256, the slice main idea is like a convolutional operation. To the slice algorithm for instance, we have a example.tiff image of size 7200*6800, convert it to a numpy ndarray named example and set a kernal matrix size of 256*256, use the kernal matrix sliding on the example matrix. stride equals 100 When sliding. So that each line we get floor((7200-256)/100+1) = 70 images, each colum we get floor((6800-256)/100+1)=66 images and in the result we get 66*70 images size of 256*256. Above operation is just like convolutional operation.

Secondly, convert label tiff image of chanel 4 to chanel 16, each chanel is a one hot encoding that means there is only one number that equals to 1 in each chanel. Because of one hot encoding we could easily calculate loss.

1. dataGenerator.py is used to load data and feed data to model.
2. train.py includes train and inference process.
3. utils folder:
    1. accuracy.py: define accuracy function.
    2. imageDevider.py: devide image of size 7200*6800 to small image of size 256*256.
    3. oneHot.py: encode 3 chanels label image to 16 chanels label image, in which every chanel is a one hot encoding.
    4. decode16CanelOutput.py: decode model output which is a 16 chanels image to submmit result.