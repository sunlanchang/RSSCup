import numpy as np


def decodeSegmap(_16ChanelImg, numClass=16):
    # 参考：https://www.learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    label_colors = np.array([
        (0, 200, 0),  # 水      田 RGB:
        (150, 250, 0),  # 水  浇 地 RGB:
        (150, 200, 150),  # 旱  耕 地 RGB:
        (200, 0, 200),  # 园      地 RGB:
        (150, 0, 250),  # 乔木林地 RGB:
        (150, 150, 250),  # 灌木林地 RGB:
        (250, 200, 0),  # 天然草地 RGB:
        (200, 200, 0),  # 人工草地 RGB:
        (200, 0, 0),  # 工业用地 RGB:
        (250, 0, 150),  # 城市住宅 RGB:
        (200, 150, 150),  # 村镇住宅 RGB:
        (250, 150, 150),  # 交通运输 RGB:
        (0, 0, 200),  # 河      流 RGB:
        (0, 150, 200),  # 湖      泊 RGB:
        (0, 200, 250),  # 坑      塘 RGB:
        (0, 0, 0)])  # 其他类别 RGB:
    r = np.zeros_like(_16ChanelImg).astype(np.uint8)
    g = np.zeros_like(_16ChanelImg).astype(np.uint8)
    b = np.zeros_like(_16ChanelImg).astype(np.uint8)

    for l in range(0, numClass):
        idx = _16ChanelImg == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=0)
    return rgb
