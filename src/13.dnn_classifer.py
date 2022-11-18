import cv2
from cv2 import dnn
import numpy as np

#1.导入模型，创建神经网络
#2.读图
#3。输入张量
#4.得到结果，并显示

#导入模型，创建神经网络
config = "model/bvlc_googlenet.prototxt"
model = "model/bvlc_googlenet.caffemodel"
net = dnn.readNetFromCaffe(config, model)

#读取图片，转成张量
img = cv2.imread("picture_material/beauty_leg2.jpg")
blob = dnn.blobFromImage(img, 1.0, #缩放因子
                  (224, 224), 
                  (104, 117, 123))

net.setInput(blob)
r = net.forward()

#读入类目
classes = []
path = "model/synset_words.txt"
with open(path, 'rt') as f:
    classes = [x [x.find(" ") + 1:]for x in f]

order = sorted(r[0], reverse = True)
z = list(range(3))
for i in list(range(0, 3)):
    print("匹配:", classes[z[i]], end = '')
    print("类所在行：", z[i] + 1, ' ', '可能性：', order[i])