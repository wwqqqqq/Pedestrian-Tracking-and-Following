from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# settings for LBP
radius = 3
n_points = 8 * radius


# 读取图像
image = cv2.imread('ewan.jpg')


# 转换为灰度图显示
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('Gray')

# 处理
lbp = local_binary_pattern(image, n_points, radius)

plt.subplot(122)
plt.imshow(lbp, cmap='gray')
plt.axis('off')
plt.title('LBP')

plt.tight_layout()
plt.show()