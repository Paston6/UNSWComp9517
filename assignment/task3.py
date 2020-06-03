import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (40.0, 50.0)
img = cv2.imread(f"DataSamples/s4.jpg", cv2.IMREAD_GRAYSCALE)
height = int(np.ceil(img.shape[0] / 3))
img_b, img_g, img_r = img[0: height + 20, :], img[height - 10: 2 * height + 10, :], img[-(height + 20):, :]
shape = img_g.shape
offset_b_g, _ = pyramid_alignment(img_b, img_g)
offset_r_g, _ = pyramid_alignment(img_r, img_g)
img_bgr = np.zeros((img_g.shape[0] + 40, img_g.shape[1] + 40, 3), "uint8")
img_bgr[20 + offset_b_g[0]: shape[0] + 20 + offset_b_g[0], 20 + offset_b_g[1]:shape[1] + 20 + offset_b_g[1], 0] = img_b
img_bgr[20:shape[0] + 20, 20:shape[1] + 20, 1] = img_g
img_bgr[20 + offset_r_g[0]: shape[0] + 20 + offset_r_g[0], 20 + offset_r_g[1]:shape[1] + 20 + offset_r_g[1], 2] = img_r
# origin
plt.subplot(1, 3, 1)
plt.imshow(img_bgr)
# after Histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_b = clahe.apply(img_b)
img_g = clahe.apply(img_g)
img_r = clahe.apply(img_r)
img_bgr[20 + offset_b_g[0]: shape[0] + 20 + offset_b_g[0], 20 + offset_b_g[1]:shape[1] + 20 + offset_b_g[1], 0] = img_b
img_bgr[20:shape[0] + 20, 20:shape[1] + 20, 1] = img_g
img_bgr[20 + offset_r_g[0]: shape[0] + 20 + offset_r_g[0], 20 + offset_r_g[1]:shape[1] + 20 + offset_r_g[1], 2] = img_r
plt.subplot(1, 3, 2)
plt.imshow(img_bgr)
# after automatically removing borders
img_bgr = img_bgr[
    max(20 + offset_b_g[0], 20, 20 + offset_r_g[0]):min(shape[0] + 20 + offset_b_g[0], shape[0] + 20, shape[0] + 20 + offset_r_g[0]),
    max(20 + offset_b_g[1], 20, 20 + offset_r_g[1]):min(shape[1] + 20 + offset_b_g[1], shape[1] + 20, shape[1] + 20 + offset_r_g[1]),
    :
]
plt.subplot(1, 3, 3)
plt.imshow(img_bgr)
