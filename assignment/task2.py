import cv2
import numpy as np
from matplotlib import pyplot as plt

def pyramid_alignment(template, base, method=calc_SAD, offset=20):
template_back = template.copy()
offset_i = 0
offset_j = 0
history = []
for pyramid, percent in zip([8, 4, 2, 1], [0.75, 0.80, 0.85, 0.90]):
    img1 = cv2.resize(template, (template.shape[0] // pyramid, template.shape[1] // pyramid))
    img2 = cv2.resize(base, (base.shape[0] // pyramid, base.shape[1] // pyramid))
    i, j = alignment(img1, img2, method, 2, percent)
    offset_i = offset_i + i * pyramid
    offset_j = offset_j + j * pyramid
    template = np.zeros((template_back.shape[0] + 2 * offset + 1, template_back.shape[1] + 2 * offset + 1), "uint8")
    template[offset + offset_i:offset + offset_i + template_back.shape[0], offset + offset_j:offset + offset_j + template_back.shape[1]] = template_back
    template = template[offset:offset + template_back.shape[0], offset:offset + template_back.shape[1]]
    history.append((offset_i, offset_j))
return (offset_i, offset_j), history

img = cv2.imread(f"DataSamples/s1.jpg", cv2.IMREAD_GRAYSCALE)
height = int(np.ceil(img.shape[0] / 3))
img_b, img_g, img_r = img[0: height + 20, :], img[height - 10: 2 * height + 10, :], img[-(height + 20):, :]
shape = img_g.shape
plt.rcParams['figure.figsize'] = (40.0, 30.0)
plt.subplot(1, 3, 1)
plt.imshow(img_b, cmap=plt.cm.gray)
plt.subplot(1, 3, 2)
plt.imshow(img_g, cmap=plt.cm.gray)
plt.subplot(1, 3, 3)
plt.imshow(img_r, cmap=plt.cm.gray)

_, his_b_g = pyramid_alignment(img_b, img_g)
_, his_r_g = pyramid_alignment(img_r, img_g)
plt.rcParams['figure.figsize'] = (40.0, 160.0)
for i, (offset_b_g, offset_r_g) in enumerate(zip(his_b_g, his_r_g), start=1):
    img_bgr = np.zeros((img_g.shape[0] + 40, img_g.shape[1] + 40, 3), "uint8")
    img_bgr[20 + offset_b_g[0]: shape[0] + 20 + offset_b_g[0], 20 + offset_b_g[1]:shape[1] + 20 + offset_b_g[1], 0] = img_b
    img_bgr[20:shape[0] + 20, 20:shape[1] + 20, 1] = img_g
    img_bgr[20 + offset_r_g[0]: shape[0] + 20 + offset_r_g[0], 20 + offset_r_g[1]:shape[1] + 20 + offset_r_g[1], 2] = img_r
    print(offset_b_g, offset_b_g)
    plt.subplot(4, 1, i)
    plt.imshow(img_bgr)

plt.rcParams['figure.figsize'] = (40.0, 50.0)
for idx in [1, 2, 3, 4, 5]:
    img = cv2.imread(f"DataSamples/s{idx}.jpg", cv2.IMREAD_GRAYSCALE)
    height = int(np.ceil(img.shape[0] / 3))
    img_b, img_g, img_r = img[0: height + 20, :], img[height - 10: 2 * height + 10, :], img[-(height + 20):, :]
    shape = img_g.shape
    offset_b_g, _ = pyramid_alignment(img_b, img_g)
    offset_r_g, _ = pyramid_alignment(img_r, img_g)
    img_bgr = np.zeros((img_g.shape[0] + 40, img_g.shape[1] + 40, 3), "uint8")
    img_bgr[20 + offset_b_g[0]: shape[0] + 20 + offset_b_g[0], 20 + offset_b_g[1]:shape[1] + 20 + offset_b_g[1], 0] = img_b
    img_bgr[20:shape[0] + 20, 20:shape[1] + 20, 1] = img_g
    img_bgr[20 + offset_r_g[0]: shape[0] + 20 + offset_r_g[0], 20 + offset_r_g[1]:shape[1] + 20 + offset_r_g[1], 2] = img_r
    plt.subplot(5, 4, 4 * (idx - 1) + 1)
    plt.imshow(img_b, cmap=plt.cm.gray)
    plt.subplot(5, 4, 4 * (idx - 1) + 2)
    plt.imshow(img_g, cmap=plt.cm.gray)
    plt.subplot(5, 4, 4 * (idx - 1) + 3)
    plt.imshow(img_r, cmap=plt.cm.gray)
    plt.subplot(5, 4, 4 * (idx - 1) + 4)
    plt.imshow(img_bgr)
