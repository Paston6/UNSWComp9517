import cv2
import numpy as np
from matplotlib import pyplot as plt

# Three suitable metrics
def calc_SSD(img1, img2):
    assert img1.shape == img2.shape
    return -np.sum(np.power(img1 - img2, 2))


def calc_SAD(img1, img2):
    assert img1.shape == img2.shape
    return -np.sum(np.abs(img1 - img2))


def calc_NCC(img1, img2):
    assert img1.shape == img2.shape
    return np.sum(img1 * img2) / np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))

# open DataSamples/s1.jpg and display
img = cv2.imread("DataSamples/s1.jpg", cv2.IMREAD_GRAYSCALE)
plt.rcParams['figure.figsize'] = (20.0, 15.0)
plt.imshow(img, cmap=plt.cm.gray)

# The process of aligning pictures
def alignment(template, base, method=calc_SAD, offset_range=20, center_percent=0.9):
    template = template / 256.
    base = base / 256.
    assert template.shape[0] <= base.shape[0] and template.shape[1] <= base.shape[1]
    offset_i = 0
    offset_j = 0
    score_max = -np.inf
    offset_border = {
        "U": -offset_range,
        "D": base.shape[0] - template.shape[0] + offset_range,
        "L": -offset_range,
        "R": base.shape[1] - template.shape[1] + offset_range,
    }
    sample_size = {
        "H": int((template.shape[0] - offset_range * 2) * center_percent),
        "W": int((template.shape[1] - offset_range * 2) * center_percent),
    }
    sample_board_template = {
        "U": (template.shape[0] - sample_size["H"]) // 2,
        "D": (template.shape[0] - sample_size["H"]) // 2 + sample_size["H"],
        "L": (template.shape[1] - sample_size["W"]) // 2,
        "R": (template.shape[1] - sample_size["W"]) // 2 + sample_size["W"],
    }
    sample_template = template[sample_board_template["U"]:sample_board_template["D"], sample_board_template["L"]:sample_board_template["R"]]
    for i in range(offset_border["U"], offset_border["D"] + 1):
        for j in range(offset_border["L"], offset_border["R"] + 1):
            sample_board_base = {
                "U": sample_board_template["U"] + i,
                "D": sample_board_template["D"] + i,
                "L": sample_board_template["L"] + j,
                "R": sample_board_template["R"] + j,
            }
            sample_base = base[sample_board_base["U"]:sample_board_base["D"], sample_board_base["L"]:sample_board_base["R"]]
            score = method(sample_template, sample_base)
            if score > score_max:
                score_max = score
                offset_i = i
                offset_j = j
    return offset_i, offset_j

# Divide the image into three parts, and add 20px to each part to ensure that the border is included. Based on the green picture (the one in the middle)
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

plt.rcParams['figure.figsize'] = (40.0, 50.0)
for idx in [1, 2, 3, 4, 5]:
    img = cv2.imread(f"DataSamples/s{idx}.jpg", cv2.IMREAD_GRAYSCALE)
    height = int(np.ceil(img.shape[0] / 3))
    img_b, img_g, img_r = img[0: height + 20, :], img[height - 10: 2 * height + 10, :], img[-(height + 20):, :]
    shape = img_g.shape
    offset_b_g = alignment(img_b, img_g)
    offset_r_g = alignment(img_r, img_g)
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
