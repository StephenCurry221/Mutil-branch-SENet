import cv2
import numpy as np
from skimage.morphology import remove_small_objects
import matplotlib.pylab as plt
"""
A rudimentary URL downloader (like wget or curl) to demonstrate Rich progress bars.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os.path
import sys
from typing import Iterable
from urllib.request import urlopen

from rich.progress import (
    BarColumn,
    DownloadColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    Progress,
    TaskID,
)


progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "",
    DownloadColumn(),
    "",
    TransferSpeedColumn(),
    "",
    TimeRemainingColumn(),
)


def copy_url(task_id: TaskID, url: str, path: str) -> None:
    """Copy data from a url to a local file."""
    response = urlopen(url)
    # This will break if the response doesn't contain content length
    progress.update(task_id, total=int(response.info()["Content-length"]))
    with open(path, "wb") as dest_file:
        progress.start_task(task_id)
        for data in iter(partial(response.read, 32768), b""):
            dest_file.write(data)
            progress.update(task_id, advance=len(data))


def download(urls: Iterable[str], dest_dir: str):
    """Download multuple files to the given directory."""
    with progress:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for url in urls:
                filename = url.split("/")[-1]
                dest_path = os.path.join(dest_dir, filename)
                task_id = progress.add_task("download", filename=filename, start=False)
                pool.submit(copy_url, task_id, url, dest_path)


if __name__ == "__main__":
    # Try with https://releases.ubuntu.com/20.04/ubuntu-20.04-desktop-amd64.iso
    if sys.argv[1:]:
        download(sys.argv[1:], "./")
    else:
        print("Usage:\n\tpython downloader.py URL1 URL2 URL3 (etc)")

'''
def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# single preprocessing
img1 = cv2.imread('./1.png', 0)
img2 = cv2.imread('./2.png', 0)
# print(img1.shape)

img3 = img2 - img1

img3 = remove_small_objects(img3.astype(np.bool), min_size=100)

# plt.imshow(img3)
# plt.show()
cv2.imshow('img', img3)
show('img3', img3 * 255)

show('img1', img1)
show('img2', img2)

print(img1.shape)
print(img2.shape)


'''

import cv2
import numpy as np

root = './ppt1.png'
root1 = './fig'


def FillHole_RG(imgPath, SavePath, SizeThreshold):
    # 读取图像为uint32,之所以选择uint32是因为下面转为0xbbggrr不溢出
    im_in_rgb = cv2.imread(imgPath).astype(np.uint32)

    # 将im_in_rgb的RGB颜色转换为 0xbbggrr  16jinzhi
    im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (im_in_rgb[:, :, 2] << 16)

    # 将0xbbggrr颜色转换为0,1,2,...
    colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)

    # 将im_in_lbl_new数组reshape为2维
    im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)

    # 创建从32位im_in_lbl_new到8位colorize颜色的映射
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 输出一下colorize中的color
    print("Colors_RGB: \n", colorize)

    # 有几种颜色就设置几层数组，每层数组均为各种颜色的二值化数组
    im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)

    # 初始化二值数组
    im_th = np.zeros(im_in_lbl_new.shape, np.uint8)  # 1000, 1000

    for i in range(len(colors)):
        for j in range(im_th.shape[0]):
            for k in range(im_th.shape[1]):
                if (im_in_lbl_new[j][k] == i):
                    im_th[j][k] = 255
                else:
                    im_th[j][k] = 0

        # 复制 im_in 图像
        im_floodfill = im_th.copy()

        # Mask 用于 floodFill,mask多出来的2可以保证扫描的边界上的像素都会被处理.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        isbreak = False
        for m in range(im_floodfill.shape[0]):
            for n in range(im_floodfill.shape[1]):
                if (im_floodfill[m][n] == 0):
                    seedPoint = (m, n)
                    isbreak = True
                    break
            if (isbreak):
                break
        # 得到im_floodfill
        cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)

        # 得到im_floodfill的逆im_floodfill_inv，im_floodfill_inv包含所有孔洞
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # 之所以复制一份im_floodfill_inv是因为函数findContours会改变im_floodfill_inv_copy
        im_floodfill_inv_copy = im_floodfill_inv.copy()
        # 函数findContours获取轮廓
        contours, hierarchy = cv2.findContours(im_floodfill_inv_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for num in range(len(contours)):
            if (cv2.contourArea(contours[num]) >= SizeThreshold):
                cv2.fillConvexPoly(im_floodfill_inv, contours[num], 0)

        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out = im_th | im_floodfill_inv
        im_result[i] = im_out

    # rgb结果图像
    im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3), np.uint8)

    # 之前的颜色映射起到了作用
    for i in range(im_result.shape[1]):
        for j in range(im_result.shape[2]):
            for k in range(im_result.shape[0]):
                if (im_result[k][i][j] == 255):
                    im_fillhole[i][j] = colorize[k]
                    break

    # 保存图像
    show('img', im_fillhole)
    # cv2.imwrite(SavePath, im_fillhole)

def FillHole_RGB(imgPath, SavePath):
    # 读取图像为uint32,之所以选择uint32是因为下面转为0xbbggrr不溢出
    im_in_rgb = cv2.imread(imgPath).astype(np.uint32)

    # 将im_in_rgb的RGB颜色转换为 0xbbggrr
    im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (im_in_rgb[:, :, 2] << 16)

    # 将0xbbggrr颜色转换为0,1,2,...
    colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)

    # 将im_in_lbl_new数组reshape为2维
    im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)

    # 创建从32位im_in_lbl_new到8位colorize颜色的映射
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 输出一下colorize中的color
    print("Colors_RGB: \n", colorize)

    # 有几种颜色就设置几层数组，每层数组均为各种颜色的二值化数组
    im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)

    # 初始化二值数组
    im_th = np.zeros(im_in_lbl_new.shape, np.uint8)

    for i in range(len(colors)):
        for j in range(im_th.shape[0]):
            for k in range(im_th.shape[1]):
                if (im_in_lbl_new[j][k] == i):
                    im_th[j][k] = 255
                else:
                    im_th[j][k] = 0
        # 复制 im_in 图像
        im_floodfill = im_th.copy()

        # Mask 用于 floodFill,mask多出来的2可以保证扫描的边界上的像素都会被处理.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        isbreak = False
        for m in range(im_floodfill.shape[0]):
            for n in range(im_floodfill.shape[1]):
                if (im_floodfill[m][n] == 0):
                    seedPoint = (m, n)
                    isbreak = True
                    break
            if (isbreak):
                break
        # 得到im_floodfill
        cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)

        #        help(cv2.floodFill.rect)
        # 得到im_floodfill的逆im_floodfill_inv
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
        im_out = im_th | im_floodfill_inv
        im_result[i] = im_out

    # rgb结果图像
    im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3), np.uint8)

    # 之前的颜色映射起到了作用
    for i in range(im_result.shape[1]):
        for j in range(im_result.shape[2]):
            for k in range(im_result.shape[0]):
                if (im_result[k][i][j] == 255):
                    im_fillhole[i][j] = colorize[k]
                    break

    # 保存图像
    show('img', im_fillhole)
    # cv2.imwrite(SavePath, im_fillhole)

# FillHole_RG(root, root1, 2000)

'''
'''