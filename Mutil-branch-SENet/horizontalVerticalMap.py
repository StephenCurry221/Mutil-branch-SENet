# coding=utf-8
"""
@File   : horizontalVerticalMap.py
@Time   : 2020/01/03
@Author : Zengrui Zhao
"""
import time
import numpy as np
import argparse
from pathlib import Path
import os
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from tqdm import tqdm
import torch
import torch.nn.functional as tf

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', type=str, default='/media/tiger/Disk0/jwang/skin/test/Annotation_instance')
    parse.add_argument('--savePth', type=str, default='/media/tiger/Disk0/jwang/skin/test/HorizontalVerticalMap')

    return parse.parse_args()

def mapGenarating(props, img, mode='v'):
    result = np.zeros_like(img, dtype='float')
    index = 0 if mode == 'v' else 1
    for prop in props:
        minIndex = np.min(np.transpose(prop['coords'])[index])
        maxIndex = np.max(np.transpose(prop['coords'])[index])
        for coord in prop['coords']:
            if coord[index] < prop['centroid'][index]:
                result[coord[0], coord[1]] = (coord[index] - minIndex) / \
                                             (prop['centroid'][index] - minIndex) - 1

            elif coord[index] > prop['centroid'][index]:
                result[coord[0], coord[1]] = (coord[index] - prop['centroid'][index]) / \
                                             (maxIndex - prop['centroid'][index])

    return result

def main(args):
    labels = sorted([i for i in os.listdir(args.path) if i.endswith('npy')])
    start = time.time()
    for i in tqdm(labels):
        img = np.load(Path(args.path) / i).squeeze().astype(np.uint32)
        props = regionprops(img)
        horizontalMap = mapGenarating(props, img, mode='h')
        verticalMap = mapGenarating(props, img, mode='v')
        if not Path(args.savePth).exists():
            os.makedirs(Path(args.savePth))

        # np.save(Path(args.savePth) / (i.split('.')[0] + '_horizontal.npy'), horizontalMap)
        # np.save(Path(args.savePth) / (i.split('.')[0] + '_vertical.npy'), verticalMap)
        # getGradient(horizontalMap, show=True)
        plt.imshow(horizontalMap, cmap='jet')
        plt.show()
        plt.imshow(verticalMap, cmap='jet')
        plt.show()
        # break
    print(f'Done! cost time: {time.time() - start}')

def getGradient(input, show=True):
    def getSobelKernel(size):
        hvRange = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
        h, v = np.meshgrid(hvRange, hvRange)
        kernelH = h / (h * h + v * v + 1.0e-15)
        kernelV = v / (h * h + v * v + 1.0e-15)

        return kernelH, kernelV

    kernelSize = 5
    mh, mv = getSobelKernel(kernelSize)
    mh = np.reshape(mh, [1, 1, kernelSize, kernelSize])
    mv = np.reshape(mv, [1, 1, kernelSize, kernelSize])
    if type(input) is torch.Tensor:
        input = input.cpu().detach().numpy()
        h, v = input[:, 1, ...][:, None, ...], input[:, 0, ...][:, None, ...]
    else:
        h, v = input[1, ...][None, None, ...], input[0, ...][None, None, ...]

    dh = tf.conv2d(torch.tensor(h, dtype=torch.double), torch.tensor(mh, dtype=torch.double), stride=1, padding=2)
    dv = tf.conv2d(torch.tensor(v, dtype=torch.double), torch.tensor(mv, dtype=torch.double), stride=1, padding=2)
    if show:
        size = 200
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(-np.squeeze(np.array(dh))[0:size, 0:size], cmap='jet')
        axes[0, 1].imshow(np.squeeze(np.array(h))[0:size, 0:size], cmap='jet')
        axes[1, 0].imshow(-np.squeeze(np.array(dv))[0:size, 0:size], cmap='jet')
        axes[1, 1].imshow(np.squeeze(np.array(v))[0:size, 0:size], cmap='jet')
        axes[0, 0].set_title('Gradient', fontsize=20)
        axes[0, 0].set_ylabel('Horizontal', fontsize=20)
        axes[0, 1].set_title('Raw', fontsize=20)
        axes[1, 0].set_ylabel('Vertical', fontsize=20)
        plt.show()
        return
    else:
        return torch.cat((dv, dh), dim=1)

if __name__ == '__main__':
    args = parseArgs()
    main(args)

