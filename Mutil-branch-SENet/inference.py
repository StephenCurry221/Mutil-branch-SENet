# coding=utf-8
"""
@File   : inference.py
@Time   : 2020/01/10
@Author : Zengrui Zhao
"""
import argparse
import torch
from postProcess import proc
from metrics import *
from dataset import Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import model
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_device((1))
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
meanStd = ((0.80508233, 0.80461432, 0.8043749), (0.14636562, 0.1467832,  0.14712358))

def parseArgs():
    parse = argparse.ArgumentParser()
    # parse.add_argument('--rootPth', type=str, default=Path(__file__).parent.parent / 'data1')
    parse.add_argument('--rootPth', type=str, default="/media/tiger/Disk0/jwang/test_jn")
    parse.add_argument('--modelPth', type=str, default="/home/cliu/PycharmProjects/model/200527-104903/final.pth")

    return parse.parse_args()

def main(args):
    data = Data(root="/media/tiger/Disk0/jwang/test_jn/",
                mode='test',
                isAugmentation=False)
    dataLoader = DataLoader(data, batch_size=1)
    # model = nn.DataParallel(model())
    net = model().to(device)
    # net = nn.DataParallel(net, device_ids=[1, 2, 3])
    # net.load_state_dict(torch.load(args.modelPth), False)
    # net = nn.DataParallel(net, device_ids=[0, 1, 2])
    net.load_state_dict(torch.load(args.modelPth, map_location='cpu'))
    # net = nn.DataParallel(net)
    net.eval()
    with torch.no_grad():
        # for img, mask in dataLoader:
        for img in dataLoader:
            original_img = img[0]
            a1 = original_img[:, :, 0:2048, 0:2048]
            a2 = original_img[:, :, :2048, 2048:4096]
            # img = img[0].to(device)
            a1 = a1.to(device)
            a2 = a2.to(device)
            # mask = mask.squeeze()
            branchSeg, branchMSE = net(a1)
            pred = torch.cat((branchSeg, branchMSE), dim=1)
            for i in range(pred.shape[0]):
                output = proc(pred[i])

                # metricPQ, _ = get_fast_pq(mask, output)
                # metricDice = get_dice_1(mask, output)
                # print(f'Dice: {metricDice}, '
                #       f'DQ: {metricPQ[0]}, '
                #       f'SQ: {metricPQ[1]}, '
                #       f'PQ: {metricPQ[2]}')
                # cv2.imwrite('/home/cliu/PycharmProjects/CGC-Net/data/proto/mask/fold_1/Grade1/{}'.format(img[1]), output)
                plt.imsave('/media/tiger/Disk0/jwang/test_jn_res/{}.png'.format(i), output, cmap='jet')
                # plt.imshow(output, cmap='jet')
                # plt.show()
                # print('debug')


if __name__ == '__main__':
    args = parseArgs()
    main(args)
    # import cv2
    # cv2.findContours()