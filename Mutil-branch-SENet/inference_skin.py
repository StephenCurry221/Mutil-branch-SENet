import argparse
import torch
from postProcess import proc
from metrics import *
from dataset_otherdata import Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.pylab as pltt
from model import model
from skimage.color import lab2rgb
from cls_postprocessing import *

# device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
meanStd = ((0.80508233, 0.80461432, 0.8043749), (0.14636562, 0.1467832,  0.14712358))


def parseArgs():

    parse = argparse.ArgumentParser()
    # parse.add_argument('--rootPth', type=str, default=Path(__file__).parent.parent / 'data1')
    parse.add_argument('--rootPth', type=str, default="/media/tiger/Disk0/jwang/tissue_seg_matlab/Area_seg/Area_Segmentation/WPATCH/train/PATCH_NORMAL_TRAIN/1503896 2018-07-30 16.19.37/988.png")
    # parse.add_argument('--modelPth', type=str, default='../model_cls/200527-104903/final.pth')
    parse.add_argument('--modelPth', type=str, default="/home/cliu/PycharmProjects/zzr_hovernet/model_cls/200903-220752/final.pth")

    return parse.parse_args()


def main(args):
    data = Data(root='/home/cliu/PycharmProjects/zzr_hovernet/skin',
                mode='test',
                isAugmentation=False)
    dataLoader = DataLoader(data, batch_size=4)

    net = model().to(device)
    net.load_state_dict(torch.load(args.modelPth, map_location='cpu'))
    with torch.no_grad():
        # for img, mask in dataLoader:
        for img, name in dataLoader:
            img = img.to(device)
            # mask = mask.squeeze()
            branchSeg, branchMSE, branchse = net(img)
            pred = torch.cat((branchSeg, branchMSE, branchse), dim=1)
            for i in range(pred.shape[0]):  # pred.shape[0]== batch_size
                output = pred[i]

                # process_instance(output, 6, remap_label=False, output_dtype='uint16')

                # output = proc(output)
                # plt.imshow(output, cmap='Blues')
                # plt.show()
                # (output, img[i, ...])
                se_cls = prob_cls(output)
                # compute_pixel_level_metrics(se_cls, mask)
                # plt.show()
                # se_probmap = output[3:9, ...]
                # se_result = np.argmax(se_probmap, axis=0)

                # output = output[3, ...]
                # root = '/home/cliu/PycharmProjects/zzr_hovernet/skin/test/Annotation_instance/14336_14336_3.npy'
                # mask = np.load(root)
                #
                # plt.imshow(mask)
                # plt.show()
                # metricPQ, _ = get_fast_pq(mask, output)
                # metricDice = get_dice_1(mask, output)
                # acc, iou, recall, precision, F1, performance = compute_pixel_level_metrics(output, mask)
                # region_re = cmp_region(output, mask)
                # print(f'Dice: {metricDice}, '
                #       f'DQ: {metricPQ[0]}, '
                #       f'SQ: {metricPQ[1]}, '
                #       f'PQ: {metricPQ[2]}')
                # cv2.imwrite('/home/cliu/PycharmProjects/zzl/pre{}.png'.format(i), output)
                # plt.imshow(output[0,...], cmap='jet')
                # pltt.show()
                plt.show()
                # plt.imsave(f"/home/cliu/PycharmProjects/data/train/Images/{name[i].replace('png', 'jpg')}", output, cmap='jet')
                print('this is {} slide'.format(i))


if __name__ == '__main__':
    args = parseArgs()
    main(args)
    # import cv2
    # cv2.findContours()
