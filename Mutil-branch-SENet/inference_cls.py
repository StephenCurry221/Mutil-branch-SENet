import argparse
import torch
from postProcess import proc
from metrics import *
from dataset import Data
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
    parse.add_argument('--rootPth', type=str, default="/media/tiger/Disk0/jwang/skin/")
    # parse.add_argument('--modelPth', type=str, default='../model_cls/200527-104903/final.pth')
    parse.add_argument('--modelPth', type=str, default="./model_cls/200903-220752/final.pth")

    return parse.parse_args()


def main(args):
    data = Data(root=os.path.join(args.rootPth, 'test/Images/'),
                mode='test',
                isAugmentation=False)
    dataLoader = DataLoader(data, batch_size=4)

    net = model().to(device)
    net.load_state_dict(torch.load(args.modelPth, map_location='cpu'))
    with torch.no_grad():
        # for img, mask in dataLoader:
        for img, name in dataLoader:
            img_ = img.to(device)
            # mask = mask.squeeze()
            branchSeg, branchMSE, branchse = net(img_)
            pred = torch.cat((branchSeg, branchMSE, branchse), dim=1)
            for i in range(pred.shape[0]):  # pred.shape[0]== batch_size
                output = pred[i]
                img_name = name[i]
                # process_instance(output, 7, remap_label=False, output_dtype='uint16')

                # instance_img = proc(output)
                # (output, img[i, ...])
                se_cls = prob_cls(output, img_name)
                # plt.show()
                # se_probmap = output[3:9, ...]
                # se_result = np.argmax(se_probmap, axis=0)

                # output = output[3, ...]
                # metricPQ, _ = get_fast_pq(mask, output)
                # metricDice = get_dice_1(mask, output)
                # print(f'Dice: {metricDice}, '
                #       f'DQ: {metricPQ[0]}, '
                #       f'SQ: {metricPQ[1]}, '
                #       f'PQ: {metricPQ[2]}')
                # cv2.imwrite('/home/cliu/PycharmProjects/zzl/pre{}.png'.format(i), output)

                # plt.imshow(output, cmap='jet')
                # # pltt.show()
                # plt.show()
                # plt.imsave(f"/home/cliu/PycharmProjects/data/train/Images/{name[i].replace('png', 'jpg')}", output, cmap='jet')
                print('this is {} slide'.format(i))


if __name__ == '__main__':
    args = parseArgs()
    main(args)
    # import cv2
    # cv2.findContours()