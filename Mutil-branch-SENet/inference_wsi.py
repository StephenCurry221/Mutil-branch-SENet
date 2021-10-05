import numpy as np
import matplotlib.pylab as plt
import torch
import argparse
import os
import os.path as osp
from openslide import  OpenSlide, OpenSlideUnsupportedFormatError
import cv2
import math
from skimage.filters import threshold_mean
from skimage.morphology import reconstruction
from skimage.measure import regionprops, label
import tqdm
from PIL import Image
import torchvision.transforms as transforms
from model import model

# some global const
using_level = 1  # 20x
img_size = 1024
trainImageSize = 144
downsampling = 2**4
stride = img_size - trainImageSize + downsampling
# stride = int(img_size / 2)
# stride=500
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# args parameters
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataPath', type=str, default='/media/tiger/Disk0/jwang/tissue_seg_matlab/Area_seg/Area_Segmentation/Train/Train_MF/', help='the test data path')
    parse.add_argument('--modelPath', type=str, default='/home/cliu/PycharmProjects/zzr_hovernet/model_cls/200903-220752/final.pth', help='the test model parameter')
    return parse.parse_args()

# a class of wsi
class WSIPyramid:
    def __init__(self, path, stride=100):
        self.stride = stride
        self.slide, downsample_image, self.level, m, n = self.read_image(path)
        self.bounding_boxes, self.rgb_contour, self.image_dilation = \
            self.find_roi_bbox_1(downsample_image, show=False)

    def get_bbox(self, cont_img, rgb_image=None, show=False):
        contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_contour = None
        if rgb_image is not None:
            rgb_contour = rgb_image.copy()*9
            line_color = (255, 0, 0)  # blue color code
            cv2.drawContours(rgb_contour, contours, -1, line_color, 1)
            if show:
                plt.imshow(rgb_contour)
                plt.show()

        bounding_boxes = [cv2.boundingRect(c) for c in contours]

        return bounding_boxes, rgb_contour

    def find_roi_bbox(self, rgb_image):   # bgr
        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        thres = threshold_mean(hsv[..., 0])
        # fig, ax = try_all_threshold(hsv[..., 0])
        # plt.show()
        mask = (hsv[..., 0] > thres).astype('uint8')

        close_kernel = np.ones((5, 5), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((7, 7), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        image_fill = hole_fill(image_open)
        image_fill = max_prop(image_fill)
        bounding_boxes, rgb_contour = self.get_bbox(np.array(image_fill), rgb_image=rgb_image, show=False)
        return bounding_boxes, rgb_contour, image_fill


    def find_roi_bbox_1(self, rgb_image, show=True):     # rgb
        # hsv -> 3 channel
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        thres = threshold_mean(hsv[..., 0])
        # fig, ax = try_all_threshold(hsv[..., 0])
        # plt.show()
        mask = (hsv[..., 0] > thres).astype('uint8')

        # close_kernel = np.ones((11, 11), dtype=np.uint8)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 返回指定形状和尺寸得结构元素
        image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)  # 进行闭运算
        # open_kernel = np.ones((7, 7), dtype=np.uint8)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)  # 进行开运算
        image_open = cv2.morphologyEx(np.array(image_open),
                                      cv2.MORPH_DILATE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        image_fill = hole_fill(image_open)
        image_fill = max_prop(image_fill)
        bounding_boxes, rgb_contour = self.get_bbox(np.array(image_fill), rgb_image, show)

        return bounding_boxes, rgb_contour, image_fill

    def read_image(self, image_path):
        try:
            image = OpenSlide(image_path)
            w, h = image.dimensions
            n = int(math.floor((h - 0) / self.stride))
            m = int(math.floor((w - 0) / self.stride))
            level = image.level_count - 1
            if level > 7:
                level = 7
            downsample_image = np.array(image.read_region((0, 0), level, image.level_dimensions[level]))[..., 0:-1]
        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None, None, None, None, None

        return image, downsample_image, level, m, n

def hole_fill(img):  # 输入的是一个一个channel的类似与mask的东西
    """
    like the function of imfill in Matlab
    :param img:
    :return:
    """
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.max()
    mask = img
    return reconstruction(seed, mask, method='erosion').astype('uint8')

def max_prop(img):  # 说白了就是通过这个函数让一张图片里面标记区域最大得那个当作这张图片得label
    """
    select the max area
    :param img:
    :return:
    """
    label_, label_num = label(np.uint8(img), return_num=True)
    props = regionprops(label_)  # 找到每个标记的区域
    filled_area = []
    label_list = []
    for prop in props:
        filled_area.append(prop.area)
        label_list.append(prop.label)
    filled_area_sort = np.sort(filled_area)
    true_label = label_list[np.squeeze(np.argwhere(filled_area == filled_area_sort[-1]))]
    img = (label_ == true_label).astype('uint8')
    return img

def transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.70624471, 0.70608306, 0.70595071),
                                                    (0.12062634, 0.1206659, 0.12071837))])

# def get_wsi_part(wsi,xml,  args):




def read_wsi_and_show_result(wsi_paths=None, args=None):
    for wsi_path in wsi_paths:
        whole_path = osp.join(args.dataPath, wsi_path)
        wsi = WSIPyramid(whole_path)
        name = os.path.splitext(wsi_path)[0]
        result = inference(wsi=wsi, args=args)
        # np.save('./{}.npy', result)
        # plt.imsave('./{}.png'.format(name), result)
        plt.imshow(result)
        plt.show()
def inference(wsi=None, args=None):
    for bounding_box in wsi.bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2] + 1)
        b_y_end = int(bounding_box[1]) + int(bounding_box[3] + 1)
        mag_factor = 2 ** (wsi.level - using_level)
        col_cords = np.arange(int(b_x_start * mag_factor / stride), int(np.ceil(b_x_end * mag_factor / stride)))
        row_cords = np.arange(int(b_y_start * mag_factor / stride), int(np.ceil(b_y_end * mag_factor / stride)))

        net = model().to(device)
        net.load_state_dict(torch.load(args.modelPath, map_location='cpu'))

        with torch.no_grad():
            # tempOutput = net(torch.rand((4, 3, img_size, img_size)))
            [a, b, c] = net((torch.rand((4, 3, img_size, img_size)).cuda()).to(device))
            # tempOutput= c
            # tempOutput=1000
            # tempOutputShape = tempOutput.shape[-1]
            # tempOutputShape=1000

            size = c[2]
            tempOutputShape = size.shape[-1]
            tempOutputShape = int(tempOutputShape/4)
            print('OPTs output shape:{}'.format(tempOutputShape))
            result = np.zeros((len(row_cords) * tempOutputShape,
                              len(col_cords) * tempOutputShape))
            row, column = len(row_cords) * tempOutputShape, len(col_cords) * tempOutputShape
            for idx, i in tqdm.tqdm(enumerate(col_cords)):
                for jdx, j in enumerate(row_cords):
                    patchSizeX = min(img_size, wsi.slide.level_dimensions[using_level][0] - stride * i)
                    patcyhSizeY = min(img_size, wsi.slide.level_dimensions[using_level][1] - stride * j)
                    img_ = np.array(wsi.slide.read_region((stride * i * 2 ** using_level,  stride * j * 2 ** using_level), using_level, (patchSizeX, patcyhSizeY)))[..., 0:-1]
                    img = Image.fromarray(img_)
                    img = transform()(img).unsqueeze(0).to(device)
                    # test
                    # root = "/media/tiger/Disk0/jwang/skin/test/Images/14336_14336_2.png"
                    # img_ = np.array(Image.open(root))
                    # img = Image.fromarray(img_)
                    # img = transform()(img).unsqueeze(0).to(device)
                    [branchSeg, branchMSE, branchse] = net(img)
                    # output = branchse
                    pre = torch.cat((branchSeg, branchMSE, branchse), dim=1)
                    for i in range(pre.shape[0]):
                        res = pre[i]
                        cls = res[3:, ...]
                        cls = torch.softmax(cls, dim=0)  # 0-1
                        cls = cls.cuda().data.cpu().numpy()
                        cls = np.argmax(cls, axis=0)
                        cls = np.resize(cls, (256, 256))

                        # cls_nu = cls.cuda().data.cpu().numpy()
                        # ff = cls_nu.transpose(1, 2, 0)
                        # ff = np.argmax(ff)
                        # ff = cls_nu[..., 3:]
                        # plt.imshow(ff)
                        # plt.show()


                        # a = output[0, 3:, ...]
                        # # b = a.transpose(1, 2, 0)
                        # b = a.cuda().data.cpu().numpy()
                        # c = b.transpose(1, 2, 0)


                        # if b.shape[0] != tempOutputShape:
                        #     row = jdx * tempOutputShape + b.shape[0]
                        # if b.shape[1] != tempOutputShape:
                        #     column = idx * tempOutputShape + output.shape[1]
                        result[jdx * tempOutputShape: (jdx+1) * tempOutputShape,
                        idx * tempOutputShape: (idx+1) * tempOutputShape] = cls
            result = result[0:row, 0:column, ...]
        return result

def main(args=None):
    path_= os.listdir(args.dataPath)
    wsi_paths=[]
    for path in path_:
        name = osp.splitext(path)[-1]
        if name=='.ndpi':
            wsi_paths.append(path)
    read_wsi_and_show_result(wsi_paths=wsi_paths, args=args)



if __name__ == '__main__':
    args = parse_args()
    main(args)
