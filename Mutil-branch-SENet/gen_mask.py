import cv2
import numpy as np
from collections import OrderedDict
import os
from glob import glob
from tqdm import tqdm


def gen_mask(color_dict, img_path, save_path):
    image_label = np.zeros((1000, 1000, 3), dtype=np.uint8)
    ovlay = cv2.imread(img_path, -1)
    for key, color in color_dict.items():
        x, y = np.where((ovlay[:, :, 0] == color[2]) & (ovlay[:, :, 1] == color[1]) & (ovlay[:, :, 2] == color[0]))
        mask_temp = np.zeros((1000, 1000), dtype=np.uint8)
        if len(x) == 0:
            print(f"==>No {key} found in {img_path}")
            continue
        else:
            mask_temp[(ovlay[:, :, 0] == color[2]) & (ovlay[:, :, 1] == color[1]) & (ovlay[:, :, 2] == color[0])] = 255
            contours_inner = fill_hole(mask_temp)
            cv2.drawContours(image_label, contours_inner, -1, color, -1)
    cv2.imshow('img', image_label)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(save_path, image_label)
            

def fill_hole(contour_mask):
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_inner = [contours[i] for i in range(0, len(contours), 2)]
    
    return contours_inner


def main():
    color_dict = OrderedDict({'Malignant epithelium': (255, 0, 0),
                              'Other': (255, 255, 0),
                              'Inflammatory': (255, 0, 255),
                              'Fibroblast': (0, 0, 255),
                              'Normal epithelium': (0, 255, 0),
                              'Endothelial': (244, 158, 66),
                              'Muscle': (0, 255, 255)})
    root = '/home/cliu/PycharmProjects/zzr_hovernet/CoNSeP/train/Overlay'
    img_list = glob(os.path.join(root, '*.png'))
    for img in tqdm([img_list[5]]):
        save_path = img.replace('.png', '_mask.png')
        gen_mask(color_dict, img, save_path)


if __name__ == '__main__':
    main()

