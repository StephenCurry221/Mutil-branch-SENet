import numpy as np
import cv2
import os
import matplotlib.pylab as plt

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


root = './CoNSeP/train/Overlay'
root1 = '/home/cliu/PycharmProjects/data/train/Images'
original_label_list = os.listdir(root)
label = []
for i in range(27):
    img_dir = original_label_list[i]
    ovlay = cv2.imread(os.path.join(root, img_dir))
    image = cv2.imread(os.path.join(root1, img_dir))
    image_label = np.zeros((1000, 1000, 3))

    image_label[(ovlay[:, :, 0] == 0) & (ovlay[:, :, 1] == 0) & (ovlay[:, :, 2] == 255)] = (0, 0, 255)  # red
    # print(image_label.max())
    # cnts, _ = cv2.findContours(image_label.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cls_label = np.zeros((1000, 1000, 3))
    # cv2.drawContours(cls_label, cnts, -1, (0, 0, 255), -1)
    # cv_show('cls_label', cls_label)


    image_label[(ovlay[:, :, 0] == 0) & (ovlay[:, :, 1] == 255) & (ovlay[:, :, 2] == 255)] = (0, 255, 255)  # yellow

    image_label[(ovlay[:, :, 0] == 255) & (ovlay[:, :, 1] == 0) & (ovlay[:, :, 2] == 255)] = (255, 0 , 255)  # pink
    image_label[(ovlay[:, :, 0] == 255) & (ovlay[:, :, 1] == 0) & (ovlay[:, :, 2] == 0)] = (255, 0 , 0)  # blue
    image_label[(ovlay[:, :, 0] == 0) & (ovlay[:, :, 1] == 255) & (ovlay[:, :, 2] == 0)] = (0, 255, 0)  # green
    image_label[(ovlay[:, :, 0] == 66) & (ovlay[:, :, 1] == 158) & (ovlay[:, :, 2] == 244)] = (66, 158, 244)  # gray yellow
    image_label[(ovlay[:, :, 0] == 255) & (ovlay[:, :, 1] == 255) & (ovlay[:, :, 2] == 0)] = (255, 255, 0)  # lake blue

    cv_show('image_label', image_label)
