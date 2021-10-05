import numpy as np
import cv2
import os
import matplotlib.pylab as plt

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cls = {'Malignant epithelium': [255, 0, 0],
         'Other': [255, 255, 0],
         'Inflammatory': [255, 0, 255],
         'Fibroblast': [0, 0, 255],
         'Normal epithelium': [0, 255, 0],
         'Endothelial':[244, 158, 66],
         'Muscle': [0, 255, 255]}

root = './CoNSeP/train/Overlay'
root1 = '/home/cliu/PycharmProjects/data/train/Images'
original_label_list = os.listdir(root)
# print(original_label_list)
label = []
for i in range(27):
    img_dir = original_label_list[i]
    image = cv2.imread(os.path.join(root, img_dir))
    image1 = cv2.imread(os.path.join(root1, img_dir))
    image2 = image1.copy()
    image_label = np.zeros((1000, 1000))
    image_label[(image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255)] = 255  # red


    # cv2.erode(image_label, (5, 5), iterations=20)
    # cv2.imwrite('1.png', image_label)
    # cv_show('22', image_label)
    contours, _ = cv2.findContours(image_label.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cls_label = np.zeros((1000, 1000, 3))
    # print(contours)
    cv2.drawContours(cls_label, contours, -1, (255, 255, 255), -1)
    # cv2.imwrite('2.png', cls_label)
    # cv2.imwrite('un_dilate.png', cls_label)
    cv_show('image1', cls_label)
    plt.imshow(cls_label)
    plt.show()
'''
    image_label[(image[:, :, 0] == 0) & (image[:, :, 1] == 255) & (image[:, :, 2] == 255)] = 255  # yellow
    image_label[(image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255)] = 255  # pink
    image_label[(image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)] = 255  # blue
    image_label[(image[:, :, 0] == 0) & (image[:, :, 1] == 255) & (image[:, :, 2] == 0)] = 255  # green
    image_label[(image[:, :, 0] == 66) & (image[:, :, 1] == 158) & (image[:, :, 2] == 244)] = 255  # gray yellow
    image_label[(image[:, :, 0] == 255) & (image[:, :, 1] == 255) & (image[:, :, 2] == 0)] = 255  # lake blue
'''









    # print(img)

