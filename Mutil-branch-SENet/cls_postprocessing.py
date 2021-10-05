import numpy as np
import matplotlib.pylab as plt
import cv2
import torch
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from skimage.morphology import remove_small_objects, watershed, remove_small_holes
import os
def compute_pixel_level_metrics(pred, target):
    '''
    compute the pixel-level tp, fp, tn, fn between predicted img and groundtruth target
    :param pred:
    :param target:
    :return: [acc, iou, recll, precision, F1, perfermance]
    '''
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)
    # tp_sum=[]
    # for i, j in pred:
    #     if pred[i, j]==target[i, j]:
    #         tp_sum.append()
    # tp = len(tp_sum)
    tp = np.sum(pred * target)
    tn = np.sum(np.abs((1-pred) * (1- target)))
    fp = np.sum(pred * np.abs((1 - target)))
    fn = np.sum(np.abs((1-pred) * target))
    precision = tp/(tp + fp + 1e-10)
    recall = tp/(tp + fn + 1e-10)
    F1 = 2 * precision * recall/(precision + recall + 1e-10)
    acc = (tp + tn)/(tp + fp + tn + fn + 1e-10)
    performance = (recall + tn/(tn + fp + 1e-10))
    iou = tp/(tp + fp + fn + 1e-10)
    return [acc, iou, recall, precision, F1, performance]

def prob_cls(output, img_name):
    # cls = output[3:, ...] # SEnet
    # cls = output[1:, ...]  # Abalation_hv
    cls = output[2:, ...]  # Abalation_seg
    a, b = os.path.splitext(img_name)
    # cls = output #Abalation_hv_seg

    cls = torch.softmax(cls, dim=0)  # 0-1
    cls = np.argmax(cls, axis=0)  # choose the max probability

    # draw a contours
    # a = cls.cpu().detach().numpy()
    # b = a
    # b[b[...]==2]=1
    # contours, hier = cv2.findContours(b, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(contours, -1, (0, 0, 255), 5)
    # pred_cls = np.zeros((output.shape[1], output.shape[2], 3))
    # pred_cls = np.zeros((200, 200, 3))
    # cls = output

    # pred_cls[cls==1] = 255
    # prob_cls[a, b, :] = (0, 0, 0)
    # prob_cls[c, d,:] = (0, 0, 255)
    '''
     pred_cls[cls == 0] = (0, 0, 0)
    # elif cls == 1:
    pred_cls[cls == 1] = (255, 0, 0)
    # elif cls == 2:
    pred_cls[cls == 2] = (0, 255, 0)
    # elif cls == 3:
    pred_cls[cls == 3] = (0, 0, 255)
    # elif cls == 4:
    pred_cls[cls == 4] = (255, 255, 0)
    # elif cls == 5:
    pred_cls[cls == 5] = (0, 255, 255)

    pred_cls[cls == 6] = (255, 0, 255)
    pred_cls[cls == 7] = (244, 158, 66)
    '''
    plt.imsave('/media/tiger/Disk0/jwang/skin/cls_res_seg/train/{}.jpg'.format(a), cls, cmap='RdBu_r')
    # plt.imshow(cls, cmap='RdBu_r')
    # plt.show()
    return cls
# a = np.ones((200, 200))
# b = prob_cls(a)

# plt.imshow(b)
# plt.show()
# cv2.imshow('img', b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def get_instance_img(instance, img):
    cnts, _ = cv2.findContours(instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image = img.copy()
    cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
    return image


def proc_np_hv(pred, return_coords=False):
    """
    Process Nuclei Prediction with XY Coordinate Map
    Args:
        pred:           prediction output, assuming
                        channel 0 contain probability map of nuclei
                        channel 1 containing the regressed X-map
                        channel 2 containing the regressed Y-map
        return_coords: return coordinates of extracted instances
    """

    blb_raw = pred[0, ...]
    h_dir_raw = pred[1,...]
    v_dir_raw = pred[2,...]

    # Processing
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb < 0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already
    #####

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    h_dir_raw = None  # clear variable
    v_dir_raw = None  # clear variable

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    h_dir = None  # clear variable
    v_dir = None  # clear variable

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    overall = np.maximum(sobelh, sobelv)
    sobelh = None  # clear variable
    sobelv = None  # clear variable
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # nuclei values form peaks so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall[overall >= 0.5] = 1
    overall[overall < 0.5] = 0
    marker = blb - overall
    overall = None  # clear variable
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    pred_inst = watershed(dist, marker, mask=blb, watershed_line=False)
    if return_coords:
        label_idx = np.unique(pred_inst)
        coords = measurements.center_of_mass(blb, pred_inst, label_idx[1:])
        return pred_inst, coords
    else:
        return pred_inst

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger instances has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

def process_instance(pred_map, nr_types, remap_label=False, output_dtype='uint16'):
    """
    Post processing script for image tiles
    Args:
        pred_map: commbined output of nc, np and hv branches
        nr_types: number of types considered at output of nc branch
        remap_label: whether to map instance labels from 1 to N (N = number of nuclei)
        output_dtype: data type of output

    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction
    """

    pred_inst = pred_map[..., nr_types:]
    pred_type = pred_map[..., :nr_types]

    pred_inst = np.squeeze(pred_inst)
    pred_type = np.argmax(pred_type, axis=-1)
    pred_type = np.squeeze(pred_type)

    pred_inst = proc_np_hv(pred_inst)

    # remap label is very slow - only uncomment if necessary to map labels in order
    if remap_label:
        pred_inst = remap_label(pred_inst, by_size=True)

    pred_type_out = np.zeros([pred_type.shape[0], pred_type.shape[1]])
    #### * Get class of each instance id, stored at index id-1
    pred_id_list = list(np.unique(pred_inst))[1:]  # exclude background
    pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
    for idx, inst_id in enumerate(pred_id_list):
        inst_tmp = pred_inst == inst_id
        inst_type = pred_type[pred_inst == inst_id]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        pred_type_out += (inst_tmp * inst_type)
    pred_type_out = pred_type_out.astype(output_dtype)

    pred_inst = pred_inst.astype(output_dtype)

    return pred_inst, pred_type_out



