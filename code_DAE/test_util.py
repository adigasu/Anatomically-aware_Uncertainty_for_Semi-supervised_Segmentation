import os
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataloaders.medical_dataloader import make_dataset, normalize_intensity, percentile_clip
from torchvision.utils import make_grid
from torchvision.utils import save_image


def test_all_case(net, root_path, split='test', normalization=None, num_classes=2, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=False, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    all_metric = [0.0 for _ in range(num_classes-1)]

    image_list = make_dataset(root_path, split)

    for index in range(len(image_list)):
        img_path, mask_path = image_list[index]
        image = nib.load(img_path).get_fdata(dtype=np.float32)
        label = nib.load(mask_path).get_fdata(dtype=np.float32)
        id = os.path.basename(img_path)
        image = normalize_intensity(image, normalization=normalization)

        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, label, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = np.zeros(4)
            for c in range(1, num_classes):
                pred_c = prediction == c
                label_c = label == c
                if np.sum(pred_c) == 0:
                    score = (0,0,0,0)
                else:
                    score = calculate_metric_percase(pred_c, label_c)
                single_metric += np.asarray(score)
                all_metric[c-1] += np.asarray(score)

        single_metric /= 4.0
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path + id + "_score_map.nii.gz")
            #nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            #nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    print('average metric of {} is {}'.format(1, all_metric[0] / len(image_list)))
    print('average metric of {} is {}'.format(2, all_metric[1] / len(image_list)))
    print('average metric of {} is {}'.format(3, all_metric[2] / len(image_list)))
    print('average metric of {} is {}'.format(4, all_metric[3] / len(image_list)))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape[-3:]

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape[-3:]

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes, ) + image.shape[-3:]).astype(np.float32)
    cnt = np.zeros(image.shape[-3:]).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                y, _ = net(test_patch)
                y = torch.sigmoid(y)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
