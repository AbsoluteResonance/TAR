import cv2
import h5py
import os
import math
from natsort import natsorted
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label

def normalize_image(data: np.ndarray):
        data_min = np.min(data)
        # normalization = (each element - min element) / (max - min)
        return (data - data_min) / (np.max(data) - data_min)

def getLargestCC(segmentation):
    labels = label(segmentation)
    #assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC

def var_all_case_LA(model, root_dir, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4):
    with open(os.path.join(root_dir, 'test.list'), 'r') as f:
        image_list = f.readlines()

    image_list = [root_dir + "/LA_data/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:] # (175, 132, 88)
        label = h5f['label'][:] # (175, 132, 88)

        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def var_all_case_BraTS19(model, root_path, num_classes, patch_size=(96, 96, 64), stride_xy=16, stride_z=4):
    image_list = []
    with open(os.path.join(root_path, "val.txt"), 'r') as f:
        case_ids = [line.strip() for line in f if line.strip()]
        image_list = [os.path.join(root_path, "data", f"{case_id}.h5") for case_id in case_ids]    
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image1 = h5f['image'][:] # (192, 192, 64), dtype: float64
        label1 = h5f['label'][:] # (192, 192, 64), dtype: uint8
        image = np.transpose(image1, (2, 1, 0))
        label = np.transpose(label1, (2, 1, 0))
        
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case_BraTS19(model, image_list, num_classes, patch_size=(96, 96, 64), stride_xy=16, stride_z=4, save_result=True, 
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):

    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        # if save_result:
        #     nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
        #     #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric

def var_all_case_Pancreas(model, root_path, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4):
    image_list = []
    with open(os.path.join(root_path, "test1.list"), "r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
        image_list = [os.path.join(root_path, f"Pancreas_data/{case_id}") for case_id in case_ids]

    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]  # 
        label = h5f['label'][:].astype(np.uint8)

        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case_Pancreas(model, image_list, num_classes, device, patch_size=(96, 96, 64), stride_xy=16, stride_z=4, save_result=True, 
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, device, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        # if save_result:
        #     nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
        #     #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric

def var_all_case_ISLES22(root_path, model, num_classes, device, patch_size=(96, 96, 64), stride_xy=16, stride_z=4):
    image_list = []
    with open(os.path.join(root_path, "val.list"), 'r') as f:
        case_ids = [line.strip() for line in f if line.strip()]
        image_list = [os.path.join(root_path, f"{case_id}.h5") for case_id in case_ids]
    
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:] # (192, 192, 64), dtype: float64
        label = h5f['mask'][:] # (192, 192, 64), dtype: uint8

        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, device, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case_ISLES22(model, image_list, num_classes, patch_size=(96, 96, 64), stride_xy=16, stride_z=4, save_result=True, 
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['mask'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        # if save_result:
        #     nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
        #     #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
        #     nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric

def test_all_case(model, image_list, num_classes, device, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                   test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, device, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'/performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

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
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    _, y, _ = model(test_patch)
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def var_all_case_LA_plus(model_l, model_r, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4):
   
    with open('../data/LA/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = ["../data/LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case_plus(model_l, model_r, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    
    with open(test_save_path+'/performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric

def test_single_case_plus(model_l, model_r, image, stride_xy, stride_z, patch_size, device, num_classes=1):
    w, h, d = image.shape

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
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).to(device)

                with torch.no_grad():
                    y1_l, _ = model_l(test_patch)
                    y1_r, _ = model_r(test_patch)
                    y1 = (y1_l + y1_r) / 2
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


# def calculate_metric_percase(pred, gt):
#     dice = metric.binary.dc(pred, gt)
#     jc = metric.binary.jc(pred, gt)
#     hd = metric.binary.hd95(pred, gt)
#     asd = metric.binary.asd(pred, gt)

#     return dice, jc, hd, asd


def calculate_metric_percase1(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    #jc = metric.binary.jc(pred, gt)
    jc = abs(metric.binary.ravd(pred, gt))
    
    # Check if gt contains any binary objects
    if gt.sum() == 0:
        hd = 0.0 # float('inf')  # or another value that indicates an undefined HD
        asd = 0.0 # float('inf')  # same here
    else:
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
    
    return dice, jc, hd, asd

def var_all_case_IAVS(model, root_path, num_classes, patch_size=(96, 96, 64), stride_xy=16, stride_z=4):
    image_list = []
    with open(os.path.join(root_path, "val.txt"), 'r') as f:
        case_ids = [line.strip() for line in f if line.strip()]
        image_list = [os.path.join(root_path, "images", f"{case_id}.nii.gz") for case_id in case_ids]    
    loader = tqdm(image_list)
    total_dice = 0.0
    for image_path in loader:
        image = nib.load(image_path).get_fdata()
        label = nib.load(image_path.replace("/images/", "/labels/")).get_fdata()
        image = np.transpose(image, (2, 1, 0))
        label = np.transpose(label, (2, 1, 0))
        
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice

def test_all_case_IAVS(model, image_list, num_classes, patch_size=(96, 96, 64), stride_xy=16, stride_z=4, save_result=True, 
                  test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):

    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    with open(test_save_path+'../performance.txt', 'w') as f:
        for image_path in loader:
            nib_image = nib.load(image_path)
            image = nib_image.get_fdata()
            label = nib.load(image_path.replace("/images/", "/labels/")).get_fdata()
            if preproc_fn is not None:
                image = preproc_fn(image)
            image = np.transpose(image, (2, 1, 0))
            label = np.transpose(label, (2, 1, 0))
            prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
            if nms:
                prediction = getLargestCC(prediction)
                
            if np.sum(prediction)==0:
                single_metric = (0,0,0,0)
            else:
                single_metric = calculate_metric_percase(prediction, label[:])
                
            if metric_detail:
                f.writelines('%02d,\t%.5f, %.5f, %.5f, %.5f\n' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
                print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]), f',{os.path.basename(image_path[:-7])}')

            total_metric += np.asarray(single_metric)
            
            if save_result:
                nib.save(nib.Nifti1Image(np.transpose(prediction.astype(np.float32), (2, 1, 0)), nib_image.affine, nib_image.header), test_save_path +  f"{os.path.basename(image_path[:-7])}_pred.nii.gz")
                pass
            ith += 1

        avg_metric = total_metric / len(image_list)
        print('average metric is {}'.format(avg_metric))
        
        f.writelines('average metric is {} \n'.format(avg_metric))
    return avg_metric