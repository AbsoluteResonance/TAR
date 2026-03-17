from scipy import ndimage
import os,shutil,nibabel as nib,numpy as np

from clDice.cldice_metric.cldice import clDice
from boundary_iou.iou3d import compute_boundary_iou_3d as bIoU

import torch
import cupy as cp

import warnings

warnings.filterwarnings("ignore")

#device = torch.device(f"cuda:0")

def Dice(outputs, labels, smooth=1e-5):
    outputs_sum = np.sum(outputs)
    labels_sum = np.sum(labels)
    if labels_sum == 0:
        if outputs_sum == 0:
            return None, 1.0
        else: return 0.0, 0.0
    else:
        intersection = np.sum(outputs * labels)
        dice = (2.0 * intersection + smooth) / (outputs_sum + labels_sum + smooth)
        return dice, dice
    
def calc_dice_nnunet(input_dir, ref_dir):
    dices_0 = []
    dices_1 = []

    cldices = []
    bious = []

    listdir = os.listdir(input_dir)
    listdir.sort()
    for filename in listdir:
        if not filename.endswith('.nii.gz'): continue
        if not os.path.exists(ref_dir + filename): continue
        
        np_output = nib.load(input_dir + filename).get_fdata()
        np_ref = nib.load(ref_dir + filename).get_fdata()
        
        dice_0, dice_1 = Dice(np_output, np_ref)

        cldice = float(clDice(torch.from_numpy(np_output).unsqueeze(0).unsqueeze(0), torch.from_numpy(np_ref).unsqueeze(0).unsqueeze(0)))
        biou = float(bIoU(torch.from_numpy(np_output).unsqueeze(0).unsqueeze(0), torch.from_numpy(np_ref).unsqueeze(0).unsqueeze(0)))
        
        if not dice_0 == None: 
            dices_0.append(dice_0)
            print(f"{filename}, {dice_0:.4f}, {dice_1:.4f}", end='')
        else:
            print(f"{filename}, None, {dice_1:.4f}", end='')
        dices_1.append(dice_1)

        print(f"cldice = {cldice:.4f}, biou = {biou:.4f}")

        cldices.append(cldice)
        bious.append(biou)

    print(input_dir)
    print(f"len(dice_0) = {len(dices_0)}, len(dice_1) = {len(dices_1)}")
    print(f"mean(dice_0) = {np.mean(dices_0):.4f}, mean(dice_1) = {np.mean(dices_1):.4f}")
    print(f"mean(cldice) = {np.mean(cldices):.4f}, mean(bious) = {np.mean(bious):.4f}")


if __name__ == "__main__":
    #input_dir = "data/333/full_f5_e500/"
    #ref_dir = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/dataplatform/lifescience/bloodflow/srcfiles22/nii/lu_90_final/labels/"
    
    #input_dir = f"../../unet/dataset/data/Dataset500_aneart/others/post_ane/"
    #ref_dir = "../dataset/nnUNet_raw/Dataset335_ane/labelsTs/"

    input_dir = f"../dataset/data/Dataset500_aneart/others/42_932/pred_aneart_merged/"
    #input_dir = f"../../unet/dataset/data/Dataset407_aneart/results/310_835/pred_aneart_merged/"
    #input_dir = f"../../unet/dataset/data/Dataset500_aneart/others/box_detection/nnDetection_334/pred_aneart_merged/"
    ref_dir = f"../../nnUNetv2/dataset/nnUNet_raw/Dataset334_aneart/labelsTs/"

    dices_0 = []
    dices_1 = []
    
    cldices = []
    bious = []
    
    listdir = os.listdir(input_dir)
    listdir.sort()
    for filename in listdir:
        if not filename.endswith('.nii.gz'): continue
        if not os.path.exists(ref_dir + filename): continue
        
        np_output = nib.load(input_dir + filename).get_fdata()
        np_ref = nib.load(ref_dir + filename).get_fdata()
        
        dice_0, dice_1 = Dice(np_output, np_ref)
        
        cldice = clDice(torch.from_numpy(np_output).unsqueeze(0).unsqueeze(0), torch.from_numpy(np_ref).unsqueeze(0).unsqueeze(0))
        biou = float(bIoU(torch.from_numpy(np_output).unsqueeze(0).unsqueeze(0), torch.from_numpy(np_ref).unsqueeze(0).unsqueeze(0)))

        
        if not dice_0 == None: 
            dices_0.append(dice_0)
            print(f"{filename}, {dice_0:.4f}, {dice_1:.4f}", end='')
        else:
            print(f"{filename}, None, {dice_1:.4f}", end='')
        dices_1.append(dice_1)

        print(f" cldice: {cldice:.4f}, biou: {biou:.4f}")

        cldices.append(cldice)
        bious.append(biou)

    print(input_dir)
    print(f"len(dice_0) = {len(dices_0)}, len(dice_1) = {len(dices_1)}")
    print(f"mean(dice_0) = {np.mean(dices_0):.4f}, mean(dice_1) = {np.mean(dices_1):.4f}")
    print(f"mean(cldice) = {np.mean(cldices):.4f}, mean(bious) = {np.mean(bious):.4f}")
