# eval_cldice.py
import os
import glob
import numpy as np
import nibabel as nib
import torch

from cldice import clDice

import warnings
warnings.filterwarnings("ignore")

def load_niigz_as_tensor(path):
    x = nib.load(path).get_fdata()
    x = (x > 0).astype(np.uint8)          # 二值化（按需改阈值）
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(0).unsqueeze(0)       # -> [B=1, C=1, D, H, W]
    return x

def main():
    pred_dirs = [
        # "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Fully_supervised_17/Prediction",
                # "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Adversarial_Network_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Cross_Pseudo_Supervision_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Regularized_Dropout_3D_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/ACMT_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Interpolation_Consistency_Training_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Uncertainty_Aware_Mean_Teacher_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Entropy_Minimization_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/UGMCL_3D_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/CML/code/model/CML/IAVS_CML_17_labeled/VNet_predictions",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Mean_Teacher_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_17labelnum/Mean_Teacher_TAR_17/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/DyCON/models/Dataset337_aneart/UNET_3D_17labels_mse_gamma2.0_Focal_Teacher_temp0.6_beta0.5-5.0_max_iterations10000/Dataset337_aneart_predictions",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/DyCON/models/Dataset337_aneart_tar/UNET_3D_17labels_mse_gamma2.0_Focal_Teacher_temp0.6_beta0.5-5.0_max_iterations10000",
                 
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Fully_supervised_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Adversarial_Network_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Cross_Pseudo_Supervision_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Regularized_Dropout_3D_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/ACMT_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Interpolation_Consistency_Training_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Uncertainty_Aware_Mean_Teacher_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Entropy_Minimization_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/UGMCL_3D_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/CML/code/model/CML/IAVS_CML_71_labeled/VNet_predictions",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Mean_Teacher_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/SSL4MIS/model/Dataset337_aneart_71labelnum/Mean_Teacher_TAR_71/Prediction",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/DyCON/models/Dataset337_aneart/UNET_3D_71labels_mse_gamma2.0_Focal_Teacher_temp0.6_beta0.5-5.0_max_iterations10000/Dataset337_aneart_predictions",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/DyCON/models/Dataset337_aneart_tar/UNET_3D_17labels_mse_gamma2.0_Focal_Teacher_temp0.6_beta0.5-5.0_max_iterations10000/Dataset337_aneart_tar_predictions",
                #  "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/DyCON/models/Dataset337_aneart_tar/UNET_3D_71labels_mse_gamma2.0_Focal_Teacher_temp0.6_beta0.5-5.0_max_iterations10000/Dataset337_aneart_tar_predictions",
                 ]
    gt_dir = "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/meshunet/semi-unet/DyCON/data/Dataset337_aneart/labels"
    for pred_dir in pred_dirs:
        pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
        scores = []

        for p in pred_paths:
            name = os.path.basename(p)
            g = os.path.join(gt_dir, name.replace("_pred2", "").replace("_pred", ""))
            if not os.path.exists(g):
                # print(f"skip (no gt): {name}")
                continue

            v_p = load_niigz_as_tensor(p)
            v_l = load_niigz_as_tensor(g)

            score = clDice(v_p, v_l)
            # print(f"{name}: clDice={score:.6f}")
            if not np.isnan(score):
                scores.append(score)
        

        scores = np.array(scores, dtype=np.float64)
        print(pred_dir)
        print(f"& {scores.mean()*100:.2f} $\pm$ {scores.std(ddof=1)*100:.2f} ")

if __name__ == "__main__":
    main()
