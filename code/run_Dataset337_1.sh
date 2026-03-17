# Description: Run the training code for DyCON-Dataset337_aneart

# Main training script
/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/anadonda3/envs/nnunetv2/envs/semi_unet/bin/python train_DyCON_IAVS.py \
--root_dir "../data/Dataset337_aneart" \
--exp "Dataset337_aneart_tar_a0.5_b1.5" \
--model "unet_3D" \
--max_iterations 10000 \
--temp 0.6 \
--batch_size 8 \
--labelnum 35 \
--gpu_id 0 \
--num_workers 0 \
--alpha 0.5 \
--beta 1.5

/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/6820-xiaofeiyang/anadonda3/envs/nnunetv2/envs/semi_unet/bin/python test_IAVS.py \
--exp "Dataset337_aneart_tar_a0.5_b1.5"