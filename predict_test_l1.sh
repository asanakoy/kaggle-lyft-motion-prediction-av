cd src/1st_level

python predict.py prepare_submission 140_xxl_xception41  --epoch 516
python predict.py prepare_submission 140_xxl_xception65  --epoch 368
python predict.py prepare_submission 140_xxl_xception71  --epoch 370
python predict.py prepare_submission 140_xxl_enet_b5  --epoch 368
python predict.py prepare_submission 140_xxl_xception41_16modes  --epoch 244  # error from write_pred_csv is ok here

# a few models have been trained on multiple GPU, for some reason not compatible with the normal training,
# probably due to syncbn used, so separate prediction
python train_distributed.py prepare_submission 140_xxl_xception41_bs128 --epoch 524 --rank 0 --world-size 1
python train_distributed.py prepare_submission 140_xxl_xception41_wp --epoch 518 --rank 0 --world-size 1
