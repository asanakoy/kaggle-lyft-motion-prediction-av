# Transformer second level model
This model is based on [Set Transformer](https://arxiv.org/abs/1810.00825)

## Usage
### 1. Define data paths
Change first level predictions, sample submission, model and final submission paths in the [config.py](config.py) file if needed (although, the default config should work as well). First level predictions must be in `.npz` format.

Example:
```
# Paths
PRED_PATHS = ['../1st_level/submissions/sub_140_xxl_xception41_516_test.npz',
              '../1st_level/submissions/test_preds/sub_140_xxl_xception41_bs128_524_test.npz',
              '../1st_level/submisisons/test_preds/sub_140_xxl_xception41_wp_518_test.npz',
              '../1st_level/submisisons/test_preds/sub_140_xxl_xception65_368_test.npz',
              '../1st_level/submisisons/test_preds/sub_140_xxl_xception71_370_test.npz',
              '../1st_level/submisisons/test_preds/sub_140_xxl_enet_b5_368_test.npz']
MODE_16_PATH = '../1st_level/submissions/sub_140_xxl_xception41_16modes_244_test.npz'
SAMPLE_SUB_PATH = '../1st_level/submissions/multi_mode_sample_submission.csv'
MODEL_PATH = './'
INFER_SAVE_PATH = './'
```

### 2. Training
We provide our pretrained final model in this repository -- `transformer.bin`. It scores 10.227 at Public LB and 9.404 at Private LB.

If you want to retrain the model from scratch, run `python train.py`.   
The model will be saved in `MODEL_PATH` as `transformer.bin`.

### 3. Inference
Run `python infer.py`. The final submission will be saved in `INFER_SAVE_PATH` as `submission.csv`.
