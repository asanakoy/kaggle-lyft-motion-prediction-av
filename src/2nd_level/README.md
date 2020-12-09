# Transformer second level model
## Usage
### 1. Define data paths
Change first level predictions, sample submission, model and final submission paths in the `config.py` file. First level predictions must be in `.npz` format.

Example:
```
# Paths
PRED_PATHS = ['../../test_preds/sub_140_xxl_xception41_516_test.npz',
              '../../test_preds/sub_140_xxl_xception41_bs128_524_test.npz',
              '../../test_preds/sub_140_xxl_xception41_wp_518_test.npz',
              '../../test_preds/sub_140_xxl_xception65_368_test.npz',
              '../../test_preds/sub_140_xxl_xception71_370_test.npz',
              '../../test_preds/sub_140_xxl_enet_b5_368_test.npz']
MODE_16_PATH = '../../test_preds/sub_140_xxl_xception41_16modes_244_test.npz'
SAMPLE_SUB_PATH = '../lyft-motion-prediction-autonomous-vehicles/multi_mode_sample_submission.csv'
MODEL_PATH = './'
INFER_SAVE_PATH = './'
```

### 2. Training
Run `train.py`. The model will be saved in `MODEL_PATH` as `transformer.bin`.

### 3. Inference
Run `infer.py`. The final submission will be saved in `INFER_SAVE_PATH` as `submission.csv`.
