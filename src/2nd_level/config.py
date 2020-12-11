# Paths
PRED_PATHS = ['../1st_level/submissions/sub_140_xxl_xception41_516_test.npz',
              '../1st_level/submissions/sub_140_xxl_xception41_bs128_524_test.npz',
              '../1st_level/submissions/sub_140_xxl_xception41_wp_518_test.npz',
              '../1st_level/submissions/sub_140_xxl_xception65_368_test.npz',
              '../1st_level/submissions/sub_140_xxl_xception71_370_test.npz',
              '../1st_level/submissions/sub_140_xxl_enet_b5_368_test.npz']
MODE_16_PATH = '../1st_level/submissions/sub_140_xxl_xception41_16modes_244_test.npz'
SAMPLE_SUB_PATH = '../1st_level/submissions/multi_mode_sample_submission.csv'
MODEL_PATH = './'
INFER_SAVE_PATH = './'

# n_modes params
N_3_MODES = 3 * len(PRED_PATHS)
N_INPUT_MODES = N_3_MODES + 16

# Training params
SEED = 1067
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 500
WEIGHT_DECAY = 1e-3
PCT_START = 0.3
DIV_FACTOR = 1e3

# Model params
MODEL_PARAMS = {'dim_input': 50 * 2 + 1,
                'num_outputs': 3,
                'dim_output': 50 * 2,
                'num_inds': 8,
                'dim_hidden': 64,
                'num_heads': 4,
                'ln': False}
