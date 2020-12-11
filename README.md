# The 3rd place solution for competition "Lyft Motion Prediction for Autonomous Vehicles" at Kaggle

### Team behind this solution:
1. Artsiom Sanakoyeu [[Homepage](https://gdude.de)] [[Twitter](https://twitter.com/artsiom_s)] [[Telegram Channel](https://t.me/gradientdude)] [[LinkedIn](https://www.linkedin.com/in/sanakoev)]
2. Dmytro Poplavskiy [[Kaggle]](https://www.kaggle.com/dmytropoplavskiy) [[LinkedIn]](https://www.linkedin.com/in/dmytropoplavskiy/)
3. Artsem Zhyvalkouski [[Kaggle]](https://www.kaggle.com/aruchomu) [[Twitter]](https://twitter.com/artem_aruchomu) [[GitHub]](https://github.com/heartkilla) [[LinkedIn]](https://www.linkedin.com/in/zhyvalkouski/)

## How to reproduce results
0. Set the paths in the configs:
  - Set path where to store prerendered dataset in [src/1st_level/config.py](src/1st_level/config.py)
  - [Optional] Set path where the predicts of the 1st level models are saved in [src/2nd_level/config.py](src/2nd_level/config.py)

1. Install dependencies. 
  - 'pip install -r requirements.txt'
  - Patch l5kit with [l5kit.patch](l5kit.patch)

2. Prepare data.
```
bash prepare_data_train.sh
```
3. Train 1st level models.
```
bash train.sh
```
4. Run inference of 1st level models on the test set.
```
bash prepare_data_test.sh
bash predict_test_l1.sh
```
4. Train 2nd level model on the predicts of the 1st level models on the test set.
```
cd src/2nd_level && python train.py
```
Make sure you've set all paths right in `2nd_level/config.py` w.r.t. the `2nd_level` directory.

6. Predict on the test set using the 2nd level model.
```
cd src/2nd_level && python infer.py
```

The file witn final predictions will be saved to `src/2nd_level/submission.csv'.

## Extra
- To skip training the 1st level models, you can download the pretrained weights by running `bash download_1st_level_weights.sh`.
- To skip training and inference of the 1st level models, you can download all predicts. More details on this are in [src/1st_level/submissions](src/1st_level/submissions).
- More details on how to use 2nd level model are in [src/2nd_level](src/2nd_level).
- Our final 2nd level model with *XX Private LB score* is already committed in this repository ([src/2nd_level/transformer.bin](src/2nd_level/transformer.bin)). To run inference using this model you can directly execute `cd src/2nd_level && python infer.py`.

