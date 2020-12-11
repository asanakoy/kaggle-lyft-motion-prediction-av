# The 3rd place solution for competition "Lyft Motion Prediction for Autonomous Vehicles" at Kaggle

### Team behind this solution:
1. Artsiom Sanakoyeu [[Homepage](https://gdude.de)] [[Twitter](https://twitter.com/artsiom_s)] [[Telegram Channel](https://t.me/gradientdude)] [[LinkedIn](https://www.linkedin.com/in/sanakoev)]
2. Dmytro Poplavskiy [[Kaggle]](https://www.kaggle.com/dmytropoplavskiy) [[LinkedIn]](https://www.linkedin.com/in/dmytropoplavskiy/)
3. Artsem Zhyvalkouski [[Kaggle]](https://www.kaggle.com/aruchomu) [[Twitter]](https://twitter.com/artem_aruchomu) [[GitHub]](https://github.com/heartkilla) [[LinkedIn]](https://www.linkedin.com/in/zhyvalkouski/)

## How to reproduce results
0. Set the paths in the configs:
  - Set path where to store prerendered dataset in [src/1st_level/config.py](src/1st_level/config.py)
  - Set path where the predits of the 1st level models are saved in [src/2nd_level/config.py](src/2nd_level/config.py)

1. Prepare data.
```
bash prepare_data_train.sh
```
2. Train 1st level models.
```
bash train.sh
```
3. Run inference of 1st level models on the test set.
```
bash prepare_data_test.sh
bash predict_test_l1.sh
```
4. Train 2nd level model on the predicts of the 1st level models on the test set.
```
python src/2nd_level/train.py
```
Make sure you've set all paths right in `2nd_level/config.py` w.r.t. the `2nd_level` directory.

5. Predict on the test set using the 2nd level model.
```
python src/2nd_level/infer.py
```

## Extra
More details on how to use 2nd level model are in [src/2nd_level/README.md](src/2nd_level/README.md)
