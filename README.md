# The 3rd place solution for competition "Lyft Motion Prediction for Autonomous Vehicles" at Kaggle

### Team behind this solution:
1. Artsiom Sanakoyeu [[homepage](https://gdude.de)] [[linkedin](TODO)]
2. Dmytro Poplavskiy
3. Artsem Zhyvalkouski [[Kaggle]](https://www.kaggle.com/aruchomu) [[Twitter]](https://twitter.com/artem_aruchomu) [[GitHub]](https://github.com/heartkilla) [[LinkedIn]](https://www.linkedin.com/in/zhyvalkouski/)

## How to replicate results
1. Prepare data.
2. Train 1st level models.
3. Run inference of 1st level models on the test set.
4. Train 2nd level model on the predicts of the 1st level models on the test set.
```
python src/2nd_level/train.py
```
Make sure you've set all paths right in `2nd_level/config.py` w.r.t. the `2nd_level` directory.

5. Predict on the test set using the 2nd level model.
```
python src/2nd_level/infer.py
```
