# 30daysML

This is top 14% solution of Kaggles [30 Days of ML competition.](https://www.kaggle.com/c/30-days-of-ml)

The dataset is used for this competition is synthetic (and generated using a CTGAN), but based on a real dataset.
The original dataset deals with predicting the amount of an insurance claim. Although the features are anonymized, 
they have properties relating to real-world features.

The solution consists of simple EDA and 2-level stacking. The data has no missing values but some of the data columns appear to be just noise.
Meaningful columns were selected using [Boruta-Shap](https://github.com/Ekeany/Boruta-Shap). I used One Hot Encoding for categorical columns.

Base models are 3 XGboost, 3 LightGBM and 1 CatBoost.
Parameters were tuned with [Optuna](https://optuna.org/). First level stacking consists of one XGboost, RandomForestRegressor and GradientBoostingRegressor models.
And final level of stacking is done with simple LinearRegression model. 

This approach gave me 0.71618 RMSE score which is 0.2% different from the first place.

I also tried different data normalization, simple feature generation and pseudo labelling but it didn't work out for me.
