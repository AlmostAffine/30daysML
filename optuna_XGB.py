import numpy as np
import optuna
import time
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm



DATA_PATH_TRAIN = r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\train.csv'
DATA_PATH_TEST = r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\test.csv'

# X_train = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\X_train.csv')
# X_valid = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\X_valid.csv')
# y_train = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\y_train.csv')
# y_valid = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\y_valid.csv')
x = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\x.csv')
y = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\y.csv')
x_golden = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\x_golden.csv')
cat_cols = [col for col in x.columns if 'cat' in col]

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    
    
    param = {
        #'tree_method': 'gpu_hist',
        # 'metric': 'rmse',
        'objective': 'reg:squarederror',
        #'scale_pos_weight': 1,
        'n_estimators': 200,
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'grow_policy':'lossguide',
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 50.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 50.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
           
        # "objective": "regression",
        # 'metric': 'rmse', 
        # 'n_estimators': trial.suggest_int("n_estimators", 5000, 100000),
        # 'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 20.0),
        # 'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 20.0),
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 1e-8, 10.0),
        # 'subsample': trial.suggest_float('subsample', 1e-8, 10.0),
        # 'learning_rate': trial.suggest_float('learning_rate', 1e-8, 10.0),
        # 'max_depth': trial.suggest_int('max_depth', 5, 100),
        # 'num_leaves': trial.suggest_int("num_leaves", 2, 256),
        # 'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
        # 'max_bin': trial.suggest_int('max_bin', 400, 800),
        # 'cat_l2': trial.suggest_float('cat_l2', 1e-8, 10.0),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0)   
    }   
    
    xgb_val_pred = np.zeros(len(y))
    # xgb_test_pred = np.zeros(len(test))
    mse = []
    kf = KFold(n_splits=10, shuffle=True)

    for trn_idx, val_idx in tqdm(kf.split(x_golden, y)):
        x_train_idx = x.iloc[trn_idx]
        y_train_idx = y.iloc[trn_idx]
        x_valid_idx = x.iloc[val_idx]
        y_valid_idx = y.iloc[val_idx]

        xgb_model = XGBRegressor(**param)
        xgb_model.fit(x_train_idx, y_train_idx, verbose=3
                        # eval_set = ((x_valid_idx,y_valid_idx)),
                        # verbose = -1, 
                        # early_stopping_rounds = 400
                        #categorical_feature=cat_cols
                    )  
        # xgb_test_pred += xgb_model.predict(test_golden)/10   needed for competition prediction
        mse.append(mean_squared_error(y_valid_idx, xgb_model.predict(x_valid_idx), squared=False ))
        
    accuracy = np.mean(mse)

    return accuracy


if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
    stop_time = time.time()
    print("Number of finished trials: {}".format(len(study.trials)))
    print('Time ellapsed: {}'.format(stop_time - start_time))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print('Param importance:')    
    importance = optuna.importance.get_param_importances(study=study)
    # print(importance)  
    
    plt.bar(range(len(importance)), list(importance.values()), align='center')
    plt.xticks(range(len(importance)), list(importance.keys()))
    plt.show()
    