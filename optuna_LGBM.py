import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import sklearn.metrics
import matplotlib.pyplot as plt

DATA_PATH_TRAIN = r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\train.csv'
DATA_PATH_TEST = r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\test.csv'

X_train = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\X_train.csv')
X_valid = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\X_valid.csv')
y_train = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\y_train.csv')
y_valid = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\y_valid.csv')


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    
    train_x, valid_x, train_y, valid_y = X_train, X_valid, y_train, y_valid
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 100),
        'max_bin': trial.suggest_int('max_bin', 400, 800),
        'cat_l2': trial.suggest_float('cat_l2', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 1e-8, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 1e-8, 10.0, log=True)

    }   
        #      'metric': 'rmse', 
        # 'n_estimators': trial.suggest_int("n_estimators", 5000, 10000),
        # 'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 20.0),
        # 'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 20.0),
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 1e-8, 10.0),
        # 'subsample': trial.suggest_float('subsample', 1e-8, 10.0),
        # 'learning_rate': trial.suggest_float('learning_rate', 1e-8, 10.0),
        # 'max_depth': trial.suggest_int('max_depth', 5, 100),
        # 'num_leaves': trial.suggest_int("num_leaves", 2, 256),
        # 'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
        # 'max_bin': trial.suggest_int('max_bin', 400, 800),
        # 'cat_l2': trial.suggest_float('cat_l2', 1e-8, 10.0)   
    #  
        
    # }
    # 'reg_alpha': 10.924491968127692,
    # 'reg_lambda': 17.396730654687218,
    # 'colsample_bytree': 0.21497646795452627,
    # 'subsample': 0.7582562557431147,
    # 'learning_rate': 0.009985133666265425,
    # 'max_depth': 18,
    # 'num_leaves': 63,
    # 'min_child_samples': 27,
    # 'max_bin': 523,
    # 'cat_l2': 0.025083670064082797
    
# lgbm_parameters = {
#     'metric': 'rmse', 
#     'n_jobs': -1,
#     #'boosting_type': 'dart',
#     'n_estimators': 50000,
#     'reg_alpha': 10.924491968127692,
#     'reg_lambda': 17.396730654687218,
#     'colsample_bytree': 0.21497646795452627,
#     'subsample': 0.7582562557431147,
#     'learning_rate': 0.009985133666265425,
#     'max_depth': 18,
#     'num_leaves': 63,
#     'min_child_samples': 27,
#     'max_bin': 523,
#     'cat_l2': 0.025083670064082797
# }
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.mean_squared_error(valid_y, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))

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
    
    
    
      
#         Best trial:
#   Value: 0.6138576770263755
#   Params:
#     lambda_l1: 7.377885420330146
#     lambda_l2: 9.188883242162573e-08
#     num_leaves: 113
#     feature_fraction: 0.44523390427772924
#     bagging_fraction: 0.9395258081240945
#     bagging_freq: 7
#     min_child_samples: 70


#   Value: 0.6124179447480895
#   Params:
#     lambda_l1: 5.214467977121649
#     lambda_l2: 2.97779661027187
#     num_leaves: 176
#     feature_fraction: 0.4117792695159365
#     bagging_fraction: 0.9761993947057349
#     bagging_freq: 2
#     min_child_samples: 20
#     max_depth: 44
#     max_bin: 695
#     cat_l2: 5.468039376287219e-05
#     subsample: 6.817951922612396
#     colsample_bytree: 3.4656061677969013e-07
