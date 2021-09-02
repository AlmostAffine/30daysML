import numpy as np
import optuna
import time
import pandas as pd
import lightgbm as lgb
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
cat_cols = [col for col in x_golden.columns if 'cat' in col]
boruta_shap = ['cont8', 'cont13', 'cont5', 
               'cont2', 'cont12', 'cat8_C', 
               'cat3_C', 'cat1_A', 'cont3', 
               'cont11', 'cont7', 'cont0', 
               'cont10', 'cont9', 'cont4', 
               'cont1', 'cont6']



# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).


'''
def objective(trial):
    
    
    param = {
        "objective": "regression",
        'metric': 'RMSE', 
        'n_estimators': 70000,
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 20.0),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 20.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 1e-8, 10.0),
        'subsample': trial.suggest_float('subsample', 1e-8, 10.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 10.0),
        'max_depth': trial.suggest_int('max_depth', 5, 100),
        'num_leaves': trial.suggest_int("num_leaves", 2, 256),
        'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
        'max_bin': trial.suggest_int('max_bin', 400, 800),
        'cat_l2': trial.suggest_float('cat_l2', 1e-8, 10.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0)   
    }   
    
    lgbm_val_pred = np.zeros(len(y))
    # lgbm_test_pred = np.zeros(len(test))
    mse = []
    kf = KFold(n_splits=10, shuffle=True)

    for trn_idx, val_idx in tqdm(kf.split(x_golden, y)):
        x_train_idx = x_golden.iloc[trn_idx]
        y_train_idx = y.iloc[trn_idx]
        x_valid_idx = x_golden.iloc[val_idx]
        y_valid_idx = y.iloc[val_idx]

        lgbm_model = lgb.LGBMRegressor(**param)
        lgbm_model.fit(x_train_idx, y_train_idx, 
                    eval_set = ((x_valid_idx,y_valid_idx)),
                    verbose = -1, 
                    early_stopping_rounds = 400
                    #categorical_feature=cat_cols
                    )  
        # lgbm_test_pred += lgbm_model.predict(test)/10   needed for competition prediction
        mse.append(mean_squared_error(y_valid_idx, lgbm_model.predict(x_valid_idx), squared=False ))
        
    accuracy = np.mean(mse)

    return accuracy


if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
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
    
'''
# Best trial:
#   Value: 0.71893509495
#   Params:
#     n_estimators: 6593
#     reg_alpha: 18.466187259889182
#     reg_lambda: 17.368126793586903
#     colsample_bytree: 0.15831916503019072
#     subsample: 5.863187839942498
#     learning_rate: 0.013135172922927346
#     max_depth: 84
#     num_leaves: 176
#     min_child_samples: 19
#     max_bin: 552
#     cat_l2: 8.310713249310417
#     feature_fraction: 0.40700481284964635
#     bagging_fraction: 0.9435647617521867

# Best trial:
#   Value: 0.7203336378452041  sqrt
#   Params:
#     n_estimators: 5283
#     reg_alpha: 0.8004566595268331
#     reg_lambda: 19.92652301640183
#     colsample_bytree: 7.677604673887766
#     subsample: 6.675340284067552
#     learning_rate: 0.20383905907770702
#     max_depth: 5
#     num_leaves: 177
#     min_child_samples: 75
#     max_bin: 797
#     cat_l2: 9.891385450714491
#     feature_fraction: 0.8487904862526569
#     bagging_fraction: 0.40533885433815675

# Best trial:
#   Value: 0.7176188261393596
#   Params:
#     n_estimators: 62828
#     reg_alpha: 4.065339331962444
#     reg_lambda: 0.1263476879597878
#     colsample_bytree: 9.935408655121904
#     subsample: 6.122664485508403
#     learning_rate: 0.19191061268931617
#     max_depth: 72
#     num_leaves: 3
#     min_child_samples: 74
#     max_bin: 781
#     cat_l2: 9.97593042739126
#     feature_fraction: 0.5527369437271845
#     bagging_fraction: 0.7835717912217517

#   GF, 50
#   Value: 0.7199973165155998
#   Params:
#     n_estimators: 57672
#     reg_alpha: 9.814764261612654
#     reg_lambda: 10.69054594624928
#     colsample_bytree: 3.3053983356454015
#     subsample: 8.084960353184298
#     learning_rate: 0.03362208617812317
#     max_depth: 63
#     num_leaves: 235
#     min_child_samples: 38
#     max_bin: 594
#     cat_l2: 6.827312202569431
#     feature_fraction: 0.5322927559875398
#     bagging_fraction: 0.47956486260271647

# Value: 0.7211878320543355
#   Params:
#     reg_alpha: 16.576088681024345
#     reg_lambda: 7.459400962544775
#     subsample: 9.788870744058924
#     learning_rate: 0.09083819180210893
#     max_depth: 15
#     num_leaves: 149
#     min_child_samples: 72
#     max_bin: 436
#     cat_l2: 0.1033274590329149
#     feature_fraction: 0.9782978413002409
#     bagging_fraction: 0.9537578467700946

# Number of finished trials: 50 cat_cols
# Time ellapsed: 6603.911472320557
# Best trial:
#   Value: 0.7172767460107918
#   Params:
#     reg_alpha: 9.801970722354566
#     reg_lambda: 18.213248648427943
#     colsample_bytree: 8.38593451669085
#     subsample: 7.513555646258862
#     learning_rate: 1.42185197481976
#     max_depth: 61
#     num_leaves: 2
#     min_child_samples: 51
#     max_bin: 700
#     cat_l2: 1.0101746659652817
#     feature_fraction: 0.40084231029252354
#     bagging_fraction: 0.9341129222600173

# not tested
# Number of finished trials: 50
# Time ellapsed: 15309.898109436035
# Best trial:
#   Value: 0.7173780909732258
#   Params:
#     reg_alpha: 8.552169602366426
#     reg_lambda: 18.306945524144606
#     colsample_bytree: 4.820553447659609
#     subsample: 0.03200709970310678
#     learning_rate: 0.10642204697831348
#     max_depth: 62
#     num_leaves: 4
#     min_child_samples: 44
#     max_bin: 576
#     cat_l2: 7.295567021948898
#     feature_fraction: 0.40845228684965235
#     bagging_fraction: 0.6273055957444446