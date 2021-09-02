import pandas as pd
from sklearn.model_selection import train_test_split
from BorutaShap import BorutaShap

DATA_PATH_TRAIN = r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\train.csv'
DATA_PATH_TEST = r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\test.csv'

train = pd.read_csv(DATA_PATH_TRAIN, index_col='id')
test = pd.read_csv(DATA_PATH_TEST, index_col='id')

# concatinate data for the sake of consistency of data preparation
X = pd.concat([train.iloc[:, :-1], test])
y = train.iloc[:, -1]

# One Hot encode categorical columns
X = pd.get_dummies(X)

# separate the data after encoding
x = X.loc[train.index]
y = y.loc[train.index]
test_proccessed = X.loc[test.index]

# train test split 
X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=121)


def save(csv_file:pd.DataFrame, name:str):
    csv_file.to_csv(rf'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\{name}.csv', index=False)


save(x, 'x')
save(y, 'y')
save(X_train, 'X_train')
save(X_valid, 'X_valid')
save(y_train, 'y_train')
save(y_valid, 'y_valid')
save(test_proccessed, 'test_proccessed')


# y = train_y['target']
# train = train_y.drop(['target'], axis=1)

cat_cols = [col for col in train.columns if 'cat' in col]
num_cols = [col for col in train.columns if 'cont' in col]

# The following due to the discussion suggestion
cat1_cat5_cat8 = [col for col in x.columns if ('cat1' in col) or ('cat5' in col) or ('cat8'in col)]
cat1_cat5_cat8.append('cat3_C')
cols_to_keep = num_cols + cat1_cat5_cat8

x_golden = x[cols_to_keep]

save(x_golden, 'x_golden')



model = XGBRegressor(tree_method= 'gpu_hist')

# if classification is False it is a Regression problem
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=False)

Feature_Selector.fit(X=train, y=y, n_trials=100, sample=False,
            	     train_or_test = 'test', normalize=True,
                     verbose=True)