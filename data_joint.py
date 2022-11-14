'''
- saves and loads example datasets that use both categorical and numerical features
- Adult (adult): https://archive.ics.uci.edu/ml/datasets/Adult
- German Credit (credit): https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
'''

import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import os

data_dir = './data/'
train_ratio = 0.7

#######################
### helper function ###
#######################

def normalize_per_feature(X_train, X_test):
    """
    We are not allowed to touch the test data, thus we do the normalization just based on the training data.
    """
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.zeros(len(X_train[0]))
    for i in range(len(X_train[0])):
        X_train_std[i] = np.std(X_train[:, i])
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train, X_test

#####################
### Adult Dataset ###
#####################

def save_adult():

    url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    label = "income"
    columns = features + [label]
    df = pd.read_csv(url_data, names=columns)
    X = df[features].to_numpy()
    y = df[label].to_numpy()
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    # categorcical preporcessing
    cat_idx = [1, 3, 5, 6, 7, 8, 9, 13]
    le = LabelEncoder()
    cat_idx = [1, 3, 5, 6, 7, 8, 9, 13]
    for i in cat_idx:
        X[:, i] = le.fit_transform(X[:, i])

    # numerical preprocessing
    num_idx = [0, 2, 4, 10, 11, 12]

    n_train = int(len(X) * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    # X_train[:, num_idx], X_test[:, num_idx] = normalize_per_feature(X_train[:, num_idx], X_test[:, num_idx])

    # saving the data
    if not os.path.exists("./data/adult"):
        os.mkdir("./data/adult")
    
    np.save('./data/adult/X_train.npy', X_train)
    np.save('./data/adult/X_test.npy', X_test)
    np.save('./data/adult/y_train.npy', y_train)
    np.save('./data/adult/y_test.npy', y_test)
    np.save('./data/adult/X_all.npy', X)
    np.save('./data/adult/y_all.npy', y)

    print('Adult data saved successfully.')

#############################
### German Credit Dataset ###
#############################

def save_credit():

    url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    column_names = [
        'status', 'months', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status', 'other_debtors', 'residence_since', 'property', 'age',
        'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone',
        'foreign_worker', 'credit'
    ]
    label = 'credit'
    features = column_names[:20]

    df = pd.read_csv(url_data, sep=' ', header=None, names=column_names)
    X = df[features].to_numpy()
    y = df[label].to_numpy()
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    label = 'credit'
    features = column_names[:20]

    # categorcical preporcessing
    cat_idx = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    le = LabelEncoder()
    for i in cat_idx:
        X[:, i] = le.fit_transform(X[:, i])

    # numerical preprocessing
    num_idx = [1, 4, 7, 10, 12, 15, 17]
    n_train = int(len(X) * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    # X_train[:, num_idx], X_test[:, num_idx] = normalize_per_feature(X_train[:, num_idx], X_test[:, num_idx])

    # saving the data
    if not os.path.exists("./data/credit"):
        os.mkdir("./data/credit")
    
    np.save('./data/credit/X_train.npy', X_train)
    np.save('./data/credit/X_test.npy', X_test)
    np.save('./data/credit/y_train.npy', y_train)
    np.save('./data/credit/y_test.npy', y_test)
    np.save('./data/credit/X_all.npy', X)
    np.save('./data/credit/y_all.npy', y)

    print('Credit data saved successfully.')
    
######################
### mammal dataset ###
######################

def save_mammal():
    '''
    returns the raw features and targets of the spambase data
    '''
    
    columns = ['bi-rads', 'age', 'shape', 'margin', 'density', 'severity']
    url_data = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data'
    df = pd.read_csv(url_data, names=columns)
    df = df.replace('?', np.nan).dropna()
    X = df.iloc[:, :5].to_numpy().astype(int)
    y = df.iloc[:, 5].to_numpy().astype(int)
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    # categorcical preporcessing
    cat_idx = [2, 3]
    le = LabelEncoder()
    for i in cat_idx:
        X[:, i] = le.fit_transform(X[:, i])

    # numerical preprocessing
    num_idx = [0, 1, 4]
    n_train = int(len(X) * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    # X_train[:, num_idx], X_test[:, num_idx] = normalize_per_feature(X_train[:, num_idx], X_test[:, num_idx])

    # saving the data
    if not os.path.exists("./data/mammal"):
        os.mkdir("./data/mammal")
    
    np.save('./data/mammal/X_train.npy', X_train)
    np.save('./data/mammal/X_test.npy', X_test)
    np.save('./data/mammal/y_train.npy', y_train)
    np.save('./data/mammal/y_test.npy', y_test)
    np.save('./data/mammal/X_all.npy', X)
    np.save('./data/mammal/y_all.npy', y)

    print('Mammal data saved successfully.')
    
######################
### bank dataset ###
######################
    
def save_bank():
    
    path = './data/bank/bank.csv'
    df = pd.read_csv(path, sep=';')
    
    X = df.iloc[:, :16].to_numpy()
    y = df.iloc[:, 16].to_numpy()
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # categorical_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14]
    # numerical_idx = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    n_classes = 2
    cat_idx = [1, 2, 3, 4, 6, 7, 8, 10, 15]
    num_idx = [0, 5, 9, 11, 12, 13, 14]
    
    # categorcical preporcessing
    le = LabelEncoder()
    for i in cat_idx:
        X[:, i] = le.fit_transform(X[:, i])

    # numerical preprocessing
    n_train = int(len(X) * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    # X_train[:, num_idx], X_test[:, num_idx] = normalize_per_feature(X_train[:, num_idx], X_test[:, num_idx])

    # saving the data
    if not os.path.exists("./data/bank"):
        os.mkdir("./data/bank")
    
    np.save('./data/bank/X_train.npy', X_train)
    np.save('./data/bank/X_test.npy', X_test)
    np.save('./data/bank/y_train.npy', y_train)
    np.save('./data/bank/y_test.npy', y_test)
    np.save('./data/bank/X_all.npy', X)
    np.save('./data/bank/y_all.npy', y)

    print('Bank data saved successfully.')
    
########################
### mushroom dataset ###
########################
    
def save_mushroom():
    '''
    returns the raw features and targets of the mushroom data
    '''
    
    url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    df = pd.read_csv(url_data, header=None)
    
    X = df.iloc[:, 1:23].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    
    n_classes = 2
    cat_idx = [i for i in range(22)]
    num_idx = []
        
    # label preprocessing
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # categorcical preporcessing
    le = LabelEncoder()
    for i in cat_idx:
        X[:, i] = le.fit_transform(X[:, i])

    # numerical preprocessing
    n_train = int(len(X) * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    # X_train[:, num_idx], X_test[:, num_idx] = normalize_per_feature(X_train[:, num_idx], X_test[:, num_idx])

    # saving the data
    if not os.path.exists("./data/mushroom"):
        os.mkdir("./data/mushroom")
    
    np.save('./data/mushroom/X_train.npy', X_train)
    np.save('./data/mushroom/X_test.npy', X_test)
    np.save('./data/mushroom/y_train.npy', y_train)
    np.save('./data/mushroom/y_test.npy', y_test)
    np.save('./data/mushroom/X_all.npy', X)
    np.save('./data/mushroom/y_all.npy', y)

    print('Mushroom data saved successfully.')













########################
### get adult dataset ###
########################

def adult(current_fold=None, n_splits=None):
    """
    The labels are in {0, 1}, so n_classes = 2.
    X_train: (22792, 14), X_test: (9769, 14), custom split (first 70% train, rest test; if no cross validation)
    """
    
    # loading train and test split
    data_path = data_dir + 'adult'
    if n_splits == None:
        X_train, X_test = np.load(os.path.join(data_path, 'X_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'X_test.npy'), allow_pickle=True)
        y_train, y_test = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True)
    else:
        X_all, y_all = np.load(os.path.join(data_path, 'X_all.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_all.npy'), allow_pickle=True)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        train_index, test_index = list(kf.split(X_all))[current_fold]
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
    
    # some meta-data
    categorical_idx = [1, 3, 5, 6, 7, 8, 9, 13]
    numerical_idx = [0, 2, 4, 10, 11, 12]
    n_classes = 2
    
    # standardization of numerical features
    X_train[:, numerical_idx], X_test[:, numerical_idx] = normalize_per_feature(X_train[:, numerical_idx], X_test[:, numerical_idx])

    return X_train, y_train, X_test, y_test, n_classes, categorical_idx, numerical_idx

##########################
### get credit dataset ###
##########################

def credit(current_fold=None, n_splits=None):
    """
    The labels are in {0, 1}.
    train: (700, 20), test: (300, 20), n classes: 2. custom split (first 70% train, rest test; if no cross validation is used)
    """
    
    # loading train and test split
    data_path = data_dir + 'credit'
    if n_splits == None:
        X_train, X_test = np.load(os.path.join(data_path, 'X_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'X_test.npy'), allow_pickle=True)
        y_train, y_test = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True)
    else:
        X_all, y_all = np.load(os.path.join(data_path, 'X_all.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_all.npy'), allow_pickle=True)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        train_index, test_index = list(kf.split(X_all))[current_fold]
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        
    # some meta-data
    categorical_idx = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    numerical_idx = [1, 4, 7, 10, 12, 15, 17]
    n_classes = 2
    
    # standardization of numerical features
    X_train[:, numerical_idx], X_test[:, numerical_idx] = normalize_per_feature(X_train[:, numerical_idx], X_test[:, numerical_idx])

    return X_train, y_train, X_test, y_test, n_classes, categorical_idx, numerical_idx

##########################
### get mammal dataset ###
##########################

def mammal(current_fold=None, n_splits=None):
    
    # loading train and test split
    data_path = data_dir + 'mammal'
    if n_splits == None:
        X_train, X_test = np.load(os.path.join(data_path, 'X_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'X_test.npy'), allow_pickle=True)
        y_train, y_test = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True)
    else:
        X_all, y_all = np.load(os.path.join(data_path, 'X_all.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_all.npy'), allow_pickle=True)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        train_index, test_index = list(kf.split(X_all))[current_fold]
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        
    # some meta-data
    categorical_idx = [2, 3]
    numerical_idx = [0, 1, 4]
    n_classes = 2
    
    # standardization of numerical features
    X_train[:, numerical_idx], X_test[:, numerical_idx] = normalize_per_feature(X_train[:, numerical_idx], X_test[:, numerical_idx])

    return X_train, y_train, X_test, y_test, n_classes, categorical_idx, numerical_idx

##########################
### get bank dataset ###
##########################

def bank(current_fold=None, n_splits=None):
    
    # loading train and test split
    data_path = data_dir + 'bank'
    if n_splits == None:
        X_train, X_test = np.load(os.path.join(data_path, 'X_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'X_test.npy'), allow_pickle=True)
        y_train, y_test = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True)
    else:
        X_all, y_all = np.load(os.path.join(data_path, 'X_all.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_all.npy'), allow_pickle=True)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        train_index, test_index = list(kf.split(X_all))[current_fold]
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        
    # some meta-data
    categorical_idx = [1, 2, 3, 4, 6, 7, 8, 10, 15]
    numerical_idx = [0, 5, 9, 11, 12, 13, 14]
    n_classes = 2
    
    # standardization of numerical features
    X_train[:, numerical_idx], X_test[:, numerical_idx] = normalize_per_feature(X_train[:, numerical_idx], X_test[:, numerical_idx])

    return X_train, y_train, X_test, y_test, n_classes, categorical_idx, numerical_idx

############################
### get mushroom dataset ###
############################

def mushroom(current_fold=None, n_splits=None):
    
    # loading train and test split
    data_path = data_dir + 'mushroom'
    if n_splits == None:
        X_train, X_test = np.load(os.path.join(data_path, 'X_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'X_test.npy'), allow_pickle=True)
        y_train, y_test = np.load(os.path.join(data_path, 'y_train.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_test.npy'), allow_pickle=True)
    else:
        X_all, y_all = np.load(os.path.join(data_path, 'X_all.npy'), allow_pickle=True), np.load(os.path.join(data_path, 'y_all.npy'), allow_pickle=True)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        train_index, test_index = list(kf.split(X_all))[current_fold]
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]
        
    # some meta-data
    categorical_idx = [i for i in range(22)]
    numerical_idx = []
    n_classes = 2
    
    # standardization of numerical features
    X_train[:, numerical_idx], X_test[:, numerical_idx] = normalize_per_feature(X_train[:, numerical_idx], X_test[:, numerical_idx])

    return X_train, y_train, X_test, y_test, n_classes, categorical_idx, numerical_idx









    
if __name__ == '__main__':
    save_adult()
    save_credit()
    save_mammal()
    save_bank()
    save_mushroom()

all_datasets_dict = {
    'adult': adult,
    'credit': credit,
    'mammal': mammal,
    'bank': bank,
    'mushroom': mushroom,
}

dataset_names_dict = {
    'adult': 'adult',
    'credit': 'credit',
    'mammal': 'mammal',
    'bank': 'bank',
    'mushroom': 'mushroom',
}
