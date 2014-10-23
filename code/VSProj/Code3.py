import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing

def cv_optimize(X_train, Y_train, clf):
    C_range = 10.0 ** np.arange(1, 4, 2)
    gamma_range = 10.0 ** np.arange(-7, -4)
    param_grid = dict(gamma=gamma_range, C=C_range)

    gs = GridSearchCV(clf, param_grid = param_grid, cv = 5, n_jobs = 16, verbose = 5)
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    clf = svm.SVC()
    clf, bp, bs = cv_optimize(X_train, Y_train, clf)    
    clf.fit(X_train, Y_train)
    return clf


if __name__ == '__main__':

    df_train_X = pd.read_csv("..\\..\\data\\X_train.csv")
    df_train_Y = pd.read_csv("..\\..\\data\\y_train.csv")
    df_test = pd.read_csv("..\\..\\data\\X_test.csv")

    df_output = pd.DataFrame({'Id': range(1954, 2791)})

    X_train = df_train_X.values[:, 1:]
    X_test = df_test.values[:, 1:]

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    yCols = df_train_Y.columns.values.tolist()

    for colName in yCols[1:]:
        Y_train = df_train_Y[colName].values

        clf = fit_clf(X_train, Y_train)
        predicted_train = clf.predict(X_train)
        predicted_test = clf.predict(X_test)

        df_train_Y[colName] = predicted_train

        df_output[colName] = predicted_test

    df_output.to_csv("..\\..\\data\\output.csv", index = False)

    df_train_Y.to_csv("..\\..\\data\\y_predict.csv", index = False)









