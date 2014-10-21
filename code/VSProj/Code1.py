import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error

def cv_optimize(X_train, Y_train, clf):
    list_C = np.logspace(-4, 3, num=8)
    list_epsilon = np.logspace(-7, 0, num=8)
    list_epsilon = [0.1]

    parameters = {"C": list_C, "epsilon": list_epsilon}
    gs = GridSearchCV(clf, param_grid = parameters, cv = 10, n_jobs = 8, scoring = 'mean_absolute_error')
    gs.fit(X_train, Y_train)
    print "gs.best_params_ = {0}, gs.best_score_ = {1}".format(gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def fit_clf(X_train, Y_train):
    clf = svm.SVR(kernel = 'linear', C = bp['C'], epsilon = bp['epsilon'])
    clf, bp, bs = cv_optimize(X_train, Y_train)
    
    clf.fit(X_train, Y_train)
    return clf


if __name__ == '__main__':

    df_train_X = pd.read_csv("..\\..\\data\\X_train.csv")
    df_train_Y = pd.read_csv("..\\..\\data\\y_train.csv")
    df_test = pd.read_csv("..\\..\\data\\X_test.csv")

    df_output = pd.DataFrame(df_test[['Id']])

    print df_train_X.shape
    print df_test.shape

    X_train = df_train_X.values[:, 1:]
    X_test = df_test.values[:, 1:]

    yCols = df_train_Y.columns.values.tolist()

    for colName in yCols[1:]:
        Y_train = df_train_Y[colName].values

        clf = fit_clf(X_train, Y_train)
        print clf.score(X_train, Y_train)
        predicted_test = clf.predict(X_test)
        df_output[colName] = predicted_test

    df_output.to_csv("..\\..\\..\\data\\output.csv", index = False)









