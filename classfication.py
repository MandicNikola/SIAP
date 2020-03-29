from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import SMOTE
from metrics import Metrics, DataHelper


data = pd.read_csv("D:\\Nikola Faks\\SIAP\\combined_diabetes_csv.csv")
# data = pd.read_csv("D:\\Nikola Faks\\SIAP\\diabetess.csv")
# SVC(kernel='linear', probability=True),
# used models for prediction
models = [SVC(kernel='linear', probability=True), KNeighborsClassifier(n_neighbors=1), LogisticRegression(random_state=0, max_iter=1000), GaussianNB()]

data_normalized = DataHelper.min_max_normalization(data)

# get x, y for data, need to put column
x, y = DataHelper.get_x_y_from_data(data, 'Outcome')
# x_normalized, y_normalized = DataHelper.get_x_y_from_data(data, 'Outcome')


x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2, random_state=42)
# x_train_normalized, x_test_normalized, y_train_normalized, y_test_normalized = train_test_split(np.array(x_normalized), np.array(y_normalized), test_size=0.2, random_state=42)

#
# SMOTE implementation
smt = SMOTE()
X_train, Y_train = smt.fit_sample(x_train, y_train)
# X_train_normalized, Y_train_normalized = smt.fit_sample(x_train, y_train)


for model in models:
    print('Model', type(model).__name__)
    print('Train shape: ', X_train.shape, Y_train.shape)
    model.fit(X_train, Y_train.ravel())
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(x_test)
    Metrics.metrics_info(Y_train, y_train_pred, 'train')
    Metrics.metrics_info(y_test_pred, y_test, 'validation')
    scores = cross_val_score(model, X_train, Y_train.ravel(), cv=5)
    print("K fold scores: ", scores)
    print('********************************************')
