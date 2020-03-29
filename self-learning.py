import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from metrics import Metrics

from pseudo import Pseudo

# models for testing self-learning
models = [SVC(kernel='linear', probability=True), KNeighborsClassifier(n_neighbors=1), LogisticRegression(random_state=0, max_iter=1000), GaussianNB()]

# labeled data
labeled_data = pd.read_csv("D:\\Nikola Faks\\SIAP\\diabetess.csv")
# unlabeled data
unlabeled_data = pd.read_csv("D:\\Nikola Faks\\SIAP\\diabetes.csv")

# get labels from columns
y = labeled_data[['Outcome']]
# data for testing, remove label `Outcome` from data
x = labeled_data[list(filter(lambda column: column != 'Outcome', list(labeled_data.columns)))]

# unlabeled data
x_unlabeled = unlabeled_data[list(filter(lambda column: column != 'Outcome', list(unlabeled_data.columns)))]

x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.3, random_state=42)

print('Train shape: ', x_train.shape, y_train.shape)
print('Test shape: ', x_test.shape, y_test.shape)

for model in models:
    print('Model', type(model).__name__)
    print('Without pseudo-labeled data')
    print('*********************************')
    model.fit(x_train, y_train.ravel())
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    Metrics.metrics_info(y_train, y_train_pred, 'Train')
    Metrics.metrics_info(y_test, y_test_pred, 'Validation')
    print('*********************************')
    print('With pseudo-labeled data')
    pseudo = Pseudo(model, x_unlabeled, 'Outcome')
    pseudo.pseudo_data()
    first_train_data = pd.DataFrame(np.hstack((x_train, y_train)), columns=list(unlabeled_data.columns))
    merged_data = pseudo.concat_data(first_train_data)
    y = merged_data[['Outcome']]
    x = merged_data[list(filter(lambda column: column != 'Outcome', list(labeled_data.columns)))]
    model.fit(np.array(x), np.array(y).ravel())
    y_train_pred = model.predict(x)
    y_test_pred = model.predict(x_test)
    Metrics.metrics_info(y, y_train_pred, 'Train')
    Metrics.metrics_info(y_test, y_test_pred, 'Validation')
    print('*********************************')



