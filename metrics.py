import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler


class Metrics:

    @staticmethod
    def metrics_info(y, y_pred, word='Train'):
        print(word, " accuracy: ", accuracy_score(y, y_pred))
        print("Precision ", word, ": ", precision_score(y, y_pred))
        print("Recall ", word, ": ", recall_score(y, y_pred))
        print("F measure: ", f1_score(y, y_pred))

    @staticmethod
    def csv_metrics(y, y_pred, classifier):
        return {
            "classifier": classifier,
            "accuracy": round(accuracy_score(y, y_pred) * 100, 2).__str__() + '%',
            "precision": round(precision_score(y, y_pred) * 100, 2).__str__() + '%',
            "recall": round(recall_score(y, y_pred) * 100, 2).__str__() + '%',
            "F measure": round(f1_score(y, y_pred) * 100, 2).__str__() + '%',
        }


class DataHelper:

    @staticmethod
    def min_max_normalization(data):
        return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=list(data.columns))

    @staticmethod
    def root_square_columns(data, columns_to_be_squared_root):
        data[columns_to_be_squared_root] = data[columns_to_be_squared_root]**(1/2)
        return data

    @staticmethod
    def get_x_y_from_data(data, label_column, rest_columns=[]):
        if len(rest_columns) == 0:
            y = data[label_column]
            x = data[list(filter(lambda column: column != label_column, list(data.columns)))]
            return x, y
        x = data[rest_columns]
        y = data[[label_column]]
        return x, y
