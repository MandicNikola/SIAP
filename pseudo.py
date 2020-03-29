import numpy as np
import pandas as pd


class Pseudo:
    """
    Pseudo class for getting labels for unlabeled data.
    Uses model sent as param in constructor, also unlabeled_data
    """

    def __init__(self, model, unlabeled_data, label_columns):
        self.model = model
        self.unlabeled_data = unlabeled_data.copy()
        self.label_columns = label_columns

    def pseudo_data(self):
        y_pseudo_labels = self.model.predict(np.array(self.unlabeled_data))
        self.unlabeled_data[self.label_columns] = y_pseudo_labels
        return self.unlabeled_data

    def concat_data(self, data):
        return pd.concat([data, self.unlabeled_data])

