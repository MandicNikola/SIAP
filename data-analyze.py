import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("D:\\Nikola Faks\\SIAP\\combined_diabetes_csv.csv")


# writing average values
for header in data:
    print(header, ': ', sum(data[header]) / len(data[header]))


