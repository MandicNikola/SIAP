from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = pd.read_csv("D:\\Nikola Faks\\SIAP\\combined_diabetes_csv.csv")

# get labels from columns
y = data[['Outcome']]
# data for testing, remove label `Outcome` from data
x = data[list(filter(lambda column: column != 'Outcome', list(data.columns)))]

# K-means classification
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(np.array(x))
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()