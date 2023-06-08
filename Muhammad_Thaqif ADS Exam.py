#!/usr/bin/env python
# coding: utf-8

# # Section A 

# In[39]:


# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # 1) Display the number of attributes available in the dataset

# In[40]:


ds = pd.read_csv('exam_dataset.csv')
pd.set_option('display.max_columns', 500)
ds.head()


# # 2) The dimension number of this dataset

# In[41]:


ds.shape


# The dimension is 2 and for columns is 24 for attributes rows is 1470 for observation

# # 3) The average of these attributes: ‘Age’, ‘Monthly Income’ and ‘Years at Company’ with 2 decimal places

# In[42]:


ds.info()


# In[115]:


data = ds.iloc[:,[0,2,20]]
data.describe().round(2)


# In[113]:


# Average value of 'Age', 'Monthly Income' and 'Years at Company'

ds.iloc[:,[0,2,20]].mean().round(2) 


# # 4) The minimum and maximum ‘Monthly Income’

# In[44]:


minc = ds.MonthlyIncome
minc.describe().round(2)


# In[45]:


minc_min = minc.min()
minc_max = minc.max()

print(f'Minimum Monthly Income is {minc_min}')
print(f'Minimum Monthly Income is {minc_max}')


# # 5) Graphical visualization by plotting Histogram of ‘Monthly Income vs staff numbers’ 

# In[46]:


plt.hist(minc)
plt.title('Monthly Income Vs Staff Numbers')
plt.xlabel('Monthly Income')
plt.ylabel('Staff Numbers')
plt.show()


# # 6) Graphical Visualization of the distribution between ‘Year at Company’ and ‘Monthly Income’ using the scatter plot

# In[47]:


yac = ds.YearsAtCompany
plt.scatter(minc,yac, color='blue')
plt.title('Year at Company vs Monthly Income')
plt.ylabel('Year at Company')
plt.xlabel('Monthly Income')
plt.show()


# # 7) The correlation between ‘Years at Company’ and ‘Monthly Income’

# In[48]:


corr = np.corrcoef(yac,minc)
print(f'The correlation is {corr}')


# In[49]:


sns.heatmap(corr, annot=True, fmt='.4g')


# # 8) Based on your findings, discuss briefly:

# a) Range of monthly income at Company A is between 1009 and 19999

# b) Most and Least Frequent monthly income values at Company A:

# In[116]:


import statistics

minc2 = np.where(minc.value_counts()==1)
print(f'Most monthly income in comapany A is {statistics.mode(minc)} whith frequency of \
{minc.value_counts().max()}')
print(f'Least monthly income in comapany A is {list(minc.value_counts().index[minc2])} whith frequency of \
{minc.value_counts().min()}')


# c) observation on the distribution of monthly income values is based on the histogram Monthly Income vs Staff Numbers where most of employee has monthly income below that average monthly income is 6502.93

# d) Based on the scatter plot, there is no linear relationship between monthly income and employees’ individual years at the company with correlation coefficient 0.514.

# # Section B

# Classification using Naïve Bayes for: ‘Age’, ‘BusinessTravel’, ‘MonthlyIncome’ and ‘JobSatisfaction’ to predict ‘Attrition’.

# In[51]:


# import dataset
ds1 = ds.iloc[:,[0,1,2,3,23]]
ds1.head()


# In[52]:


# The relevant attributes as input and output

x = ds1.iloc[:, :-1]
y = ds1.iloc[:,4]


# In[53]:


# set unique dataset

i=0

for i in range(0,4):
    d = list(set(x.iloc[:,i].values))
    d.sort()
    print(i,d)


# In[54]:


# LabelEncoder to encode categorical data

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x.iloc[:,1] = labelencoder_x.fit_transform(x.iloc[:,1])


# In[55]:


# Split data into training and test sets with the appropriate proportions

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[57]:


# Normalized data using StandardScaler

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


# In[58]:


# Fiting the Naïve Bayes Classifier

from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(x_train,y_train)


# In[59]:


# Predict the results

y_pred=classifier.predict(x_test)


# In[63]:


# Results evaluation using confusion matrix and calculate the prediction accuracy

from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
score = accuracy_score(y_test,y_pred)
print(cm)
print(f'The prediction score is {score}')


# # Discuss the results and findings 

# Based on accuracy value, 82% of the time the model will make correct prediction. Based on the findings we can say that ‘Age’, ‘BusinessTravel’, ‘MonthlyIncome’ and ‘JobSatisfaction’ is the best predactors so far to predict company A ‘Attrition’ 

# # Section C

# Clustering comparison between K-Means and DBSCAN 

# # 1) K-Means clustering

# In[94]:


# Import necessary libraries

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# In[95]:


# import dataset

wh = pd.read_csv('clustering.csv')
wh.head()


# In[96]:


wh.info()


# In[97]:


wh.isnull().values.any() # check null value in the dataset


# In[98]:


wh1 = wh.iloc[:, [1,2]]
corr = wh1.corr()


# In[99]:


sns.heatmap(corr, square=True)


# In[100]:


# Feature scalling 

ss = StandardScaler()
X = wh1.values
X_scaled = ss.fit_transform(X)
X_scaled


# In[101]:


# Finding the number of clusters using the Elbow Method

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()


# In[102]:


# K-Means Clustering:

kmeans = KMeans(n_clusters=4, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X_scaled)
y_kmeans


# In[103]:


# Visualizing K-Means Clustering:

sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=["cluster-{}".format(x) for x in y_kmeans])
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='orange', label='Centroids')
plt.title('Clusters of Centroids')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()


# In[104]:


score = silhouette_score(X_scaled, kmeans.labels_, metric='euclidean')
print(score)


# The number of cluster determined using of the elbow method is 4. The silhouette score is 0.43 that is nearest to 0. This is show that the cluster is not densed and not seperated well and there was overlapping with each cluster that can refer in the scatter plot

# # 2) DBSCAN clustering

# In[106]:


# import necessary libraries

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


# In[107]:


# Graphical visualization (Knee Locator)

nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:,10], axis=0)
i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex',
direction='increasing', interp_method='polynomial')
fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")


# In[108]:


# Graphical visualization (Optimum Knee)

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")
plt.savefig("knee.png", dpi=300)
print(distances[knee.knee])


# In[112]:


# Graphical visualization (with optimum eps)

db = DBSCAN(eps=distances[knee.knee],min_samples=10).fit(X_scaled)
labels = db.labels_
fig = plt.figure(figsize=(5, 5))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=["cluster-{}".format(x) for x in labels])


# Based on the DBSCAN clustering we observed that thare have 2 clustering in the dataset. The green one is outlier that not include inside both clusters. DBSCAN works well in complex clustering

# 3) Comparison studies on the two techniques (K-Means and DBSCAN), with graphical visualization comparisonsBased on both K-Means and DBSCAN clsutering technique i can conclude that DBSCAN is the better clustering technique for this dataset 

# In[ ]:




