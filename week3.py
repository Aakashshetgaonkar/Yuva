from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris=load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species']= pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())
print(df.isna().sum())
df.to_csv('iris.csv')

target=df['species']
print(target.iloc[:5])
df=df.drop('species',axis=1)

scaler=StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled= pd.DataFrame(X_scaled,columns=df.columns)


def length_width(df):
    plt.figure(figsize=(10,6))
    color=['red','blue','green']
    labels=['🔴:sertosa','🔵:versicolor','🟢:verginica']

    plt.subplot(221)
    plt.title('sepal length vs sepal width')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    for i in range(3):
        j=i*50
        plt.scatter(df['sepal length (cm)'][j:j+50],df['sepal width (cm)'][j:j+50],c=color[i],label=labels[i])
    plt.legend()


    plt.subplot(223)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    for i in range(3):
        j=i*50
        plt.scatter(df['sepal length (cm)'][j:j+50],df['sepal width (cm)'][j:j+50],c='k')

    plt.subplot(222)
    plt.title('petal length vs petal width')
    plt.xlabel('petal lenth')
    plt.ylabel('petal width')
    for i in range(3):
        j=i*50
        plt.scatter(df['petal length (cm)'][j:j+50],df['petal width (cm)'][j:j+50],c=color[i],label=labels[i])
    plt.legend()

    plt.subplot(224)
    plt.xlabel('petal lenth')
    plt.ylabel('petal width')
    for i in range(3):
        j=i*50
        plt.scatter(df['petal length (cm)'][j:j+50],df['petal width (cm)'][j:j+50],c='k')
    plt.show()

length_width(df)

def plot_inertia(lis,k):
    plt.figure(1)
    plt.plot(k,lis,marker='o')
    plt.xlabel('no. of cluster')
    plt.ylabel('inertia')
    plt.show()

def plot_silhoutte(lis,k):
    plt.figure(2)
    plt.plot(k,lis,marker='o')
    plt.xlabel('no. of cluster')
    plt.ylabel('silhoutte score')
    plt.show()


inertias=[]
silhouette_lis=[]
models=[]
for i in range(2,6):
    model=KMeans(n_clusters=i,random_state=23,n_init=12,init='k-means++')
    model.fit(X_scaled)
    models.append(model)
    inertias.append(model.inertia_)
    silhouette_lis.append(silhouette_score(X_scaled,model.labels_))


plot_inertia(inertias,range(2,6))
plot_silhoutte(silhouette_lis,range(2,6))


best_index = silhouette_lis.index(max(silhouette_lis))
print(best_index)
best_k = list(range(2,6))[best_index]
best_model = models[best_index]

print("\nBest K:", best_k)
print("Best Silhouette Score:",silhouette_lis[best_index])
print("Best Inertia:",best_model.inertia_)

clusters = best_model.predict(X_scaled)
centers = best_model.cluster_centers_

plt.figure()
plt.subplot(121)
plt.scatter(X_scaled['petal length (cm)'],X_scaled['petal width (cm)'],c=clusters,cmap='viridis',s=60)
plt.scatter(centers[:, 2],centers[:, 3],marker='X',s=200,label='Centroids')
plt.xlabel("Petal Length (Standardized)")
plt.ylabel("Petal Width (Standardized)")
plt.title(f"K-Means Clustering (K={best_k})")
plt.legend()

plt.subplot(122)
plt.scatter(X_scaled['sepal length (cm)'],X_scaled['sepal width (cm)'],c=clusters,cmap='viridis',s=60)
plt.scatter(centers[:, 0],centers[:, 1],marker='X',s=200,label='Centroids')
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Setal Width (Standardized)")
plt.title(f"K-Means Clustering (K={best_k})")
plt.legend()
plt.show()
