from sklearn.decomposition import PCA 
from data import *
X_scaled=alldata
# 保留数据的前两个主成分
pca = PCA(n_components=2)
pca.fit(X_scaled)
# 将数据变换到前两个主成分的方向上
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))     # Original shape: (50000, 3072)
print("Reduced shape: {}".format(str(X_pca.shape)))         # Reduced shape: (50000, 2)

print(X_pca[:10])
