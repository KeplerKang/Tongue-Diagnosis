import pickle
import numpy as np
import glob
from sklearn.decomposition import PCA


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


filelist = glob.glob('cifar-10-batches-py/data*')
for i, name in enumerate(filelist):
    file = unpickle(name)
    data = file['data'.encode()]
    labels = file['labels'.encode()]
    if i == 0:
        alldata = data
        alllabels = labels
    else:
        alldata = np.concatenate([alldata, data])
        alllabels = np.concatenate([alllabels, labels])

testfile = unpickle('cifar-10-batches-py/test_batch')
testdata = testfile['data'.encode()]
testlabels = np.array(testfile['labels'.encode()])

# 保留数据的前两个主成分
pca = PCA(n_components=1000)
pca.fit(alldata)
# 将数据变换到前两个主成分的方向上
train_pca = pca.transform(alldata)
test_pca = pca.transform(testdata)

if __name__ == '__main__':
    print(alllabels.shape, alldata.shape)
    print("Reduced shape: {}".format(str(train_pca.shape)))  # Reduced shape: (50000, 2)
    print(testlabels.shape, testdata.shape)
    print("Reduced shape: {}".format(str(test_pca.shape)))
