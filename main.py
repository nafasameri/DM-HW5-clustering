from functions import *

X = read_data()
X = normalize(X)


print('===================== decision tree =====================')
kmeans(X)

print('===================== gaussian =====================')
dbscan(X)
