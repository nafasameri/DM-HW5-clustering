from functions import *

X = read_data()
X = normalize(X)


print('===================== kmeans =====================')
kmeans(X)

print('===================== dbscan =====================')
dbscan(X)

print('===================== Setting MinPts and epsilon parameters =====================')
setMinPtsEpsilonParameters(X)
