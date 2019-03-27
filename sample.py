import numpy as np
import kmeans2
import matplotlib
import matplotlib.pyplot as plt

data=kmeans2.load_dataset('testSet.txt')
data=np.array(data)

a,b=kmeans2.kmeans(data,4,2)
distances=b[:,1]
print(distances)
error=np.sum(distances)
print(error)

fig=plt.figure()
rect=[0.1,0.1,0.8,0.8]
scatterMarkers=['s','o','^','8','p','d','v','h','>','<']
axprops=dict(xticks=[],yticks=[])
ax1=fig.add_axes(rect,label='ax1',frameon=False)

for i in range(4):
	ptsInCurrCluster=data[np.nonzero(b[:,0]==i)[0],:]
	markerstyle=scatterMarkers[i%len(scatterMarkers)]
	ax1.scatter(ptsInCurrCluster[:,0],ptsInCurrCluster[:,1],marker=markerstyle,s=30)

ax1.scatter(a[:,0],a[:,1],marker='+',s=300)

plt.show()









