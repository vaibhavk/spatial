import numpy as np
import pandas as pd
import kmeans2
import matplotlib
import matplotlib.pyplot as plt

data=pd.read_csv('Accident_Information.csv',low_memory=False, nrows=10000)
data.dropna(inplace=True)

f=data[['Latitude','Longitude']]

x=f[pd.to_numeric(f['Latitude'],errors='coerce').notnull()]
x=x[pd.to_numeric(x['Longitude'],errors='coerce').notnull()]
x=np.array(x)

a,b=kmeans2.kmeans(x,4,2)
distances=b[:,1]
#print(distances)
error=np.sum(distances)
print(error)

fig=plt.figure()
rect=[0.1,0.1,0.8,0.8]
scatterMarkers=['s','o','^','8','p','d','v','h','>','<']
axprops=dict(xticks=[],yticks=[])
ax1=fig.add_axes(rect,label='ax1',frameon=False)

for i in range(4):
    ptsInCurrCluster=x[np.nonzero(b[:,0]==i)[0],:]
    markerstyle=scatterMarkers[i%len(scatterMarkers)]
    ax1.scatter(ptsInCurrCluster[:,0],ptsInCurrCluster[:,1],marker=markerstyle,s=30)

ax1.scatter(a[:,0],a[:,1],marker='+',s=300)

plt.show()
