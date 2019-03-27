import numpy as np
import math
from geopy.distance import vincenty
from geopy.distance import great_circle


def load_dataset(filename):
    datamat=[]
    fr=open(filename)
    for line in fr.readlines():
            x=line.strip()
            x=x.split('\t')
            f=list(map(float,x))
            datamat.append(f)
    
    return(datamat)


def dist_euclid(vec1,vec2):
	n=vec1.shape[0]
	#print(n)
	s=0.0
	for i in range(n):
		x=vec1[i]-vec2[i]
		y=math.pow(x,2)
		s=s+y
	h=math.sqrt(s)
	return(h)

def dist_vincenty(vec1,vec2):
	di=vincenty(vec1,vec2).kilometers
	return(di)

def dist_greatcircle(vec1,vec2):
	di=great_circle(vec1,vec2).kilometers
	return(di)




	

def rand_centroid(dataset,k):
	#n,m=dataset.shape[1]
	#centroids=np.array(np.zeros((k,2)))

	centroids=np.zeros((k,2))
	for j in range(2):
		min_j=min(dataset[:,j])
		range_j=float(max(dataset[:,j])-min_j)
		temp=min_j+range_j*np.random.rand(k,1)

		for i in range(k):
			centroids[i][j]=temp[i][0]


	return(centroids)



def kmeans(dataset,k,dist_measure):
	m=dataset.shape[0]
	cluster_assign=np.zeros((m,2))
	centroids=rand_centroid(dataset,k)
	cluster_changed=True
	while(cluster_changed):
		cluster_changed=False
		for i in range(m):
			min_dist=math.inf
			min_index=-1

			for j in range(k):

				if(dist_measure==1):
					dist_ji=dist_euclid(centroids[j,:],dataset[i,:])
				if(dist_measure==2):
					dist_ji=dist_vincenty(centroids[j,:],dataset[i,:])
				if(dist_measure==3):
					dist_ji=dist_greatcircle(centroids[j,:],dataset[i,:])

				if(dist_ji<min_dist):
					min_dist=dist_ji
					min_index=j
			if(cluster_assign[i,0]!=min_index):
				cluster_changed=True
			cluster_assign[i,:]=min_index,min_dist**2
		print(centroids)
		for cent in range(k):
			points_in_cluster=dataset[np.nonzero(cluster_assign[:,0]==cent)[0]]
			centroids[cent,:]=np.mean(points_in_cluster,axis=0)
	return centroids,cluster_assign











        