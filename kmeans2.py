import numpy as np
from math import asin, atan, atan2, cos, pi, sin, sqrt, tan, inf

it=0

EARTH_RADIUS = 6371.009

ELLIPSOIDS = {
    # model           major (km)   minor (km)     flattening
    'WGS-84':        (6378.137, 6356.7523142, 1 / 298.257223563),
    'GRS-80':        (6378.137, 6356.7523141, 1 / 298.257222101),
    'Airy (1830)':   (6377.563396, 6356.256909, 1 / 299.3249646),
    'Intl 1924':     (6378.388, 6356.911946, 1 / 297.0),
    'Clarke (1880)': (6378.249145, 6356.51486955, 1 / 293.465),
    'GRS-67':        (6378.1600, 6356.774719, 1 / 298.25)
}

ITERATIONS = 20

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
        y=x**2
        s+=y
        h=sqrt(s)
    return(h)

def dist_vincenty(vec1,vec2):
    lat1,lng1=vec1[0],vec1[1]
    lat2,lng2=vec2[0],vec2[1]

    major,minor,f=ELLIPSOIDS['WGS-84']

    delta_lng=lng2-lng1

    reduced_lat1=atan((1-f)*tan(lat1))
    reduced_lat2=atan((1-f)*tan(lat2))

    sin_reduced1,cos_reduced1=sin(reduced_lat1),cos(reduced_lat1)
    sin_reduced2,cos_reduced2=sin(reduced_lat2),cos(reduced_lat2)

    lambda_lng=delta_lng
    lambda_prime=2*pi

    iter_limit=ITERATIONS

    i=0

    while (i==0 or (abs(lambda_lng-lambda_prime)>10e-12 and i<=iter_limit)):
        i += 1

        sin_lambda_lng, cos_lambda_lng = sin(lambda_lng), cos(lambda_lng)
        sin_sigma = sqrt((cos_reduced2 * sin_lambda_lng) ** 2 +
                        (cos_reduced1 * sin_reduced2 -
                         sin_reduced1 * cos_reduced2 * cos_lambda_lng) ** 2)

        if sin_sigma==0:
            return 0

        cos_sigma=(sin_reduced1*sin_reduced2+cos_reduced1*cos_reduced2*cos_lambda_lng)

        sigma=atan2(sin_sigma,cos_sigma)

        sin_alpha=(cos_reduced1*cos_reduced2*sin_lambda_lng/sin_sigma)
        cos_sq_alpha=1-sin_alpha**2

        if cos_sq_alpha!=0:
            cos2_sigma_m=cos_sigma-2*(sin_reduced1*sin_reduced2/cos_sq_alpha)
        else:
            cos2_sigma_m = 0.0  # Equatorial line

        C=f/16. *cos_sq_alpha*(4+f*(4-3*cos_sq_alpha))

        lambda_prime=lambda_lng
        lambda_lng=(
            delta_lng + (1 - C) * f * sin_alpha * (
                sigma + C * sin_sigma * (
                    cos2_sigma_m + C * cos_sigma * (
                        -1 + 2 * cos2_sigma_m ** 2
                    ))))

    if i > iter_limit:
        raise ValueError("Vincenty formula failed to converge!")

    u_sq=cos_sq_alpha*(major**2 - minor**2)/minor**2

    A=1+u_sq/16384.*(4096+u_sq*(-768+u_sq*(320-175*u_sq)))

    B=u_sq/1024.*(256+u_sq*(-128+u_sq*(74-47*u_sq)))

    delta_sigma=(B*sin_sigma*(cos2_sigma_m+B/4.*(
            cos_sigma* (-1+2*cos2_sigma_m**2) 
                - B/6.*cos2_sigma_m*(-3+4*sin_sigma**2) *
                (-3+4*cos2_sigma_m**2)
        )))

    s=minor*A*(sigma-delta_sigma)
    return s

def dist_greatcircle(vec1,vec2):
    t1= pow(( cos(vec2[0])* sin(vec2[1]-vec1[1])),2)
    t2= pow(( cos(vec1[0])* sin(vec2[0])- sin(vec1[0])* cos(vec2[0])* cos(vec2[1]-vec1[1])),2)

    numerator= atan( sqrt(t1+t2))

    a= sin((vec1[0])*( pi/180))* sin((vec2[0])*( pi/180))
    b= cos((vec1[0])*( pi/180))* cos((vec2[0])*( pi/180))* cos((vec1[1]-vec2[1])*( pi/180))

    angle=numerator/abs((a+b))
    dist=angle*6371.0

    return dist

def rand_centroid(dataset,k):
    return dataset[:k]

def kmeans(dataset,k,dist_measure):
    m=dataset.shape[0]
    cluster_assign=np.zeros((m,2))
    centroids=rand_centroid(dataset,k)
    cluster_changed=True
    print('Processing now....\n')
    while(cluster_changed):
        cluster_changed=False
        for i in range(m):
            min_dist=inf
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
        global it
        print("Iteration",it)
        it+=1

        for cent in range(k):
            points_in_cluster=dataset[np.nonzero(cluster_assign[:,0]==cent)[0]]
            centroids[cent,:]=np.mean(points_in_cluster,axis=0)
    return centroids,cluster_assign
