import numpy as np

def sliding(img,window=[6,6]):
    out=np.array([])
    for i in range(3):
        s=img.shape
        row=s[1]
        col=s[2]
        col_extent = col - window[1] + 1
        row_extent = row - window[0] + 1
        start_idx = np.arange(window[0])[:,None]*col + np.arange(window[1])
        offset_idx = np.arange(row_extent)[:,None]*col + np.arange(col_extent)
        if len(out)==0:
            out=np.take (img[i],start_idx.ravel()[:,None] + offset_idx.ravel())
        else:
            out=np.append(out,np.take (img[i],start_idx.ravel()[:,None] + offset_idx.ravel()),axis=0)
    return out

def triangleDist(patches,centroids,verbose=False):
    ret=[]
    c2=np.power(centroids,2).sum(1)
    last=0
    for i in range(0,len(patches),1024):
        if verbose:
            print 'mapping',i
        last=np.minimum(i+1024,len(patches))
        x2=np.power(patches[i:last],2).sum(1)
        xc=patches[i:last].dot(centroids.T)
        dist=np.sqrt(-2*xc+x2[:,None]+c2)
        u=dist.mean(1)
        ret.extend(np.maximum(-dist+u[:,None],0).tolist())
    return np.array(ret)

def kmeans(X_train,numCentroids=1600,rfSize=6):
    whitening=True
    numPatches = 400000
    CIFAR_DIM=[32,32,3]
    
    #create unsurpervised data
    patches=[]
    for i in range(numPatches):
        if(np.mod(i,10000) == 0):
            print "sampling for Kmeans",i,"/",numPatches
        start_r=np.random.randint(CIFAR_DIM[0]-rfSize)
        start_c=np.random.randint(CIFAR_DIM[1]-rfSize)
        patch=np.array([])
        img=X_train[np.mod(i,X_train.shape[0])]
        for layer in img:
            patch=np.append(patch,layer[start_r:start_r+rfSize].T[start_c:start_c+rfSize].T.ravel())
        patches.append(patch)
    patches=np.array(patches)
    #normalize patches
    patches=(patches-patches.mean(1)[:,None])/np.sqrt(patches.var(1)+10)[:,None]
    
    del X_train
    
    print "whitening"
    M=patches.mean(0)
    [D,V]=np.linalg.eig(np.cov(patches,rowvar=0))
    
    P = V.dot(np.diag(np.sqrt(1/(D + 0.1)))).dot(V.T)
    patches = patches.dot(P)
    
    centroids=np.random.randn(numCentroids,patches.shape[1])*.1
    num_iters=50
    batch_size=1024#CSIL do not have enough memory, dam
    for ite in range(num_iters):
        print "kmeans iters",ite+1,"/",num_iters
        hf_c2_sum=.5*np.power(centroids,2).sum(1)
        counts=np.zeros(numCentroids)
        summation=np.zeros_like(centroids)
        for i in range(0,len(patches),batch_size):
            last_i=min(i+batch_size,len(patches))
            idx=np.argmax(patches[i:last_i].dot(centroids.T)                  -hf_c2_sum.T,                  axis=1)        
            S=np.zeros([last_i-i,numCentroids])
            S[range(last_i-i),
              np.argmax(patches[i:last_i].dot(centroids.T)-hf_c2_sum.T
                        ,axis=1)]=1
            summation+=S.T.dot(patches[i:last_i])
            counts+=S.sum(0)
        centroids=summation/counts[:,None]
        centroids[counts==0]=0 # some centroids didn't get members, divide by zero

    colmean=np.zeros(len(centroids))
    for i in range(0,len(patches),1024):
        last=np.minimum(i+1024,len(patches))-1
        colmean+=triangleDist(patches[i:last],centroids).sum(0)
    colmean/=len(patches)

    cov=np.zeros([len(centroids),len(centroids)])
    for i in range(0,len(patches),1024):
        last=np.minimum(i+1024,len(patches))-1
        dist=triangleDist(patches[i:last],centroids)-colmean

        cov+=dist.T.dot(dist)
    
    [D,V]=np.linalg.eig(cov)
    Pback = V.dot(np.diag(np.sqrt(1/(D + 0.1)))).dot(V.T)

    return centroids,P,M,Pback,colmean
        #the thing is, they will stay zero forever

def avg_pool(x,stride=3):
    """Return maximum in groups of 2x2 for a N,h,w image"""
    N,h,w = x.shape
    x = x.reshape(N,h/stride,stride,w/stride,stride).swapaxes(2,3).reshape(N,h/stride,w/stride,stride**2)
    return np.mean(x,axis=3)   

def extract_features(X_train,centroids,P,M,Pback,colmean,
        selected_feats=None,rfSize=6):
    trainXC=[]
    idx=0
    CIFAR_DIM=[32,32,3]
    for img in X_train:
        idx+=1
        if not np.mod(idx,1000):
            print "extract features",idx,'/',len(X_train)
        patches=sliding(img,[rfSize,rfSize]).T
        #normalize
        patches=(patches-patches.mean(1)[:,None])/(np.sqrt(patches.var(1)+10)[:,None])
        #map to feature space
        patches=(patches-M).dot(P)
        #calculate distance using x2-2xc+c2
        x2=np.power(patches,2).sum(1)
        c2=np.power(centroids,2).sum(1)
        xc=patches.dot(centroids.T)

        dist=np.sqrt(-2*xc+x2[:,None]+c2)
        u=dist.mean(1)
        patches=np.maximum(-dist+u[:,None],0)
       # if selected_feats is None:
       #     patches=(patches-colmean).dot(Pback.T)
       # else:
       #     patches=(patches-colmean).dot(Pback[:selected_feats].T)

        rs=CIFAR_DIM[0]-rfSize+1
        cs=CIFAR_DIM[1]-rfSize+1
        patches=np.reshape(patches,[rs,cs,-1])
        q=[]
        q.append(patches[0:rs/2,0:cs/2].sum(0).sum(0))
        q.append(patches[0:rs/2,cs/2:cs-1].sum(0).sum(0))
        q.append(patches[rs/2:rs-1,0:cs/2].sum(0).sum(0))
        q.append(patches[rs/2:rs-1,cs/2:cs-1].sum(0).sum(0))
        q=np.array(q).ravel()
        trainXC.append((q-q.mean())/(np.sqrt(q.var()+.01)))
    return np.array(trainXC)
