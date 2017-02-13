# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:39:25 2017

@author: Maria Osipenko

Principal expectile algorithm from:
    N. Tran, M. Osipenko, W. Haerdle (2014):
        Principal Component Analysis in an Asymmetric Norm, arXiv:1401.3229


Reference list:
     
[1] Principal Component Analysis in an Asymmetric Norm, N. Tran, M. Osipenko, W. Haerdle, arXiv:1401.3229

[2]  W.K Newey and J.L. Powell. Asymmetric least squares estimation and testing.
Econometrica: Journal of the Econometric Society, pages 819–847, 1987.


Summary:
    Principal component analysis (PCA) is a widely used dimension reduction tool
in the analysis of many kind of high-dimensional data. However, in many of the
above applications, one is interested in capturing the tail of the data rather than the
mean. 
    Pricipal expectile algorithm is an analogue of PCA for expectiles.
    Expectiles are tail measures analogue quantiles. Both quantiles and expectiles
can be seen as minimizers of an asymmetric norm:
    
    m_tau(X,alpha) = argmin E ||X - m ||_tau,alpha
    
with ||x||_tau,alpha = sum_j(|x_j|^alpha) ((1-tau)1(x_j<0) + tau 1(x_j>0)). If 
alpha = 1 we have the definition of tau-quantile, if alpha = 2 we have tau-expectile.
While a quantile is a generalization of the median, an expectile generalizes the mean.
    
    Principal expectile algorithm is a dimension reduction tool for such multivariate 
expectiles. It is maximizes the tau-variance of the projection, see [1] for 
further information. 
    The computation is based on iterative least squares from [2]. 


"""

import numpy as np
import matplotlib.pyplot as plt

#%%
#create a functional library for computing principal expectile
def mhat(yvec,lab, tau=0.5):
    """Computes the tau-expectile of a sequence of numbers in R
        with a given weight vector lab    
    """
    
    n = len(yvec)
    nplus=sum([x for x in lab if x>0])
    nminus = -sum([x for x in lab if x<0])
    sumplus = sum([yvec[i] for i in range(n) if lab[i]>0])
    summinus = sum([yvec[i] for i in range(n) if lab[i]<0])
    top = tau*sumplus + (1-tau)*summinus
    bottom = tau*nplus + (1-tau)*nminus
    return(top/bottom)

#%%
def expectile(yvec, tau):
    """ iteratively find the tau-expectile of a sequence of numbers in R
    """
    
    yvec=np.array(yvec)
    n=len(yvec)
    lab=np.copysign(1,(np.random.uniform(0,1,n)-0.5))
    change = True
    while change:
        change = False
        ehat = mhat(yvec,lab,tau)
        newlab = np.copysign(1,yvec-ehat)
        change = np.not_equal(lab, newlab).any()
        lab = np.copy(newlab)

    return([ehat,lab])
    

#%%
def psihat(ymat,lab,tau):
    """ returns first eigenvector psi given the partition
    """
    
    n,p = len(ymat), len(ymat[0])
    m=[]
    for i in range(p):
        yvec = ymat.transpose()[i,]
        m.append(mhat(yvec,lab,tau))
        
    #m = np.array(m)    
    yplus = (ymat -m)[lab>0]
    yminus =(ymat -m)[lab<0]
    cplus = tau/n*np.transpose(yplus).dot(yplus)
    cminus = (1-tau)/n*np.transpose(yminus).dot(yminus)
    c = cplus + cminus
    eival,eivec=np.linalg.eig(c)
    v = eivec[:, eival.argmax()]
    return(v)

#%%      
def expdir(ymat,psi,tau):
    """returns the weights of directional expectile
    """
    
    obs = psi.dot(np.transpose(ymat))
    ehat = expectile(obs,tau)[0]
    diff=obs-ehat
    lab=np.copysign(1,diff)
    return(lab)
    

#%%
def init(n,ymat):
    """ initialize the labels at mean
    """
    
    cymat = np.dot(ymat.transpose(),ymat)
    eival,eivec=np.linalg.eig(cymat)
    v = eivec[:, eival.argmax()]
    labini = expdir(ymat,v,0.5)
    return(labini)
    

#%%    
def prdir(ymat,tau,iter_tol=10,reset_tol=50):
    """ find the principal direction by iteratively updating the psi and the weights
    """
    
    n=len(ymat)
    #initiate the weight vector at random/ at mean 
    lab=np.copysign(1,(np.random.uniform(0,1,n)-0.5))
    #lab = init(n,ymat)
    change,iter,reset= True,1,0
    while change:
        change = False
        psi = psihat(ymat,lab,tau)
        newlab=expdir(ymat,psi,tau)
        change = np.not_equal(lab, newlab).any()
        lab = np.copy(newlab)
        print ("Iteration %d" %iter)
        iter = iter+1
        #can get stuck in a local minimum. In this case reinitialize
        if iter>iter_tol:
            print("Reset %d" %reset)
            lab=np.copysign(1,(np.random.uniform(0,1,n)-0.5))
            iter = 1
            reset = reset+1
            change = True
            if reset>reset_tol:
                change = False
                print ("The principal expectile algorithm did not converge!")
                  
    return([psi,lab])

#%%
def compare_basis(basis_fix,basis_old):
    """project basis_old on the span of basis_fix for comparison
    """
    
    ones_basis_old = np.vstack((np.ones(len(basis_old)),basis_old)).T
    coefs = np.linalg.lstsq(ones_basis_old,basis_fix)[0]
    basis_new = np.dot(ones_basis_old,coefs)
    return(basis_new)
    
#%%
# example:
# generate a sample from a mixture of distributions
np.random.seed(1234)#set the seed
n,p,fvar1,fvar2,mix_prob = 1000,20,8,4,0.8 #dimensions, factor variance, mixing probability
scr1 = fvar1*np.random.randn(n) #factor scores on f1
scr2 = fvar2*np.random.randn(n) #factor scores on f2
f1 = np.sin(np.arange(-np.pi,np.pi,2*np.pi/p)) #factors
f2 = np.cos(np.arange(-np.pi,np.pi,2*np.pi/p))
eps = 0.01*np.random.randn(n,p) #error term
#random draws for mixing
what_mix = (np.random.uniform(0.,1.,n)>mix_prob)*1
#construct a factorized sample from the mixture
yfactor = np.zeros((n,p),np.float)
for i in range(n):
    yfactor[i] = scr1[i]*f1*what_mix[i]+scr2[i]*f2*(1-what_mix[i])+eps[i]

#%%
#estimate the principal directions (prdir) for the expectile levels 0.99,0.5; 0.5 is equivalent to PCA
taus = [0.99,0.5]
#true prdir capturing the largest variation (for tau = 0.99 it is f1, for tau = 0.5 it is f2) 
f,nam = np.vstack((f1,f2)),["f1","f2"]
#plot parameters
cols=[["r","g"],["m","b"]]
lins=[["-","--"],["-","--"]]

for i in range(len(taus)):
    tau=taus[i]
    np.random.seed(1234+i)
    psi,_ = prdir(yfactor,tau,iter_tol=10) #estimator for prdir
    target_f,col,lin = f[i],cols[i],lins[i]
    ax1 = plt.subplot(111)
    true = ax1.plot(target_f,color=col[0],ls=lin[0],label = "true prdir for tau ="+str(tau)) #real basis
    est = ax1.plot(compare_basis(target_f,psi),color=col[1],ls=lin[1],label = "estimate for tau ="+str(tau)) #estimator
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)   
    
""" the principal expectile algorithm should find f1 as the direction of the 
major variation in the extremes while the PCA picks f2 as the first principal 
component 
"""
