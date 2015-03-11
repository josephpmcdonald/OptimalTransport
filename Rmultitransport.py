"""
Rmultitransport.py

NOTES
7/29/14: routine works well with the following parameters. Some issues with the MC plot, but Analytical works well
In alpha, eps = 0.0001, npoints = 10, L = 5
In main, eps = 1e-7

8/13/14: changing to allow for multiple centers
10/23/14: caps beta values to limit step size per iteration

10/27/14: alter multitransport.py to allow for regression between 

"""



from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.cluster.vq as vq

norm = np.linalg.norm
exp = math.exp
log = math.log
pi = math.pi
sqrt = math.sqrt
inner = np.inner

def minus(x,z):
    return [x[i]-z[i] for i in range(len(x))] 

def F(x,z,a):
    #xmzovera = [(x-z)/a for i in range(len(x))]

    #grad = F*(-0.5)*2*(x-z)/a**2 = F*(-(x-z))/a**2
    return exp(-0.5*norm((x-z)/a)**2)/(2*pi*a**2)

def kernel(X, z, a):
    #used for kernel density estimation. Note that the density used here is a Gaussian of variance a^2.
    d = len(X[0])
    n = len(X)

    return sum([(exp(-0.5*norm((x-z)/a)**2)/(a*sqrt(2*pi))**d)/n for x in X]) 

def alpha(X, Y, z):
    d = len(X[0])
    eps = 0.0001 #LIST THIS
    npoints = 10 #LIST THIS
    #L = 5

    a = (npoints/(n+m)*(1/(kernel(X,z,1)+eps) + 1/(kernel(Y, z, 1)+eps)))**(1/d)
    # 1/(2*pi)*sum([exp(-0.5*norm(y-z)**2) for y in Y])))**(1/d)
    # a = max([a,eps])
    
    a = max(a, 1)

    return a


if __name__ == "__main__":

    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    np.random.seed(7)
    #random.seed(0)
    plt.ion()
    plt.figure(1, figsize=(8,6))
    plt.show()
#    plt.figure(2)
#    plt.show()
    
    d = 2 # dimensions
    n = 500 # number of x samples
    m = n # number of y samples
    T = 10000 # number of iterations
    eps = 1e-7
    betaMax = 10000

    X = np.random.uniform(0,1,(n,2))
    #print X
#   XMC = np.array(X)
    XAn = np.array(X)
    mean = [0,0]
    cov = [[1,0],[0,1]]
    Y = np.random.multivariate_normal(mean,cov,n)
    #print Y[100]

    # Create lists for bandwidths of perturbation centers (aAn)
    # and associated coefficients (b,c,d)
    centers = 5
    aAn = [None]*centers
#    aMC = [None]*centers
    bsq = [None]*centers
    csq = [None]*centers
    dsq = [[None]*centers for row in range(centers)]
    
    # initialize array for sample density estimates (initially uniform density)
    # also Kullback-Leibler and log-likelihood
    rho = [1.0]*n
    KL = [None]*T
    LL = [None]*T

    for t in range(T):
        
#        XMCOld = np.array(XMC)
        XAnOld = np.array(XAn)

        #[Z, dist] = vq.kmeans(XAn,centers)
        Z = np.random.uniform(-4,4,(centers,2))
    
        #index = np.random.randint(0,m)
        #z = [random.uniform(-4,4),random.uniform(-4,4)]
        for i,z in enumerate(Z):
            aAn[i] = alpha(XAn, Y, z)
            bsq[i] = 1+1/(aAn[i]**2)
            csq[i] = 1/2+1/(aAn[i]**2)
            
        for i in range(centers):
            for j in range(centers):
                dsq[i][j] = 1/2+1/(2*aAn[i]**2)+1/(2*aAn[j]**2)
#            aMC = alpha(XMC, Y, z)

        GxAn = np.zeros(centers)
        GyAn = np.empty(centers)
        GAn = np.empty(centers)
        HAn = np.empty([centers, centers])

        for i,z in enumerate(Z):
            for x in XAn:
                GxAn[i] += F(x, z, aAn[i])

            GxAn[i] = GxAn[i]/n
            GyAn[i] = 1/(2*pi*aAn[i]**2)*1/bsq[i]*exp(norm(z)**2*(1/(2*aAn[i]**4*bsq[i])-1/(2*aAn[i]**2)))
            GAn[i] = GxAn[i] - GyAn[i]

        #Construct the Hessian H
        for i in range(centers):
            for j in range(i):
                zi = Z[i]
                zj = Z[j]
                z = zi/aAn[i]**2 + zj/aAn[j]**2
                HAn[i][j] = (1/(8*pi**2*aAn[i]**4*aAn[j]**4*dsq[i][j])*
                    (1/dsq[i][j]+inner(z/(2*dsq[i][j])-zi, z/(2*dsq[i][j])-zj))*
                    exp(norm(zi)**2/(4*dsq[i][j]*aAn[i]**4)+norm(zj)**2/(4*dsq[i][j]*aAn[j]**4)+
                    inner(zi,zj)/(2*dsq[i][j]*aAn[i]**2*aAn[j]**2)-
                    norm(zi)**2/(2*aAn[i]**2)-norm(zj)**2/(2*aAn[j]**2)))
                HAn[j][i] = HAn[i][j]

        for i,z in enumerate(Z):
            HAn[i][i] = (1/(8*pi**2*aAn[i]**8*csq[i])*
                (1/csq[i]+(1/(aAn[i]**2*csq[i])-1)**2*norm(z)**2)*
                exp(norm(z)**2*(1/(aAn[i]**4*csq[i])-1/aAn[i]**2)))
            #HAn[i][i] = (1/(2*pi*aAn**4))**2*1/(2*pi)*(pi/csq)*exp(norm(z)**2*(1/(aAn**4*csq)-1/aAn**2))*(1+(1/(aAn**2*csq)-1)**2*norm(z)**2)

        Hinv = np.linalg.inv(HAn)
        betaAn = -np.dot(Hinv,GAn)

#        if norm(betaAn) > betaMax:
#            b = betaMax/norm(betaAn)
#            betaAn = betaAn*b

        #betaAn = -GAn/HAn

#        GxMC = 0
#        GyMC = 0
#        HMC = 0

#        for x in XMC:
#            GxMC += F(x, z, aMC)
#        GxMC = GxMC/n

#        for y in Y:
#            f = F(y, z, aMC)
#            GyMC += f
#            HMC += (f**2)*(1/(aMC**4))*norm(y-z)**2
#        GyMC = GyMC/m
#        GMC = GxMC - GyMC
#        HMC = max(HMC/m, eps)
#        betaMC = -GMC/HMC
 
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 't =', t
        print 'Z =', Z
        print 'aAn =', aAn
        print 'GxAn =', GxAn
        print 'Gy analytical =', GyAn
        print 'H analytical  =', HAn
        print 'Analytical beta =', betaAn
#        print 'aMC =', aMC
#        print 'GxMC =', GxMC
#        print 'Gy with MC    =', GyMC
#        print 'H with MC     =', HMC


#        for i in range(len(XMC)):
#            x = XMC[i]
#            XMC[i] = x + betaMC*F(x, z, aMC)*(-(x-z)/aMC**2)
        
        # construct Jacobian matrix at each  and update the densities
        for k,x in enumerate(XAn):
            J00 = (1+sum([betaAn[i]*F(x, z, aAn[i])/(-aAn[i]**2) for i,z in enumerate(Z)])
                    +sum([betaAn[i]*F(x, z, aAn[i])*(x[0]-z[0])**2/aAn[i]**4 for i,z in enumerate(Z)]))
            J11 = (1+sum([betaAn[i]*F(x, z, aAn[i])/(-aAn[i]**2) for i, z in enumerate(Z)])
                    +sum([betaAn[i]*F(x, z, aAn[i])*(x[1]-z[1])**2/aAn[i]**4 for i,z in enumerate(Z)]))
            J01 = sum([betaAn[i]*F(x, z, aAn[i])*(x[0]-z[0])*(x[1]-z[1])/aAn[i]**4 for i,z in enumerate(Z)])
            # note: J10 = J01
            rho[k] = rho[k]/(J00*J11-J01**2)

        #KL[t] = 1/n*sum([log(rho[k]*2*pi)+norm(x)**2/2 for k,x in enumerate(XAn)])
        #LL[t] = 0

        # update the positions of each sample
        for k in range(len(XAn)):
            x = XAn[k]
            for i,z in enumerate(Z):
                XAn[k] += betaAn[i]*F(x, z, aAn[i])*(-(x-z)/aAn[i]**2)    

        if (t % 1 == 0):
            plt.figure(1)
            plt.clf()
            plt.suptitle('t = %i' %t)
            plt.subplot(221)
            plt.scatter(X[:,0],X[:,1])
            plt.subplot(222)
            plt.scatter(Y[:,0],Y[:,1])
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            plt.subplot(223)
#            plt.scatter(XMCOld[:,0],XMCOld[:,1])
#            plt.subplot(225)
            plt.scatter(XAnOld[:,0],XAnOld[:,1])
            plt.scatter(Z[:,0],Z[:,1],c='red')
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            ax = plt.subplot(224)
#            plt.title('MC')
#            plt.scatter(XMC[:,0],XMC[:,1])
            #plt.scatter(Z[:,0],Z[:,1],'or')
#            plt.subplot(226)
            plt.title('Analytical')
            plt.xlim((xmin, xmax))
            plt.ylim((ymin, ymax))
            plt.scatter(XAn[:,0],XAn[:,1])
            plt.scatter(Z[:,0],Z[:,1],c='red')
            for i,z in enumerate(Z):
                ax.add_patch(plt.Circle(z, radius = aAn[i], fill=False, color='g'))

            plt.draw()
            
#            plt.figure(2)
#            plt.clf()
#            plt.subplot(121)
#            plt.title('KL Divergence') 
#            plt.plot(KL[:t+1])
#            plt.draw()

            #plt.show(block=False)
            #text = raw_input()
            #plt.close()

        #if text == 'break':
        #    break



