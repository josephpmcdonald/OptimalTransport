"""
transport.py

NOTES
7/29/14: routine works well with the following parameters. Some issues with the MC plot, but Analytical works well
In alpha, eps = 0.0001, npoints = 10, L = 5
In main, eps = 1e-7

"""



from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import random

norm = np.linalg.norm
exp = math.exp
pi = math.pi
sqrt = math.sqrt

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

    a = (npoints/(n+m)*(1/(kernel(X,z,1)+eps) + 1/(kernel(Y, z, 1)+eps)))**(1/d)     #1/(2*pi)*sum([exp(-0.5*norm(y-z)**2) for y in Y])))**(1/d)
    #a = max([a,eps])
    
    #a = min(a,L)

    return a


if __name__ == "__main__":

    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    np.random.seed(0)
    #random.seed(1)
    plt.ion()
    plt.figure(1, figsize=(8,9))
    plt.show()

    d = 2
    n = 500 # number of x samples
    m = n # number of y samples
    T = 10000 # number of iterations
    eps = 1e-7

    X = np.random.uniform(0,1,(n,2))
    #print X
    XMC = np.array(X)
    XAn = np.array(X)
    #X = Xnp.tolist()
    mean = [0,0]
    cov = [[1,0],[0,1]]
    Y = np.random.multivariate_normal(mean,cov,n)
#    Y = np.random.uniform(0,1,(n,2))
#    for i in range(len(Y)):
#        Y[i,0] = (1-Y[i,1])*(Y[i,0]-0.5)+0.5
    #Y = Ynp.tolist()*2
    #print Y[100]

    for t in range(T):
        
        XMCOld = np.array(XMC)
        XAnOld = np.array(XAn)
        #XAn = np.array(XMC)

        #index = np.random.randint(0,m)
        #z = Y[index]
        #z = [random.uniform(-4,4),random.uniform(-4,4)]
        z = np.random.uniform(-4,4,(1,2))

        aMC = alpha(XMC, Y, z)
        aAn = alpha(XAn, Y, z)
        bsq = 1+1/(aAn**2)
        csq = 1/2+1/(aAn**2)
        GxMC = 0
        GyMC = 0
        GxAn = 0
        HMC = 0
        for x in XMC:
            GxMC += F(x, z, aMC)

        GxMC = GxMC/n

        for x in XAn:
            GxAn += F(x, z, aAn)

        GxAn = GxAn/n

        for y in Y:
            f = F(y, z, aMC)
            GyMC += f
            HMC += (f**2)*(1/(aMC**4))*norm(y-z)**2

        GyMC = GyMC/m
        GMC = GxMC - GyMC

        #GyAn = 1/(2*pi)*(2*pi/bsq)*exp(norm(z)**2*(1/(2*aAn**4*bsq)-1/(2*aAn**2)))
        GyAn =1/(2*pi*aAn**2)*1/bsq*exp(norm(z)**2*(1/(2*aAn**4*bsq)-1/(2*aAn**2)))
        GAn = GxAn - GyAn

        HMC = max(HMC/m, eps)
        HAn =(1/(2*pi*aAn**4))**2*1/(2*pi)*(pi/csq)*exp(norm(z)**2*(1/(aAn**4*csq)-1/aAn**2))*(1/csq+(1/(aAn**2*csq)-1)**2*norm(z)**2)

        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 't =', t
        print 'z =', z
        print 'aMC =', aMC
        print 'aAn =', aAn
        print 'GxMC =', GxMC
        print 'GxAn =', GxAn
        print 'Gy with MC    =', GyMC
        print 'Gy analytical =', GyAn
        print 'H with MC     =', HMC
        print 'H analytical  =', HAn

        betaMC = -GMC/HMC
        betaAn = -GAn/HAn
        print 'Monte Carlo beta =', betaMC, ', Analytical beta =', betaAn

        for i in range(len(XMC)):
            x = XMC[i]
            XMC[i] = x + betaMC*F(x, z, aMC)*(-(x-z)/aMC**2)

        for i in range(len(XAn)):
            x = XAn[i]
            XAn[i] = x + betaAn*F(x, z, aAn)*(-(x-z)/aAn**2)

        if (t % 50 == 0):
            plt.clf()
            plt.suptitle('t = %i'% t)
            plt.subplot(321)
            plt.scatter(X[:,0],X[:,1])
            plt.subplot(322)
            plt.scatter(Y[:,0],Y[:,1])
            plt.subplot(323)
            plt.scatter(XMCOld[:,0],XMCOld[:,1])
            plt.subplot(325)
            plt.scatter(XAnOld[:,0],XAnOld[:,1])
            plt.subplot(324)
            plt.title('MC')
            plt.scatter(XMC[:,0],XMC[:,1])
            #plt.plot(z[0],z[1],'or')
            plt.subplot(326)
            plt.title('Analytical')
            plt.scatter(XAn[:,0],XAn[:,1])
            plt.draw()

            #text = raw_input()

#        if (text == 'next'):
#            break

#    X = np.array(XMC)
#    Y = np.random.uniform(-1,0,(n,2))
#
#    for t in range(T):
#
#        if (t % 50 == 0):
#            plt.subplot(321)
#            plt.scatter(X[:,0],X[:,1])
#            plt.subplot(322)
#            plt.scatter(Y[:,0],Y[:,1])
#            plt.subplot(323)
#            plt.scatter(XMC[:,0],XMC[:,1])
#            plt.subplot(325)
#            plt.scatter(XAn[:,0],XAn[:,1])
#            
#        index = np.random.randint(0,m)
#        z = Y[index]
#        #z1 = 
#        z = [random.uniform(-4,4),random.uniform(-4,4)]
#
#        aMC = alpha(XMC, Y, z)
#        aAn = alpha(XAn, Y, z)
#        bsq = 1+1/(aAn**2)
#        csq = 1/2+1/(aAn**2)
#        GxMC = 0
#        GyMC = 0
#        GxAn = 0
#        HMC = 0
#        for x in XMC:
#            GxMC += F(x, z, aMC)
#
#        GxMC = GxMC/n
#
#        for x in XAn:
#            GxAn += F(x, z, aAn)
#
#        GxAn = GxAn/n
#
#        for y in Y:
#            f = F(y, z, aMC)
#            GyMC += f
#            HMC += (f**2)*(1/(aMC**4))*norm(y-z)**2
#
#        GyMC = GyMC/m
#        GMC = GxMC - GyMC
#
#        #GyAn = 1/(2*pi)*(2*pi/bsq)*exp(norm(z)**2*(1/(2*aAn**4*bsq)-1/(2*aAn**2)))
#        GyAn = 1/bsq*exp(norm(z)**2*(1/(2*aAn**4*bsq)-1/(2*aAn**2)))
#        GAn = GxAn - GyAn
#
#        HMC = HMC/m
#        HAn = 1/(2*pi)*(pi/csq)*exp(norm(z)**2*(1/(aAn**4*csq)-1/aAn**2))*(1+(1/(aAn**2*csq)-1)**2*norm(z)**2)
#
#        print 'Gy with MC    =', GyMC
#        print 'Gy analytical =', GyAn
#        print 'H with MC     =', HMC
#        print 'H analytical  =', HAn
#
#        betaMC = -GMC/HMC
#        betaAn = -GAn/HAn
#        print 'Monte Carlo beta =', betaMC, ', Analytical beta =', betaAn
#
#        for i in range(len(XMC)):
#            x = XMC[i]
#            XMC[i] = x + betaMC*F(x, z, aMC)*(-(x-z)/aMC**2)
#
##        for i in range(len(XAn)):
##            x = XAn[i]
##            XAn[i] = x + betaAn*F(x, z, aAn)*(-(x-z)/aAn**2)
#
#        if (t % 50 == 0):
#            plt.subplot(324)
#            plt.title('MC')
#            plt.scatter(XMC[:,0],XMC[:,1])
#            plt.subplot(326)
#            plt.title('Analytical')
#            plt.scatter(XAn[:,0],XAn[:,1])
#            plt.show(block=False)
#            text = raw_input()
#            plt.close()

       



