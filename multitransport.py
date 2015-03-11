"""
multitransport.py

NOTES
7/29/14: routine works well with the following parameters. Some issues with the MC plot, but Analytical works well
In alpha, eps = 0.0001, npoints = 10, L = 5
In main, eps = 1e-7

8/13/14: changing to allow for multiple centers


"""



from __future__ import division
import math
import numpy as np
#import argparse
import matplotlib.pyplot as plt
import random
#import scipy.cluster.vq as vq
import matplotlib.colors
import matplotlib.cm


norm = np.linalg.norm
exp = math.exp
log = math.log
pi = math.pi
sqrt = math.sqrt
inner = np.inner

def minus(x, z):
    return [x[i]-z[i] for i in range(len(x))] 

def F(x, z, a):
    #xmzovera = [(x-z)/a for i in range(len(x))]
    d = len(x)

    #grad = F*(-0.5)*2*(x-z)/a**2 = F*(-(x-z))/a**2
    return exp(-0.5*norm((x-z)/a)**2)/(2*pi*a**2)**(d/2)

def kernel(X, z, a):
    #used for kernel density estimation. Note that the density used here is a Gaussian of variance a^2.
    d = len(X[0])
    n = len(X)

    return sum([(exp(-0.5*norm((x-z)/a)**2)/(2*pi*a**2)**(d/2))/n for x in X]) 

def alpha(X, z):
    d = len(X[0])
    eps = 0.0001 #LIST THIS
    npoints = 10 #LIST THIS
    #L = 5

    #a = (npoints/(n+m)*(1/(kernel(X,z,1)+eps) + 1/(kernel(Y, z, 1)+eps)))**(1/d)
    a = (npoints/(n)*(1/(kernel(X,z,.1)+eps) + 1/(exp(-(norm(z)**2)/2)+eps)))**(1/d)
    # 1/(2*pi)*sum([exp(-0.5*norm(y-z)**2) for y in Y])))**(1/d)
    
    a = max(a, 1)

    return a

def I(d):
    if d == 1:
        return 1/2
    elif d == 0:
        return sqrt(pi)/2
    else:
        return (d-1)/2*I(d-2)

def S(d):
    if d == 0:
        return 2
    else:
        return 2*pi*V(d-1)

def V(d):
    if d == 0:
        return 1
    else:
        return S(d-1)/d

if __name__ == "__main__":

    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    #np.random.seed(4)
    #random.seed(0)
    plt.ion()
    plt.figure(1, figsize=(8,9))
    plt.show()
#    plt.figure(2)
#    plt.show()
    
    d = 2 # dimensions
    n = 500 # number of x samples
    m = n # number of y samples
    T = 10000 # number of iterations
    eps = 1e-7
    betaMax = 10000
    betaCap = True
#    nstar = 50 # number of unassigned X samples

    X = np.random.uniform(0,1,(n,d))
    #X = np.random.multivariate_normal([-1,-1],[[2,0],[0,2]],n)
    XMC = np.array(X)#
    XAn = np.array(X)
    mean = np.zeros(d)
    cov = np.eyes(d)
    Y = np.random.multivariate_normal(mean,cov,n)
#    Xstar = np.random.uniform(0,1,(nstar,d))
#    Xstarold = np.array(Xstar)


    color = [norm(x - [0.5,0.5]) for x in X]
    color = np.array(color)
    Norm = matplotlib.colors.Normalize(vmin = np.min(color), vmax = np.max(color), clip = False)

    # Create lists for bandwidths of perturbation centers (aAn)
    # and associated coefficients (b,c,d)
    centers = 1 
    aAn = [0.]*centers
##    aMC = [0.]*centers#
    B = [0.]*centers
    C = [0.]*centers
    D = [[0.]*centers for row in range(centers)]

    
    # initialize array for sample density estimates (initially uniform density)
    # also Kullback-Leibler, log-likelihood, and transport cost
    rho = [1.0]*n
    KL = [0.]*T
    LL = [0.]*T
    cost = [0.]*T

    for t in range(T):
        
        XAnOld = np.array(XAn)
##        XMCOld = np.array(XMC)#

        #Choose the centers of the perturbations here, 
        Z = np.random.uniform(-4,4,(centers,d))  #Uniformly distributed through box



#        Z = np.random.multivariate_normal(mean,cov,centers)  #Normally distributed about origin
        #Poor for 1 center. after 900 runs stagnates at a square, even with large cap on betamax 
        

        #Alternate between sample points and normal dist points
        #works for 1 center, np = 50, and betamax 100, though a little poor
#        indices = np.random.randint(n, size=centers) 
#        Z = np.empty([centers,d])
#        for i in range(centers):
#            choice = np.random.randint(2)
#            if choice == 0:
#                Z[i] = XAn[indices[i]]
#            else:
#                Z[i] = np.random.multivariate_normal(mean,cov)

        
        #[Z, dist] = vq.kmeans(XAn,centers) #Centers by kmeans
 
        #index = np.random.randint(0,m)
        for i,z in enumerate(Z):
            aAn[i] = alpha(XAn, Y, z)
##            aMC[i] = alpha(XMC, Y, z)
            B[i] = sqrt(1+1/(aAn[i]**2))
            C[i] = sqrt(1/2+1/(aAn[i]**2))
            
        for i in range(centers):
            for j in range(centers):
                D[i][j] = 1/2+1/(2*aAn[i]**2)+1/(2*aAn[j]**2)
##            aMC = alpha(XMC, Y, z)

        GxAn = np.zeros(centers)
        GyAn = np.empty(centers)
        GAn = np.empty(centers)
        HAn = np.empty([centers, centers])

##        GxMC = np.zeros(centers)#
##        GyMC = np.empty(centers)#
##        GMC = np.empty(centers)#
##        HMC = np.empty([centers, centers])#

        for i,z in enumerate(Z):
            for x in XAn:
                GxAn[i] += F(x, z, aAn[i])
##            for x in XMC:#
##                GxMC[i] += F(x, z, aMC[i])#

##            for y in Y:#
##                f = F(y, z, aMC[i])#
##                GyMC[i] += f#
            

            GxAn[i] = GxAn[i]/n
            GyAn[i] = 1/((2*pi)**(d/2)*aAn[i]**d)*1/(B[i]**d)*exp(norm(z)**2*(1/(2*aAn[i]**4*B[i]**2)-1/(2*aAn[i]**2)))
            GAn[i] = GxAn[i] - GyAn[i]

##            GxMC[i] = GxMC[i]/n#
##            GyMC[i] = GyMC[i]/n#
##            GMC[i] = GxMC[i] - GyMC[i]#

        #Construct the Hessian H
        for i in range(centers):
            for j in range(i):
                zi = Z[i]
                zj = Z[j]
                z = zi/aAn[i]**2 + zj/aAn[j]**2
                HAn[i][j] = (1/((2*pi)**(3*d/2)*aAn[i]**(d+2)*aAn[j]**(d+2))
                    *(S(d)/D[i][j]**(d+2)*I(d+1)+inner(z/(2*D[i][j]**2)-zi,z/(2*D[i][j]**2)-zj)*(pi/D[i][j]**2)**(d/2))
                    *exp(norm(zi)**2/(4*D[i][j]**2*aAn[i]**4)+norm(zj)**2/(4*D[i][j]**2*aAn[j]**4)
                    +inner(zi,zj)/(2*D[i][j]**2*aAn[i]**2*aAn[j]**2)
                    -norm(zi)**2/(2*aAn[i]**2)-norm(zj)**2/(2*aAn[j]**2)))
                HAn[j][i] = HAn[i][j]

##                for y in Y:#
##                    gradFi = F(y, zi, aMC[i])*-(y-zi)/(aMC[i]**2)#
##                    gradFj = F(y, zj, aMC[j])*-(y-zj)/(aMC[j]**2)#
##                    HMC[i][j] += inner(gradFi, gradFj)#
##
##                HMC[j][i] = HMC[i][j]#

        for i,z in enumerate(Z):
            HAn[i][i] = (1/((2*pi)**(3*d/2)*aAn[i]**(2*d+4))
                *exp((1/(aAn[i]**4*C[i]**2)-1/aAn[i]**2)*norm(z)**2)
                *1/(C[i]**d)*(S(d)/(C[i]**2)*I(d+1)+(1/(aAn[i]**2*C[i]**2)-1)**2*norm(z)**2*pi**(d/2)))

##            for y in Y:#
##                gradFi = F(y, z, aMC[i])*-(y-z)/(aMC[i]**2)#
##                HMC[i][i] += inner(gradFi, gradFi)#

            #HAn[i][i] = (1/(2*pi*aAn**4))**2*1/(2*pi)*(pi/csq)*exp(norm(z)**2*(1/(aAn**4*csq)-1/aAn**2))*(1+(1/(aAn**2*csq)-1)**2*norm(z)**2)

        Hinv = np.linalg.inv(HAn)
        betaAn = -np.dot(Hinv, GAn)

##        HMCinv = np.linalg.inv(HMC)#
##        betaMC = -np.dot(HMCinv, GMC)#


        if norm(betaAn) > betaMax and betaCap:
            b = betaMax/norm(betaAn)
            betaAn = betaAn*b

 
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 't =', t
        print 'Z =', Z
        print 'aAn =', aAn
        print 'GxAn =', GxAn
        print 'Gy analytical =', GyAn
        print 'H analytical  =', HAn
        print 'Analytical beta =', betaAn
        print 'Norm beta =', norm(betaAn)
#        print 'aMC =', aMC
#        print 'GxMC =', GxMC
#        print 'Gy with MC    =', GyMC
#        print 'H with MC     =', HMC


#        for i in range(len(XMC)):
#            x = XMC[i]
#            XMC[i] = x + betaMC*F(x, z, aMC)*(-(x-z)/aMC**2)
        
        # construct Jacobian matrix at each  and update the densities
        if d == 2:
            for k,x in enumerate(XAn):
                J00 = (1+sum([betaAn[i]*F(x, z, aAn[i])/(-aAn[i]**2) for i,z in enumerate(Z)])
                        +sum([betaAn[i]*F(x, z, aAn[i])*(x[0]-z[0])**2/aAn[i]**4 for i,z in enumerate(Z)]))
                J11 = (1+sum([betaAn[i]*F(x, z, aAn[i])/(-aAn[i]**2) for i,z in enumerate(Z)])
                        +sum([betaAn[i]*F(x, z, aAn[i])*(x[1]-z[1])**2/aAn[i]**4 for i,z in enumerate(Z)]))
                J01 = sum([betaAn[i]*F(x, z, aAn[i])*(x[0]-z[0])*(x[1]-z[1])/aAn[i]**4 for i,z in enumerate(Z)])
                # note: J10 = J01
                rho[k] = rho[k]/(J00*J11-J01**2)

        #KL[t] = 1/n*sum([log(rho[k]*2*pi)+norm(x)**2/2 for k,x in enumerate(XAn)])
        #LL[t] = 0

        # update the positions of each sample
        for k in range(n):
            x = XAn[k]
            for i,z in enumerate(Z):
                XAn[k] += betaAn[i]*F(x, z, aAn[i])*(-(x-z)/aAn[i]**2)

##            x = XMC[k]#
##            for i,z in enumerate(Z):#
##                XMC[k] += betaMC[i]*F(x, z, aMC[i])*(-(x-z)/aMC[i]**2)#

#        for k in range(nstar):
#            x = Xstar[k]
#            for i,z in enumerate(Z):
#                Xstar[k] += betaAn[i]*F(x, z, aAn[i])*(-(x-z)/aAn[i]**2)

        diff = XAn-X
        cost = sum([norm(row)**2 for row in diff])/n
        print 'Cost = ', cost

        if (t % 1 == 0): 
            plt.figure(1)
            plt.clf()
            plt.suptitle('t = %i' %t)
            plt.subplot(321)
            if d == 1:
                plt.hist(X, 50)
            else:
                plt.scatter(X[:,0],X[:,1],20,color,marker='o', cmap=matplotlib.cm.jet, norm=Norm)
#                plt.scatter(Xstarold[:,0],Xstarold[:,1],c='black')
            plt.subplot(322)
            if d == 1:
                plt.hist(Y, 50)
                xmin, xmax = plt.xlim()
            else:
                plt.scatter(Y[:,0],Y[:,1])
                xmin, xmax = plt.xlim()
                ymin, ymax = plt.ylim()
            az = plt.subplot(323)
            if d == 1:
                plt.hist(XAnOld, 50)
                plt.xlim((xmin, xmax))
            else:
                plt.scatter(XAnOld[:,0],XAnOld[:,1])
                plt.scatter(Z[:,0],Z[:,1],c='violet')
                plt.xlim((xmin, xmax))
                plt.ylim((ymin, ymax))
                for i,z in enumerate(Z):
                    az.add_patch(plt.Circle(z, radius = aAn[i], fill=False, color='g'))
            ax = plt.subplot(324)
            plt.title('Analytical')
            if d == 1:
                plt.hist(XAn, 50)
                plt.plot(Z,[0]*centers,'ro')
                plt.xlim((xmin, xmax))
            else:
                plt.xlim((xmin, xmax))
                plt.ylim((ymin, ymax))
                plt.scatter(XAn[:,0],XAn[:,1],20,color,marker='o',cmap=matplotlib.cm.jet,norm=Norm)
#                plt.scatter(Xstar[:,0],Xstar[:,1],c='black')
                plt.scatter(Z[:,0],Z[:,1],c='violet')
                for i,z in enumerate(Z):
                    ax.add_patch(plt.Circle(z, radius = aAn[i], fill=False, color='g'))
##            plt.subplot(325)
##            if d == 1:
##                plt.hist(XMC, 50)
##                plt.plot(Z,[0]*centers,'ro')
##            else:
##                plt.scatter(XMC[:,0],XMC[:,1])
            

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



