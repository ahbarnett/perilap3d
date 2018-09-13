# 3D electrostatic triply-periodic dipole sum. python module, w/ numpy+numba.
# Alex Barnett, Sept 2018. Potential collab w/ Gabor Csanyi.

import numpy as np
from numpy import array,zeros,ones,eye,empty,random
from numpy.linalg import norm,cond
import scipy.linalg as la
from time import perf_counter as tic
import lap3dkernels as l3k

import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import myplotutils

class lap3d3p:
    """Laplace 3D Triply-periodic class. Needs: lap3dkernels, myplotutils

    Inputs:
      lat - 3x3 matrix, each row is a lattice vector (right-handed)
      verb - (integer, optional) verbosity: 0 silent, 1 text
 
    See: test_lap3d3p() below

    Issues: * annoying "self hell", not sure of struct vs class style
    """
    def __init__(self,lat,verb=1):
        # attributes supposed to list here (annoying; I don't). some defaults
        self.lat = lat
        self.ilat = la.inv(lat)        # cols are recip vectors
        self.nors = self.ilat.T.copy() # since each col gives a normal direc
        for i in range(3):
            self.nors[i] = self.nors[i]/norm(self.nors[i])   # normalize row
        if verb:
            print('lap3d3p init: cond # unit cell = %.3g'%(cond(lat)))
        self.badness = max(1.4,cond(lat))
        if self.badness>10:
            print('unit cell bad aspect ratio: will be slow & bad!')
        self.tol = None
        self.m = None    # colloc pts per face side
        self.scale = None   # linear factor to include as "near", ie 1+2*nei
        self.Np = None   # num proxy src pts
        self.Nc = None   # num surf colloc pts
        self.c=None      # face colloc pts 3(dir) * 2 * m^2(ind) * 3(xyz coord)
        self.cn=None     # face colloc nors
        self.p=None      # aux (proxy) src pts np * 3(xyz coord)
        self.pn=None     # aux src pt nors
        self.facename = [['L','R'],['B','F'],['U','D']]   # NB list of lists
        self.Q = None    # periodizing matrix (also there's self.Qfac*)
        self.facmeth = None
        
    def show(self):
        """plot the unit cell on a new figure. returns the axis handle
        """
        fig = pl.figure()
        ax = fig.gca(projection='3d',aspect='equal')  # equal fails here
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        l=1.0; ax.set_xlim(-l,l); ax.set_ylim(-l,l); ax.set_zlim(-l,l) # hack
        #ax.scatter(0,0,0,color='k')  # origin
        for d in range(3):  # plot wireframe of lattice cell lat on axes ax
            for i in range(2):
                for j in range(2):
                    r = array([[0,i,j],[1,i,j]]) - 1/2    # a line seg
                    r = np.roll(r,d,axis=1)   # cycle xyz coords
                    r = r.dot(self.lat)       # transform to lattice
                    ax.plot(r[:,0],r[:,1],r[:,2],color='k')

        if self.c is not None: # colloc pts. NB != fails when self.c is array
            for d in range(3):     
                n = 0.3*self.nors[d]      # (short) face normal for this dir
                for i in range(2):
                    r = self.c[d,i,:,:]
                    ax.scatter(r[:,0],r[:,1],r[:,2],s=1,color='b')
                    mr = np.mean(r,axis=0)
                    ax.text(mr[0],mr[1],mr[2],self.facename[d][i],color='b',
                            fontsize=20)
                    #ax.plot(mr[0]+[0,n[0]],mr[1]+[0,n[1]],mr[2]+[0,n[2]],lw=2,color='r')   # plain normals: no arrow
                    ar = myplotutils.Arrow3D(mr[0]+[0,n[0]],mr[1]+[0,n[1]],mr[2]+[0,n[2]],mutation_scale=20,lw=2,arrowstyle='-|>',color='xkcd:sky blue')
                    ax.add_artist(ar)
                    
        if self.p is not None:   # aux src pts
            ax.scatter(self.p[:,0],self.p[:,1],self.p[:,2],s=1,color='r')

        ax.set_title('unit cell, face names & normals, face colloc pts')
        pl.show()
        return ax
            
    def UCsurfpts(self,m,grid='u'):
        """return m^2 points & normals covering each of the 3*2 unit cell faces.
        Non-self inputs:
          m    - integer number of pts per face side
          grid (optional) - 'u' uniform, 'g' Gauss-Legendre, on each 1d edge
        Output is tuple of:
          pts,  float[3,2,m^2,3] - set of all points on the six faces
          nors, float[3,2,m^2,3] - set of corresponding +ve normal directions
        For each,
        0th index is direction, 1st is back/front, 2nd is point index, 3rd coord
        """
        pts = zeros([3,2,m**2,3])
        nors = zeros([3,2,m**2,3])
        if grid=='u':                    # build 1d colloc pts in [0,1]
            g = (np.arange(m)+1/2)/m            
        else:
            x,w = np.polynomial.legendre.leggauss(m)
            g = (x+1)/2
        for d in range(3):                  # normal direction: x,y, then z
            nors[d,:,:,:] = self.nors[d]    # b'cast; NB +ve facing.
            for i in range(2):              # back then front face
                xx,yy = np.meshgrid(g,g)
                r = np.vstack([i+0*xx.ravel(), xx.ravel(), yy.ravel()]).T - 1/2
                r = np.roll(r,d,axis=1)             # cycle xyz coords
                pts[d,i,:,:] = r.dot(self.lat)      # transform to lattice
        return pts, nors

    def precomp(self,tol=1e-3,proxytype='s',facmeth='s',verb=1):
        """Periodizing setup and Q factorization.
        Optionally can set:
          tol : tolerance, ie desired relative precision.  *** not yet used
          proxytype - 's' (charges) or 'd' (dipoles), for aux rep
          facmeth - Q factorization: 's' for SVD, 'q' for QR, 'p' for pivoted QR
                                     '' for don't factor, use LSQ solve in eval.
          verb - (integer) verbosity: 0 silent, 1 text, 2 and figs,...
        """
        self.tol=tol
        self.proxytype=proxytype
        digits = -np.log10(tol)
        self.m = int(6+1.3*digits*self.badness)     # colloc pts per face side
        self.scale = 1.8   # inflation factor for near summation, in (1,3]
        self.gap = (self.scale-1)/2   # gap from near box to std UC bdry = "nei"
        self.c,self.cn = self.UCsurfpts(self.m,grid='g')   # make colloc pts
        self.Nc = self.c.shape[1]
        P = int(5+1.6*digits*self.badness)    # aux pts per face side = "order"
        p,pn = self.UCsurfpts(P)      # make aux src pts, normals
        self.Np = 6*p.shape[2]   # num aux (proxy) pts
        self.p = self.scale * p.reshape([self.Np,3]) # gather all pts together
        self.pn = pn.reshape([self.Np,3])
        if verb:
            print('precomp: m=%d per face side, P=%d per proxy side'%(self.m,P))
        
        t0=tic()         # setup periodizing matrix & factorized inverse...
        self.fillQ()     # keeps Q around as an attribute
        if verb:
            print('Q size %d*%d, fill time %.3g s. factorizing...'%(self.Q.shape[0],self.Q.shape[1],tic()-t0))
        t0=tic(); self.facmeth=facmeth
        if facmeth=='q':
            self.QfacQ,R = la.qr(self.Q,mode='economic')   # no pivoting, fast
            self.invR = la.inv(R)
            if verb:
                print('QR + inv(R) time %.3g s, norm(invR)=%.3g'%(tic()-t0,norm(self.invR)))
        elif facmeth=='p':    # pivoted QR is 4x slower vs QR! (true in matlab)
            self.QfacQ,R,self.Qperm = la.qr(self.Q,mode='economic',pivoting=True)
            self.invR = la.inv(R)
            if verb:
                print('piv-QR + inv(R) time %.3g s, norm(invR)=%.3g'%(tic()-t0,norm(self.invR)))
            #print(self.invR)  # guess mult by invR will be bkw stable, good.
        else:        # use SVD: observe smaller soln norms, even if best
            U,svals,Vh = la.svd(self.Q,full_matrices=False,check_finite=False)
            isvals = 1/svals
            cutoff = max(1e-3*tol,1e-15)     # well below tolerance
            isvals[svals<cutoff*svals[0]] = 0.0    # truncated pinv
            self.QfacSUh = isvals[:,None] * U.T.conj()    # combine 2 factors
            self.QfacV = Vh.T.conj()            # Hermitian transpose
            if verb:
                print('SVD time %.3g s'%(tic()-t0))

    def fillQ(self):
        """fill quasiperiodizing 6m^2-by-Np matrix commonly called Q, in self.
        Q maps aux pt source strengths to discrepancies on UC wall colloc pts.
        self.proxytype controls if charges or dipoles
        """
        self.Q = zeros([0,self.Np])     # empty row to vstack on
        m2 = self.m**2
        s = [m2,self.Np]           # size of each tmp block
        Qp = zeros(s); Qm = zeros(s)     # tmp blocks (n=norm deriv)
        Qnp = zeros(s); Qnm = zeros(s)   # (m=minus,p=plus)
        for d in range(3):                # xyz face directions
            if self.proxytype=='s':
                l3k.lap3dchargemat_numba(self.p,self.c[d,1,:,:],self.cn[d,1,:,:],Qp,Qnp)
                l3k.lap3dchargemat_numba(self.p,self.c[d,0,:,:],self.cn[d,1,:,:],Qm,Qnm)
            elif self.proxytype=='d':
                l3k.lap3ddipolemat_numba(self.p,self.pn,self.c[d,1,:,:],self.cn[d,1,:,:],Qp,Qnp)
                l3k.lap3ddipolemat_numba(self.p,self.pn,self.c[d,0,:,:],self.cn[d,0,:,:],Qm,Qnm)
            else:
                print('proxytype',self.proxytype,'unknown!')
            self.Q = np.vstack([self.Q,Qp-Qm,Qnp-Qnm])

    # uncomment following line only for line_profiler via kernprof...
    #@profile
    def eval(self,y,d,x,verb=0):
        """Evaluate triply-periodic Laplace 3D dipole sum at non-self targets.
        See lap3ddipole_native for free-space definitions.
        Optional inputs:
          verb - (integer) verbosity: 0 silent, 1 text, 2 and figs,...
        """
        Ns=y.shape[0]      # num srcs
        Nt=x.shape[0]      # num targs
        t1=tic() # sum all near images of sources, to the targets:
        y0 = y.dot(self.ilat)   # srcs xformed back as if std UC [-1/2,1/2]^3
        ynr = zeros([0,3])   # all nr srcs, or use list for faster append?
        dnr = zeros([0,3])   # all nr src strength vecs
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    ijk = array([i,j,k])         # integer translation vec
                    y0tr = y0 + ijk                 # broadcast, transl srcs
                    ii = np.max(np.abs(y0tr),axis=1) < (1/2+self.gap) # is near?
                    ynr = np.vstack([ynr, y[ii,:] + ijk.dot(self.lat)])
                    dnr = np.vstack([dnr, d[ii,:]])
        pot = 0*x[:,0]; grad = 0*x     # alloc output arrays
        # NB stacking nr srcs & doing single call here good if # targs big...
        l3k.lap3ddipole_numba(ynr,dnr,x,pot,grad)   # eval direct near sum
        if verb:
            print('eval nt=%d, ns=%d: %d near src, time\t%.3g ms' % (Nt,Ns,ynr.shape[0],1e3*(tic()-t1)))
        
        # compute discrepancy of near source (always dipole) images on UC faces
        t0=tic()
        ynr0 = ynr.dot(self.ilat)  # hack to get all nr src as if std UC (fast)
        discrep = array([])        # length-0 row vec
        m2=self.m**2               # colloc pts per face
        df = np.empty(m2)          # val discrep on a face pair
        gf = np.empty([m2,3])      # grad "
        for f in range(3):                    # xyz face directions
            ii = ynr0[:,f] > (-1/2+self.gap)  # nr srcs "gap-far" from -ve face?
            l3k.lap3ddipole_numba(ynr[ii],dnr[ii],self.c[f,0,:,:],df,gf)
            df = -df; gf = -gf                         # sign for -ve face
            ii = ynr0[:,f] < (1/2-self.gap)   # nr srcs "gap-far" from +ve face?
            l3k.lap3ddipole_numba(ynr[ii],dnr[ii],self.c[f,1,:,:],df,gf,add=True)
            dnf = np.sum(self.cn[f,0,:,:]*gf,axis=1)   # n-deriv = grad dot nor
            discrep = np.hstack([discrep,df,dnf])
        if verb:
            print('\tdiscrep eval\t\t\t\t%.3g ms'%(1e3*(tic()-t0)))

        discrep = -discrep     # since BVP solve aims to cancel discrep
        t0=tic() # solve Q.xi = discrep,  for aux strengths xi...
        #xi = random.rand(self.Np); discrep = self.Q.dot(xi) # TEST ONLY
        if self.facmeth=='':
            xi = la.lstsq(self.Q,discrep)[0]   # O(N^3), too slow, obviously
        elif self.facmeth=='q' or self.facmeth=='p':
            #xi = la.solve_triangular(self.QfacR, self.QfacQ.T @ discrep) # slow
            xi = self.invR @ (self.QfacQ.T @ discrep)   # "solve" : ~2ms
            if self.facmeth=='p':
                xiperm=xi
                xi = 0*xiperm
                xi[self.Qperm] = xiperm   # inverts piv QR perm, takes only 20us
        else:
            xi = self.QfacV @ (self.QfacSUh @ discrep)            
        if verb:
            print('\tsolve lin sys for xi \t\t\t%.3g ms'%(1e3*(tic()-t0)))
            print('\tresid rel nrm %.3g'%(norm(self.Q.dot(xi) - discrep)/norm(discrep))) # adds 1ms
            print('\t|discrep|=%.3g, soln |xi|=%.3g'%(norm(discrep),norm(xi)))
        
        t0=tic() # add aux proxy rep to pot (dipole case: srcdip = xi * direcs)
        if self.proxytype=='s':
            l3k.lap3dcharge_numba(self.p,xi,x,pot,grad,add=True)  # xi = charges
        else:
            l3k.lap3ddipole_numba(self.p,xi[:,None]*self.pn,x,pot,grad,add=True)
        if verb:
            print('\taux rep eval at targs\t\t%.3g ms'%(1e3*(tic()-t0)))
            print('\ttotal eval time: \t\t\t%.3g ms'%(1e3*(tic()-t1)))

        if verb>1:       # plot all near src, all targs...
            pl.gca().scatter(ynr[:,0],ynr[:,1],ynr[:,2],s=1,color='k')
            pl.gca().scatter(x[:,0],x[:,1],x[:,2],color='g',s=3)

        return pot, grad

def test_lap3d3p(tol=1e-3,verb=1):
    """Tester and performance for lap3d3p (Laplace 3D triply-periodic) class.
    Optional inputs:
       tol - requested tolerance
       verb - verbosity: 1 for text, 2 for text+pictures
    """
    l3k.warmup()
    random.seed(seed=0)
    L = array([[1,0,0],[0.3,1,0],[-0.2,0.3,0.9]])  # rows are lattice vecs
    #L = eye(3)        # cubical (best-case) lattice
    p = lap3d3p(L)     # make a periodizing object
    p.precomp(tol)     # doesn't need the src pts
    if verb>1:
        pl.ion(); ax=p.show()     # plot it
    ns = 500                      # sources
    y = (random.rand(ns,3)-1/2).dot(L)    # in [-1/2,1/2]^3 then xform to latt
    d = random.randn(ns,3)        # dipole strength vectors
    x = (random.rand(ns,3)-1/2).dot(L)   # same num of new targets
    # change the first 8 targs to the corners, to check periodicity...
    x[0:8,:] = (np.vstack([zeros(3),eye(3),1-eye(3),ones(3)]) - 1/2).dot(L)
    reps=10       # time the eval (jits have been warmed up)...
    t0=tic()
    for i in range(reps):
        u,g = p.eval(y,d,x)
    teval=(tic()-t0)/reps
    print('3d3p eval time = %.3g ms'%(1e3*teval))
    #print(u[0:8])     # peri-checking pot vals - NB arbitrary constant offset
    #print(g[0:8])     # peri-checking field vals
    print('max corner pot peri abs err: %.3g'%(np.max(u[0:8])-np.min(u[0:8])))
    print('max corner grad peri rel err: %.3g'%(np.max(np.abs(g[0]-g[1:8]))/norm(g[0])))
    if verb>0:
        u,g = p.eval(y,d,x,verb)   # maybe report more info?
    t0=tic()      # time the free-space eval...
    for i in range(reps):
        l3k.lap3ddipole_numba(y,d,x,u,g)
    tnonper=(tic()-t0)/reps
    print('cf non-per eval time %.3g ms (ratio: %.3g)'%(1e3*tnonper,teval/tnonper))

#def convtest_lap3d3p():

    
def slicepts(z0=0.0,a=2.0,m=200):
    """return n*3 array of pts on a square slice z=z0, size a*a in xy plane
    Used for 3d plotting. n=m^2, where m is optional argument.
    """
    gr = np.linspace(-a,a,num=m)    # targets: grid size per dim
    xx,yy = np.meshgrid(gr,gr)
    nt = xx.size
    return np.hstack((xx.ravel()[:,None], yy.ravel()[:,None], z0*np.ones([nt,1]))), xx, yy

def show_perislice(tol=1e-3):
    """eval on a xformed slice through unit cell, plot image, viz check periodic
    """
    L = array([[1,0,0],[0.3,1,0],[-0.2,0.3,0.9]])  # rows are lattice vecs
    p = lap3d3p(L)
    p.precomp(1e-3,verb=0)
    ns = 100                              # sources
    y = (random.rand(ns,3)-1/2).dot(L)    # in [-1/2,1/2]^3 then xform to latt
    d = random.randn(ns,3)                # dipole strength vectors
    x,xx,yy = slicepts(z0=0.3,a=0.5)
    x = x.dot(L)                          # xform targs to lattice
    pot,grad = p.eval(y,d,x)
    pot -= np.mean(pot)                   # since arbitrary offset

    pot3x3 = np.tile(pot.reshape(xx.shape), (3,3))
    c = 20   # colorscale
    fig,ax = pl.subplots()
    im = ax.imshow(pot3x3,cmap='jet',vmin=-c,vmax=c,extent=(-3/2,3/2,-3/2,3/2))
    ax.set_xlabel('x_1 for std UC'); ax.set_ylabel('x_2 for std UC')
    ax.set_title('3x3 copy of x_3-slice, as if std UC; should join smoothly')
    # fig.colorbar(im)  # if no interaction wanted, or:
    myplotutils.myshow(ax,im,fig)   # drag and drag colorbar
    pl.show()   # equiv of drawnow
