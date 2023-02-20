import numpy as np
import pandas
from numpy.linalg import solve
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, Matrix, lambdify, Expr
from scipy.linalg import toeplitz, eig
#import cProfile
from time import time
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import LinearLocator
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, eigs
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class HSS:
    # HSS is a base class. The state space for a particular system
    # is implemented as a child class. All functionality for harmonic
    # state space formulated systems is included in the parent class HSS.

    def __init__(self, N = 4):
        # Time domain related attributes
        #self.w0 = 100 * np.pi
        self.T = 2 * np.pi / self.w0
        self.t = symbols('t')

        # State space matrices and vectors, all symbolic
        self.A = None   # State matrix
        self.B = None   # State input matrix
        self.C = None   # State output matrix
        self.D = None   # State feedthrough matrix
        self.x = []     # State variables
        self.f = []     # Differential equations
        self.u = []     # Input vector
        self.y = []     # Output vector
        self.p = []     # Parameters for parametric studies
        self.x0 = []    # Initial guess state variables
        self.u0 = []    # Input
        self.p_range = [] # Parameter range for sweep
        self.paramvals = None # Replaces p_range
        self.toe_l = None
        self.setup()    # Set the above attributes

        # Sizes etc
        self.N = N      # Harmonic order
        self.Nx = len(self.x)
        self.Nu = len(self.u)
        self.Ny = len(self.y)
        self.Nt = 1000

        # Time domain arrays
        self.ht = self.T / self.Nt  # Sampling rate time domain signal
        self.t_arr = np.linspace(0, self.T - self.ht, self.Nt)
        self.xt = np.zeros((self.Nx, self.Nt))                  # States
        self.ut = np.zeros((self.Nu, self.Nt))                  # Inputs
        self.yt = np.zeros((self.Ny, self.Nt))                  # Outputs
        self.ft = np.zeros((self.Nx, self.Nt))                  # Differential equations
        self.At = np.zeros((self.Nx * self.Nx, self.Nt))        # State matrix
        self.Bt = np.zeros((self.Nx * self.Nu, self.Nt))        # Input matrix
        self.Ct = np.zeros((self.Ny * self.Nx, self.Nt))        # Output matrix
        self.Dt = np.zeros((self.Ny * self.Nu, self.Nt))        # Feedthrough matrix
        self.A_lam = None                                       # lambdified A for time domain
        self.B_lam = None                                       # lambdified B for time domain
        self.C_lam = None                                       # lambdified C for time domain
        self.D_lam = None                                       # lambdified D for time domain
        self.f_lam = None                                       # lambdified f for time domain
        self.y_lam = None                                       # lambdified y for time domain
        self.ut_lam = None                                      # lambdified u for time domain

        # Frequency domain
        # Fourier coefficient arrays
        self.cx = 1j*np.zeros(self.Nx*(2*self.N+1))             # States
        self.cf = 1j*np.zeros(self.Nx*(2*self.N+1))             # Differential equations
        self.cA = 1j*np.zeros((self.Nx * self.Nx, 2*self.N+1))  # State matrix
        self.cB = 1j*np.zeros((self.Nx * self.Nu, 2*self.N+1))  # Input matrix
        self.cC = 1j*np.zeros((self.Nx * self.Ny, 2*self.N+1))  # Output matrix C
        self.cD = 1j*np.zeros((self.Ny * self.Nu, 2*self.N+1))  # Feedthrough matrix
        # Toeplitz arrays
        self.toeA = 1j * np.zeros((self.Nx * (2 * self.N + 1), self.Nx * (2 * self.N + 1)))
        self.toeB = 1j * np.zeros((self.Nx * (2 * self.N + 1), self.Nu * (2 * self.N + 1)))
        self.toeC = 1j * np.zeros((self.Ny * (2 * self.N + 1), self.Nx * (2 * self.N + 1)))
        self.toeD = 1j * np.zeros((self.Ny * (2 * self.N + 1), self.Nu * (2 * self.N + 1)))
        # Miscellaneous
        self.Nvec = np.linspace(-self.N,self.N,2*self.N+1)      # Harmonics vector
        diags = np.asarray([[1j*i*self.w0]*self.Nx for i in range(-self.N, self.N+1)])
        self.Nblk = np.diag(diags.ravel())                      # N block diagonal matrix
        self.angles = None                                      # angles for reduced DFT
        self.exp_arr = None                                     # complex phasors for reduced DFT
        self.htm = 1j * np.zeros((self.N * 2 + 1, self.N * 2 + 1))          # Harmonic transfer matrix

        # Modal analysis
        self.eigen = []
        self.weak_damp = 0
        #self.peval = [1,20]
        self.calc_init()

        # Parametric study
        self.peval = [0] * len(self.p)  # Inserted values in p
        self.XYmesh = (None, None)
        self.damps = np.empty(0)

    def setup(self):
        pass

    def find_pss(self):
        self.err = 100
        u_inp = [self.t_arr] + self.peval
        for ind, xtf in enumerate(self.ut_lam):
            y = xtf(*u_inp)
            if type(y) == np.ndarray:
                self.ut[ind, :] = y
            else:
                self.ut[ind, :] = np.ones(self.Nt) * y

        for i in range(10):
            # Iterate
            # Iterations are cheap, except solve.
            #print(i)
            #print('---------------')
            # Calculate xt from fft coeffs
            cx = np.reshape(self.cx,(self.Nx,-1),'F')  # Reshape to 2-D format
            self.xt = np.real(self.idft_reduced(cx))
            #self.plotstates()

            self.calc_td()  # Calc f and A in time domain

            # Calculate fourier coeffs
            cf = self.dft_reduced(self.ft)
            self.cA = self.dft_reduced(self.At)

            # Reshaping - setup toeplitz A matrix and ravel f
            self.toepA()
            self.cf = cf.ravel('F')
            tol = 1e-10

            lhs = self.Nblk - self.toeA
            rhs = self.cf - np.dot(self.Nblk, self.cx)
            lhs_sp = csc_matrix(lhs)
            dx = spsolve(lhs_sp, rhs)

            dx2 = np.reshape(np.copy(dx), (self.Nx,-1), 'F')
            #print(dx2)
            dx2[:,-1:-self.N-1:-1] = np.conj(dx2[:,0:self.N])
            dx2 = dx2.ravel('F')
            # Todo: Numeric error in dx, so use only one side
            #print(dx2)

            self.cx += dx2
            # dx = solve(lhs, rhs)
            # self.cx += dx
            err = np.sqrt(np.dot(dx2, dx2))
            # print(abs(err))
            # print(dx)

            if abs(err) < tol:
                print('Converged after {} iterations, err: {:.2e}'.format(i + 1, abs(err)))
                break
            elif abs(err) > abs(self.err)*10:

                self.calc_eigs()
                print('Weakest mode damping: {}'.format(self.weak_damp))
                raise RuntimeError('Diverging after {} iterations, err : {:.2e}'.format(i+1, abs(err)))
                break
            else:
                print('Err: {:.2e}'.format(abs(err)))
                self.err = err

    def dft_reduced(self, xin):
        return np.dot(xin, np.transpose(self.exp_arr)) / (self.Nt)

    def idft_reduced(self, cs):
        return np.dot(cs, self.exp_arr ** -1)

    def calc_init(self):
        # Calculates initial time domain values for x and u from sympy expressions

        # Set time domain signals
        self.angles = np.outer(self.Nvec, self.t_arr)
        self.exp_arr = np.exp(-1j * self.w0 * self.angles)

        xt_lam = [lambdify(self.t, i, 'numpy') for i in self.x0]
        for ind, xtf in enumerate(xt_lam):
            y = xtf(self.t_arr)
            if type(y) == np.ndarray:
                self.xt[ind, :] = y
            else:
                self.xt[ind, :] = np.ones(self.Nt) * y
        cx = self.dft_reduced(self.xt)  # Initial x coeffs
        self.cx = cx.ravel('F')

        self.lambdify_funcs()  # For f and A (this is expensive)

    def lambdify_funcs(self):
        var_lst = self.x + [self.t] + self.u + self.p  # Inputs to f
        start = time()
        self.A_lam = [lambdify(var_lst, i, 'numpy') for i in self.A]            # lambdified A for time domain
        self.f_lam = [lambdify(var_lst, i, 'numpy') for i in self.f]            # lambdified f for time domain
        self.ut_lam = [lambdify([self.t]+self.p, i, 'numpy') for i in self.u0]  # lambdified u for time domain
        self.y_lam = [lambdify(var_lst, i, 'numpy') for i in self.y]            # lambdified y for time domain
        self.B_lam = [lambdify(var_lst, i, 'numpy') for i in self.B]            # lambdified B for time domain
        #self.B_lam = [lambdify(var_lst, self.B[0], 'numpy')]
        self.C_lam = [lambdify(var_lst, i, 'numpy') for i in self.C]            # lambdified C for time domain
        self.D_lam = [lambdify(var_lst, i, 'numpy') for i in self.D]            # lambdified D for time domain
        # Setting up symbolic toeplitz matrix (only way to construct toeplitz matrix of matrices)
        Tps = symbols('Tp0:{}'.format(2 * self.N + 1))
        Tns = symbols('Tn0:{}'.format(2 * self.N + 1))
        Toep = Matrix(toeplitz(Tps, Tns))
        self.toe_l = lambdify(list(Tps) + list(Tns), Toep, 'numpy')

        end = time()
        #print(f'1 took {end - start} seconds!')

    def calc_td(self):

        self.f_inp = [self.xt[i, :] for i in range(self.Nx)]
        self.f_inp.append(self.t_arr)  # append time domain
        u_inp = [self.ut[i, :] for i in range(self.Nu)]
        self.f_inp += u_inp

        #self.peval = [np.ones(self.Nt)*i for i in self.peval]
        self.f_inp += self.peval
        #print(self.f_inp)
        #for i in f_inp:
        #plt.plot(self.tarr,i)
        #print(*self.f_inp)
        for ind, xtf in enumerate(self.f_lam):
            y = xtf(*self.f_inp)
            if type(y) == np.ndarray:
                if np.max(np.imag(y)) > 1e-1:
                    print('Something wrong with idft, max imag = {}'.format(np.max(np.imag(y))))
                    plt.plot(self.t_arr,np.imag(self.xt[0,:]))
                    plt.show()

                self.ft[ind,:] = np.real(y)
            else:
                self.ft[ind, :] = np.ones(self.Nt) * y
                #plt.plot(self.tarr,y)

        #plt.show()

        # Calculating At here as well
        for ind, xtA in enumerate(self.A_lam):
            y = xtA(*self.f_inp)
            if type(y) == np.ndarray:
                #if np.max(np.imag(y)) > 1e-15:
                #print('max imag(A) = {}'.format(np.max(np.imag(y))))
                #print(self.A)
                #print('max imag(inp) = {}'.format(np.max(np.imag(self.xt))))
                self.At[ind,:] = np.real(y)
            else:
                self.At[ind,:] = np.ones(self.Nt) * np.real(y)

    def toepA(self):

        # Numeric values (arrays) for toeplitz elements in state matrix A
        toe_pos = [self.cA[:,i].reshape((self.Nx, self.Nx)) for i in range(self.N,2*self.N+1)]
        toe_neg = [self.cA[:,i].reshape((self.Nx, self.Nx)) for i in range(self.N,-1,-1)]

        toe_neg += [np.zeros((self.Nx, self.Nx))]*self.N
        toe_pos += [np.zeros((self.Nx, self.Nx))] * self.N

        T = toe_pos+toe_neg

        # Calculate toeplitz matrix and reshape to proper size (square Nx*(2N+1) matrix)
        # Correct reshape is not very intuitive

        self.toeA = self.toe_l(*T)
        self.toeA = np.reshape(np.transpose(self.toeA, (0, 2, 1, 3)), (self.Nx * (2 * self.N + 1), -1))

    def toepBCD(self):
        # Calculating the toeplitz B matrix
        # Calulate time domain array
        for ind, xtB in enumerate(self.B_lam):
            y = xtB(*self.f_inp)
            if type(y) == np.ndarray:
                self.Bt[ind, :] = np.real(y)
            else:
                self.Bt[ind, :] = np.ones(self.Nt) * np.real(y)
        # Fourier coeffs
        self.cB = self.dft_reduced(self.Bt)
        # Numeric values (arrays) for toeplitz elements in input matrix B
        toe_pos = [self.cB[:, i].reshape((self.Nx, self.Nu)) for i in range(self.N, 2 * self.N + 1)]
        toe_neg = [self.cB[:, i].reshape((self.Nx, self.Nu)) for i in range(self.N, -1, -1)]

        toe_neg += [np.zeros((self.Nx, self.Nu))] * self.N
        toe_pos += [np.zeros((self.Nx, self.Nu))] * self.N

        T = toe_pos + toe_neg

        # Calculate toeplitz matrix and reshape to proper size: Nx*(2N+1) X Nu*(2N+1)

        self.toeB = self.toe_l(*T)
        self.toeB = np.reshape(np.transpose(self.toeB, (0, 2, 1, 3)), (self.Nx * (2 * self.N + 1), -1))
        print('Shape Toeplitz B')
        print(np.shape(self.toeB))

        # Time domain values for C
        for ind, xtC in enumerate(self.C_lam):
            y = xtC(*self.f_inp)
            if type(y) == np.ndarray:
                self.Ct[ind, :] = np.real(y)
            else:
                self.Ct[ind, :] = np.ones(self.Nt) * np.real(y)
        self.cC = self.dft_reduced(self.Ct)

        # Numeric values (arrays) for toeplitz elements in output matrix C
        toe_pos = [self.cC[:, i].reshape((self.Ny, self.Nx)) for i in range(self.N, 2 * self.N + 1)]
        toe_neg = [self.cC[:, i].reshape((self.Ny, self.Nx)) for i in range(self.N, -1, -1)]

        toe_neg += [np.zeros((self.Ny, self.Nx))] * self.N
        toe_pos += [np.zeros((self.Ny, self.Nx))] * self.N

        T = toe_pos + toe_neg

        # Calculate toeplitz matrix and reshape to proper size: Nx*(2N+1) X Nu*(2N+1)

        self.toeC = self.toe_l(*T)
        self.toeC = np.reshape(np.transpose(self.toeC, (0, 2, 1, 3)), (self.Ny * (2 * self.N + 1), -1))
        print('Shape Toeplitz C')
        print(np.shape(self.toeC))

        # Time domain values for D
        for ind, xtD in enumerate(self.D_lam):
            y = xtD(*self.f_inp)
            if type(y) == np.ndarray:
                self.Dt[ind, :] = np.real(y)
            else:
                self.Dt[ind, :] = np.ones(self.Nt) * np.real(y)
        self.cD = self.dft_reduced(self.Dt)

        # Numeric values (arrays) for toeplitz elements in feedthrough matrix D
        toe_pos = [self.cD[:, i].reshape((self.Ny, self.Nu)) for i in range(self.N, 2 * self.N + 1)]
        toe_neg = [self.cD[:, i].reshape((self.Ny, self.Nu)) for i in range(self.N, -1, -1)]

        toe_neg += [np.zeros((self.Ny, self.Nu))] * self.N
        toe_pos += [np.zeros((self.Ny, self.Nu))] * self.N

        T = toe_pos + toe_neg

        # Calculate toeplitz matrix and reshape to proper size: Nx*(2N+1) X Nu*(2N+1)

        self.toeD = self.toe_l(*T)
        self.toeD = np.reshape(np.transpose(self.toeD, (0, 2, 1, 3)), (self.Ny * (2 * self.N + 1), -1))
        print('Shape Toeplitz D')
        print(np.shape(self.toeD))

    def calc_eigs(self):
        cx = np.reshape(self.cx, (self.Nx, -1), 'F')  # Reshape to 2-D format
        self.xt = np.real(self.idft_reduced(cx))
        self.calc_td()
        self.eigs, self.rev = eig(self.toeA - self.Nblk)
        self.lev = np.linalg.inv(self.rev)
        trc_inds = np.abs(np.imag(self.eigs)) <= (self.N-1)*self.w0
        self.eigen = self.eigs[trc_inds]
        self.weak_damp = np.max(np.real(self.eigen))
        #norm = np.dot(self.lev, self.rev).diagonal() ** 0.5
        self.p_f = np.multiply(self.lev.T, self.rev)
        self.p_f = np.abs(self.p_f) / np.abs(self.p_f).max(axis=0)
        pf_n0 = self.p_f[self.N*self.Nx:self.N*self.Nx+self.Nx]
        colidxs = np.where(pf_n0 > 0.9999999999)[1]
        colidxs = np.unique(colidxs)
        self.pf_filt = pf_n0[:,colidxs]
        self.eigred = self.eigs[colidxs]
        rows = []
        self.npf = np.zeros((self.Nx, len(colidxs)), dtype=np.int8)
        for i in range(self.Nx):
            rowidxs = np.argmax(self.p_f[i::self.Nx, colidxs],axis=0)
            self.pf_filt[i, :] = self.p_f[rowidxs*self.Nx+i, colidxs]
            self.npf[i,:] = rowidxs-self.N

    def calc_eigs2(self):
        cx = np.reshape(self.cx, (self.Nx, -1), 'F')  # Reshape to 2-D format
        self.xt = np.real(self.idft_reduced(cx))
        self.calc_td()
        self.eigs, self.rev = eig(self.toeA - self.Nblk)
        self.lev = np.linalg.inv(self.rev)
        trc_inds = np.abs(np.imag(self.eigs)) <= (self.N-1)*self.w0
        self.eigen = self.eigs[trc_inds]
        self.weak_damp = np.max(np.real(self.eigen))
        #norm = np.dot(self.lev, self.rev).diagonal() ** 0.5
        self.p_f = np.multiply(self.lev.T, self.rev)
        self.p_f = np.abs(self.p_f) / np.abs(self.p_f).max(axis=0)
        pf_n0 = self.p_f[self.N*self.Nx:self.N*self.Nx+self.Nx]
        colidxs = np.where(pf_n0 > 0.9999999999)[1]
        colidxs = np.unique(colidxs)
        self.pf_filt = pf_n0[:,colidxs]
        self.eigred = self.eigs[colidxs]
        rows = []
        self.npf = np.zeros((self.Nx, len(colidxs)), dtype=np.int8)
        for i in range(self.Nx):
            rowidxs = np.argmax(self.p_f[i::self.Nx, colidxs],axis=0)
            self.pf_filt[i, :] = self.p_f[rowidxs*self.Nx+i, colidxs]
            self.npf[i,:] = rowidxs-self.N


    def find_center_eig(self):
        """
        Deprecated
        :return:
        """
        from sklearn.cluster import DBSCAN

        def find_correct_eigcenter(nc, eigcenter, heigs):
            if max(abs(heigs.real-eigcenter.real)) > 1e-8:
                idx3 = int((nc-1)/4)*4
            else:
                # Very bad implementation, this essentially tells the function to do nothing
                idx3 = 2*self.N
            neweigs = abs(np.real(eigcenter-heigs)).argsort()[:idx3+1]
            newcenter = heigs[neweigs].imag.argsort()[int(idx3/2)]
            newloc = np.abs(self.eigs-heigs[neweigs][newcenter]).argmin()
            return newloc
        imrest = (self.eigs.imag + 0.1) % (100 * np.pi)
        eigs2 = np.vstack((self.eigs.real, imrest)).T
        db = DBSCAN(eps=1e-3, min_samples=self.N - 3).fit(eigs2)
        labels = db.labels_
        #assert max(labels)+1 == self.Nx  # Check that DBSCAN found as many clusters as there are states
        #imlist = [sorted(self.eigs.imag[labels==i]) for i in range(max(labels)+1)]  # Sorted imaginary values
        #im_centers = [eigfreqs[int(len(eigfreqs)/2)] for eigfreqs in imlist]
        #re_centers = [sum(self.eigs.real[labels==i])/len(self.eigs.real[labels==i]) for i in range(max(labels)+1)]
        rows = [f'{i} s{j:+}jw' for j in range(self.N,-self.N-1,-1) for i in self.x]
        pfs = pd.DataFrame(index=rows)
        for i in range(max(labels)+1):
            clustereigs = self.eigs[labels == i]

            nc = len(clustereigs)
            eigsortidx = np.argsort(clustereigs.imag)
            eigsort = clustereigs[eigsortidx]
            idx1 = int((nc-1)/2)
            eigcenter = eigsort[idx1]
            if (eigcenter.imag+1e-5)%self.w0>2e-5:
                idx = find_correct_eigcenter(nc, eigcenter, self.eigs[labels==i])
                eigcenter = self.eigs[idx]
            else:
                idx = (np.abs(self.eigs - eigcenter)).argmin()
            #idx = (np.abs(self.eigs - eigcenter)).argmin()
            col = '{0:.2f}'.format(eigcenter)
            #pfs = abs(self.p_f[idx,abs(self.p_f[idx,:])>0.1])
            pfs[col] = abs(self.p_f[:,idx])

        pfs = pfs[pfs.max(axis=1) > 1e-1]

        return pfs

    def find_center_eig2(self):
        """
        Deprecated
        :return:
        """
        from sklearn.cluster import DBSCAN

        imrest = (self.eigs.imag+0.1) % (100 * np.pi)
        eigs2 = np.vstack((self.eigs.real, imrest)).T
        db = DBSCAN(eps=1e-3, min_samples=self.N-3).fit(eigs2)
        labels = db.labels_
        #assert max(labels)+1 == self.Nx  # Check that DBSCAN found as many clusters as there are states
        #imlist = [sorted(self.eigs.imag[labels==i]) for i in range(max(labels)+1)]  # Sorted imaginary values
        #im_centers = [eigfreqs[int(len(eigfreqs)/2)] for eigfreqs in imlist]
        #re_centers = [sum(self.eigs.real[labels==i])/len(self.eigs.real[labels==i]) for i in range(max(labels)+1)]
        rows = [f'{i} s{j:+}jw' for j in range(self.N,-self.N-1,-1) for i in self.x]
        pfs = pd.DataFrame(index=rows)
        for i in range(max(labels)+1):
            clustereigs = self.eigs[labels == i]
            nc = len(clustereigs)
            eigsortidx = np.argsort(clustereigs.imag)
            eigsort = clustereigs[eigsortidx]
            idx1 = int((nc-1)/2)
            if abs(max(eigsort.imag))+2 < abs(min(eigsort.imag)) and idx1%2>0:
                idx1 = idx1-2
            eigcenter = eigsort[idx1]

            idx = (np.abs(self.eigs - eigcenter)).argmin()
            col = '{0:.2f}'.format(eigcenter)
            #pfs = abs(self.p_f[idx,abs(self.p_f[idx,:])>0.1])
            pfs[col] = abs(self.p_f[:,idx])

        pfs = pfs[pfs.max(axis=1) > 1e-1]

        return pfs

    def pf_table(self):
        col = ['{0:.3g}'.format(x) for x in self.eigs]
        rows = [f'{i} s{j:+}jw' for j in range(self.N,-self.N-1,-1) for i in self.x]
        p_f = pd.DataFrame(np.abs(self.p_f), columns=col, index=rows)
        rev_abs = pd.DataFrame(np.abs(self.rev), columns=col, index=rows)
        rev_ang = pd.DataFrame(np.angle(self.rev) * 180 / np.pi, columns=col, index=rows)
        self.rev_abs = rev_abs
        self.pf = p_f
        pf0 = self.pf.iloc[91:98]
        return p_f, rev_abs, rev_ang

    def pf_filtered(self, re=(-1e5,1e5), im=None, part_min=0.1):
        """
        Filters the participation factor table.
        Default values yield eigenvalues in the fundamental strip.
        :param re: Eigenvalue real part interval
        :param im: Eigenvalue imaginary part interval
        :param part_min: minimum participation factor value
        :param damping_min: minimum relative damping value
        :return: participation factors (DataFrame)
        """
        if not im:
            im = (0,self.w0/2+1)
        pf = self.pf
        for x in self.eigs:
            cond = np.real(x) < re[0] or np.real(x) > re[1] or \
                   np.abs(np.imag(x)) < im[0] or np.abs(np.imag(x)) > im[1]
            if cond:
                pf = pf.drop(columns =['{0:.3g}'.format(x)], errors = 'ignore')
                self.rev_abs = self.rev_abs.drop(columns =['{0:.3g}'.format(x)], errors = 'ignore')

        cond = np.logical_or(np.logical_or(np.logical_or(np.real(self.eigs) < re[0],np.real(self.eigs) > re[1]),
                                           np.abs(np.imag(self.eigs)) < im[0]), np.abs(np.imag(self.eigs)) > im[1])

        self.eigen = self.eigs[np.logical_not(cond)]
        self.weak_damp = np.max(np.real(self.eigen))
        pf = pf[pf.max(axis = 1)> part_min]
        self.rev_abs = self.rev_abs[self.rev_abs.max(axis=1)>0.1]
        return pf


    def hss_to_htf(self,freqs, tol=1e-10):
        # Supports only first output and input variable
        sI = 2*np.pi*1j*np.eye((2*self.N+1)*self.Nx)
        sI = np.array([sI*k for k in freqs])
        a = self.toeC @ (np.linalg.inv(sI + self.Nblk - self.toeA) @ self.toeB)
        """
        N_plot = self.N
        N_plot = N_plot*2+1
        #if ax == None:
        #    fig, ax = plt.subplots(1)

        fig, axs = plt.subplots(N_plot,N_plot)


        for i in range(N_plot):
            for j in range(N_plot):
                if max(abs(a[:,i,j])<1e-10):
                    fig.delaxes(axs[i,j])
                else:
                    axs[i,j].plot(freqs,abs(a[:,i,j]))


        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        for i in range(N_plot):
            if max(abs(a[:, i, self.N]) > 1e-10):
                #print(np.size((a[:, i, self.N])))
                ax.plot3D(freqs, freqs-(self.N-i)*self.w0/(2*np.pi), abs(a[:, i, self.N]))

        ax.set_xlabel('Input frequency')
        ax.set_ylabel('Output frequency')
        ax.set_zlabel('|H|')


        htfs = []
        idxs = []

        for i in range(self.N*2+1):
            if max(abs(a[:, i, self.N]) > tol):
                htfs.append(np.array(a[:, i, self.N]))
                idxs.append(i)

        return htfs, idxs
        """
        return HTM(freqs, a)


    def plotstates(self):
        ax = plt.subplot()
        legends = []
        # Time domain values for y
        self.calc_td()
        for ind, xtC in enumerate(self.y_lam):
            y = xtC(*self.f_inp)
            if type(y) == np.ndarray:
                self.yt[ind, :] = np.real(y)
            else:
                self.yt[ind, :] = np.ones(self.Nt) * np.real(y)

        for ind, xx in enumerate(self.xt):
            #if ind == 3:
            #    break
            ax.plot(self.t_arr, xx, label='{}'.format(self.x[ind]))
        plt.legend()
        return ax

    def parametric_sweep(self):
        num_p = len(self.paramvals)
        if num_p == 1:
            pass
        elif num_p == 2:
            pass
        pspace = self.paramvals
        damps = 2*np.ones((len(pspace[0]),len(pspace[1])))
        print(damps)
        print(pspace)
        for indi, ival in enumerate(pspace[0]):
            if(len(pspace) > 1):
                for indj, jval in enumerate(pspace[1]):
                    self.peval = [ival,jval]
                    self.find_pss()
                    self.calc_eigs()
                    self.pf_table()
                    pfs = self.pf_filtered(re = (-1e5,1e5), im = (0,self.w0/2+1), part_min = 0.1)
                    damps[indi,indj] = self.weak_damp
        self.XYmesh = np.meshgrid(pspace[0], pspace[1])
        self.damps = damps.T

    def root_locus(self):
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(1)
        pspace = self.paramvals[0]
        n_runs = len(pspace)
        colmap = plt.cm.get_cmap('hsv', self.Nx+1)
        for indi, ival in enumerate(pspace):
            self.peval = [ival]
            self.find_pss()
            self.calc_eigs()
            self.pf_table()
            pfs = self.pf_filtered(re = (-1e5,1e5), im = (0,self.w0/2+1), part_min = 0.1)
            maxidxs = pfs.idxmax()
            for i in range(self.Nx):
                maxidx = maxidxs[i]
                xname = maxidx.split()[0]
                for j in range(self.Nx):
                    if xname == self.x[j].name:
                        break
                eigi = self.eigen[i]
                ax.plot(eigi.real, eigi.imag, 'x', color=colmap(j), markersize=20-indi*16/n_runs)
            ax.axvline(0, color='k', linewidth=0.5)
            ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(f'Param {self.p[0].name} from {pspace[0]:.2} to {pspace[-1]:.2}')
        patches = [mpatches.Patch(color=colmap(i), label=self.x[i].name) for i in range(self.Nx)]
        ax.legend(handles=patches)
        ax.grid(True)

    def eigenloci(self):
        pspace = self.paramvals[0]
        xnames = [x.name for x in self.x]
        xharm = [f'{x.name}_n' for x in self.x]
        cols = xnames+xharm+['eig_re','eig_im',self.p[0].name]
        pftables = pandas.DataFrame(columns=cols)
        for indi, ival in enumerate(pspace):
            self.peval = [ival]
            self.find_pss()
            self.calc_eigs()
            for xi in range(len(self.eigred)):
                pfs_i = self.pf_filt[:,xi].tolist()+self.npf[:,xi].tolist()+[self.eigred[xi].real]+[self.eigred[xi].imag]+[ival]
                pftables.loc[len(pftables)] = pfs_i

        pftables[xharm] = pftables[xharm].astype(int)
        return pftables

    def save_eigenloci(self, pftables, params, params_txt):

        df = pftables
        # define variables
        x_name = 'eig_re'
        y_name = 'eig_im'
        param_name = pftables.keys()[-1]

        x = df[x_name]
        y = df[y_name]/(2*np.pi)
        paramvals = df[param_name].values
        states = list(pftables.keys())[0:self.Nx]

        txt_y = 1.5
        dy = 0.15
        indicators = {state: {'name': state,
                              'values': df[state].values,
                              'harm': df[f'{state}_n'].values,
                              'poly': Polygon([(0, txt_y - 0.1 - dy * idx), (0, txt_y + 0.1 - dy * idx),
                                               (2, txt_y + 0.1 - dy * idx), (2, txt_y - 0.1 - dy * idx)])}
                      for idx, state in enumerate(states)}
        idx = len(states)

        param_dict = {'name': param_name, 'values': paramvals,
                      'poly': Polygon([(0, 1.7), (0, 1.9),
                                       (2, 1.9), (2, 1.7)])}

        # explictit function to hide index
        # of subplots in the figure

        # import required modules
        from matplotlib.gridspec import GridSpec

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})

        # create objects
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 4, figure=fig)

        # create sub plots as grid
        ax1 = fig.add_subplot(gs[:, 0:2])
        ax2 = fig.add_subplot(gs[0,2])
        ax3 = fig.add_subplot(gs[0,3], sharex=ax2, sharey=ax2)
        ax4 = fig.add_subplot(gs[1, 2], sharex=ax2, sharey=ax2)
        ax5 = fig.add_subplot(gs[1, 3], sharex=ax2, sharey=ax2)
        ax2.tick_params(labelbottom=False)
        ax3.tick_params(labelbottom=False, labelleft=False)
        ax5.tick_params(labelleft=False)
        # depict illustration
        plt.show()

        # define figure
        cmap = plt.get_cmap("cividis")

        # Plot
        def plot_sc(x, y, ax):
            sc = ax.scatter(x, y)
            values = param_dict['values']
            sc.set_norm(plt.Normalize(np.nanmin(values), np.nan_to_num(values).max()))
            sc.set_array(values)
            sc.set_cmap(cmap)
            ax.grid(visible=True)
            return sc

        def set_opacity(sc, state):
            partf = indicators[state]['values']
            sc.set_alpha(partf / np.max(partf))

        ax1.set_xlabel('Real [rad/s]', fontsize=12)  # , color='w'
        ax1.set_ylabel('Imag [Hz]', fontsize=12)  # , color='w'

        sc1 = plot_sc(x,y,ax1)
        sc2 = plot_sc(x,y,ax2)
        sc3 = plot_sc(x,y,ax3)
        sc4 = plot_sc(x,y,ax4)
        sc5 = plot_sc(x,y,ax5)

        p0 = params[0]
        p1 = params[1]
        p2 = params[2]
        p3 = params[3]
        set_opacity(sc2, p0)
        set_opacity(sc3, p1)
        set_opacity(sc4, p2)
        set_opacity(sc5, p3)


        p0 = params_txt[0]
        p1 = params_txt[1]
        p2 = params_txt[2]
        p3 = params_txt[3]

        ax2.text(0.1, 0.1, p0, transform=ax2.transAxes, fontsize=16)
        ax3.text(0.1, 0.1, p1, transform=ax3.transAxes, fontsize=16)
        #ax4.text(0.1, 0.1, r'$x_{\beta}$', transform=ax4.transAxes, fontsize=16)
        ax4.text(0.1, 0.1, p2, transform=ax4.transAxes, fontsize=16)
        ax5.text(0.1, 0.1, p3, transform=ax5.transAxes, fontsize=16)

        axs = np.array(fig.axes)
        fig.colorbar(sc1, ax=axs.ravel().tolist())

        #fig.colorbar(sc1, ax=ax1)


        # clean text in axis 2 and reset color of axis 1

    def eigenloci_plot(self, pftables):

        df = pftables
        # define variables
        x_name = 'eig_re'
        y_name = 'eig_im'
        param_name = pftables.keys()[-1]

        x = df[x_name]
        y = df[y_name]/(2*np.pi)
        paramvals = df[param_name].values
        states = list(pftables.keys())[0:self.Nx]

        txt_y = 1.5
        dy = 0.15
        indicators = {state: {'name': state,
                              'values': df[state].values,
                              'harm': df[f'{state}_n'].values,
                              'poly': Polygon([(0, txt_y - 0.1 - dy * idx), (0, txt_y + 0.1 - dy * idx),
                                               (2, txt_y + 0.1 - dy * idx), (2, txt_y - 0.1 - dy * idx)])}
                      for idx, state in enumerate(states)}
        idx = len(states)

        param_dict = {'name': param_name, 'values': paramvals,
                      'poly': Polygon([(0, 1.7), (0, 1.9),
                                       (2, 1.9), (2, 1.7)])}

        # define figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [2, 1]}, facecolor='#393939')
        ax1.tick_params(axis='both')  #, colors='w'
        cmap = plt.get_cmap("cividis")
        ax1.set_title(f'Parametric eigenvalue study of a {self.__class__.__name__}\n '
                      f'{param_name} swept from {paramvals[0]:.2g} to {paramvals[-1]:.2f}', color='w')

        # scatter plot
        sc = ax1.scatter(x, y)
        fig.colorbar(sc, ax=ax1)
        values = param_dict['values']
        sc.set_norm(plt.Normalize(np.nanmin(values), np.nan_to_num(values).max()))
        sc.set_array(values)
        sc.set_cmap(cmap)
        ax1.set_xlabel('Real [rad/s]', color='w')
        ax1.set_ylabel('Imag [Hz]', color='w')

        # axis 2 ticks and limits
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim(0, 2)
        ax2.set_ylim(0, 2)

        # place holder for country name in axis 2
        cnt = ax2.text(1, 1.8, '', ha='center', fontsize=12)

        # indicator texts in axis 2
        txt_x = 0.2
        # txt_y = 1.5
        for ind in indicators.keys():
            n = indicators[ind]['name']
            indicators[ind]['txt'] = ax2.text(txt_x, txt_y, n.ljust(len(n)), ha='left', fontsize=8)
            txt_y -= dy

        # annotation / tooltip
        annot = ax1.annotate("", xy=(0, 0), xytext=(5, 5), textcoords="offset points",
                             bbox=dict(boxstyle="round,pad=0.3", fc="w", lw=2))
        annot.set_visible(False)

        # xy limits
        ax1.set_xlim(x.min() - 5, x.max() + 5)
        ax1.set_ylim(y.min() - 5, y.max() + 5)

        # notes axis 2
        note = 'Participation factors'
        ax2.text(0.5, 1.7, note, ha='left', va='top', fontsize=8)

        def change_opacity(values, annotation):
            clean_ax2()
            annotation.set_color('#2A74A2')
            sc.set_alpha(values/np.max(values))

        # clean text in axis 2 and reset color of axis 1
        def clean_ax2():
            for ind in indicators.keys():
                indicators[ind]['txt'].set_color('black')
            cnt.set_color('black')
            #sc.set_color('black')

        # cursor hover
        def hover(event):
            # check if event was in axis 1
            if event.inaxes == ax1:
                #clean_ax2()
                # get the points contained in the event
                cont, ind = sc.contains(event)
                if cont:
                    # change annotation position
                    annot.xy = (event.xdata, event.ydata)
                    # write the name of every point contained in the event
                    points = "{}".format(', '.join([f'{paramvals[n]:.4g}' for n in ind["ind"]]))
                    annot.set_text(points)
                    annot.set_visible(True)
                    # get swept parameter
                    param = ind["ind"][0]
                    # set axis 2 param text
                    cnt.set_text(f'{param_name}: {paramvals[param]:.4g}')
                    # set axis 2 indicators values
                    for ind in indicators.keys():
                        n = indicators[ind]['name']
                        txt = indicators[ind]['txt']
                        val = indicators[ind]['values'][param]
                        harm = indicators[ind]['harm'][param]
                        txt.set_text(f'{n} ({harm}): {val: .2f}')

                # when stop hovering a point hide annotation
                else:
                    annot.set_visible(False)
            # check if event was in axis 2
            elif event.inaxes == ax2:
                # bool to detect when mouse is not over text space
                reset_flag = False
                for ind in indicators.keys():
                    # check if cursor position is in text space
                    if indicators[ind]['poly'].contains(Point(event.xdata, event.ydata)):
                        # clean axis 2 and change color map
                        clean_ax2()
                        change_opacity(indicators[ind]['values'], indicators[ind]['txt'])
                        reset_flag = False
                        break
                    elif param_dict['poly'].contains(Point(event.xdata, event.ydata)):
                        clean_ax2()
                        sc.set_alpha(np.ones_like(indicators[ind]['values']))
                        reset_flag = False
                        break

                    else:
                        reset_flag = True
                # If cursor not over any text clean axis 2
                #if reset_flag:
                #clean_ax2()
            fig.canvas.draw_idle()

        # when leaving any axis clean axis 2 and hide annotation
        def leave_axes(event):
            #clean_ax2()
            annot.set_visible(False)

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect('axes_leave_event', leave_axes)
        plt.show()

    def plot_parametric_study2d(self, fig = None, **kwargs):
        if not fig:
            fig, ax = plt.subplots()
        else:
            pass
            #TODO get axis
        X = self.XYmesh[0]
        Y = self.XYmesh[1]
        if 'levels' in kwargs:
            levels = kwargs['levels']
            contourf_ = ax.contourf(X,Y,self.damps, levels=levels,extend='max')
            cbar = fig.colorbar(contourf_, ticks=np.linspace(levels[0], levels[-1], 4))

        else:
            contourf_ = ax.contourf(X, Y, self.damps)
            cbar = fig.colorbar(contourf_)

        ax.set_xlabel(self.p[0])
        ax.set_ylabel(self.p[1])
        #xidxs = np.arange(len(self.paramvals[0]))
        yidxs = self.damps.argmin(axis=0)

        colvals = [self.paramvals[1][i] for i in yidxs]
        #dampvals = [self.damps[i, j] for i, j in zip(xidxs, yidxs)]
        # Print trace of highest damping
        #ax.plot(self.paramvals[0], np.asarray(colvals),linewidth=3)
        plt.show()
        return ax

    def plot_parametric_study3d(self, ax=None):
        if not ax:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = self.XYmesh[0]
        Y = self.XYmesh[1]
        surf = ax.plot_surface(X, Y, self.damps, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel(self.p[0])
        ax.set_ylabel(self.p[1])
        plt.show()
        return ax


class HTM:
    def __init__(self, freqs, htm):
        """

        :param htm: numpy 3-dim array, shape: (# frequency points, # htm rows, #htm cols)
        """
        self.freqs = freqs
        self.htm = htm
        shp = np.shape(htm)
        self.nf = len(freqs)
        assert shp[1]==shp[2]
        self.N = int((shp[1]-1)/2)

    def plot(self, Nplot=None, axs = np.empty(0), xscale = 'linear', yscale = 'linear'):
        if not Nplot or Nplot>self.N:
            Nplot = self.N
        if axs.size==0:
            fig, axs = plt.subplots(Nplot*2+1, Nplot*2+1, sharex=True)
        idxs = range(self.N-Nplot, self.N+Nplot+1)
        for n, row in enumerate(idxs):
            for m, col in enumerate(idxs):
                amp = np.abs(self.htm[:,row,col])
                if np.max(amp)>1e-10:
                    axs[n,m].plot(self.freqs, np.abs(self.htm[:,row,col]))
                    axs[n,m].set_yscale(yscale)
                    axs[n,m].set_xscale(xscale)

        return axs

    def get_mid_col(self, tol= 1e-2, N = 6):
        htfs = []
        idxs = []
        midcol = self.htm[:,:,self.N]
        maxhtf = np.max(abs(midcol), axis=1)
        for i in range(self.N * 2 + 1):
            if max(abs(self.htm[:, i, self.N])/maxhtf > tol) and abs(i-self.N)<N+1:
                htfs.append(np.array(self.htm[:, i, self.N]))
                idxs.append(i-self.N)
        return MidcolHTM(self.freqs, htfs, idxs, self.N)

    def siso_eq(self, other):
        """

        :param other: Grid impedance (HTM)
        :return:
        """
        a = self.htm[:,:,self.N]
        a = np.reshape(np.delete(a, (self.N), axis=1),(len(self.freqs),1,2*self.N))
        b = self.htm[:,self.N,:]
        b = np.reshape(np.delete(b, (self.N), axis=1),(len(self.freqs),2*self.N,1))
        alph = other.htm[:,:,self.N]
        bet = other.htm[:,self.N,:]
        H0 = self.htm[:,self.N,self.N]
        Q = np.delete(self.htm, (self.N), axis=1)
        Q = np.delete(Q, (self.N), axis=2)
        otherQ = np.delete(other.htm, (self.N), axis=1)
        otherQ = np.delete(otherQ, (self.N), axis=2)
        fbk = np.linalg.inv(np.eye(self.N*2)+np.matmul(otherQ,Q))
        Hsiso = np.reshape(H0,(len(self.freqs),1,1))-np.matmul(np.matmul(a,fbk),np.matmul(otherQ,b))
        Hsiso = np.reshape(Hsiso, (len(self.freqs)))

        return SISOeq(self.freqs, Hsiso)

    def siso_rm0(self):
        return SISOeq(self.freqs, self.htm[:,self.N, self.N])

    def __invert__(self):
        return HTM(self.freqs, np.linalg.inv(self.htm))

    def __mul__(self, other):
        return HTM(self.freqs, np.einsum('nij,njk->nik',self.htm,other.htm))


class MidcolHTM:
    def __init__(self, freqs, htfs, idxs, N):
        self.freqs = freqs
        self.htfs = htfs
        self.idxs = idxs
        self.N = N
        self.nf = len(freqs)

    def plot(self, axs = None, xscale = 'linear', yscale = 'linear', rot=0, **kwargs):
        if not axs:
            fig, axs = plt.subplots(2, sharex=True)

        #plt.rcParams.update({
        #    "text.usetex": True,
        #    "font.family": "sans-serif",
        #    "font.sans-serif": ["Helvetica"]})
        ax = axs[0]
        ax2 = axs[1]
        for htf, idx in zip(self.htfs, self.idxs):
            ax.plot(self.freqs, abs(htf), label=f'$H_{{{idx}}}$', **kwargs)

            ax2.plot(self.freqs, np.angle(htf*np.exp(1j*rot*idx)), **kwargs)
            ax.set_yscale(yscale)
            ax.set_xscale(xscale)
            ax.grid(visible=True)
        ax.legend()
        return axs

class SISOeq:
    def __init__(self, freqs, hsiso):
        self.freqs = freqs
        self.hsiso = hsiso

    def bode(self, axs=[]):
        if not list(axs):
            fig, axs = plt.subplots(2,1, sharex=True)

        axs[0].plot(self.freqs, np.abs(self.hsiso))
        axs[1].plot(self.freqs, np.angle(self.hsiso))

        return axs

    def plot_nyquist(self, other, fig=None):
        if not fig:
            fig, ax = plt.subplots(1)
        else:
            ax = fig.get_axes()
        L = self.hsiso*other.hsiso
        ax.plot(np.real(L), np.imag(L))
        re, im = (np.real(L), np.imag(L))
        #for i in range(0,len(L),10):
        #    ax.arrow(re[i], im[i], re[i+1]-re[i], im[i+1]-im[i], shape='full', lw=0, length_includes_head=True, head_width=.05)
        #for i in range(0,len(L),int(len(L)/20)):
            #ax.annotate(f'{self.freqs[i]/(2*np.pi):.2f}', xy=(re[i],im[i]), textcoords='data')  # <--
        ax.grid()
        return fig, ax


class SogiPll(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 1
        super().__init__()

    def setup(self):
        #print('Setup symbolic HSS')

        # States
        xa, xb, xpll, xd = symbols('xa xb xpll xd')
        self.x = [xa,xb,xpll,xd]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        # Params for parametric studies
        ksog, a_pll = symbols('ksog a_pll')
        self.p = [ksog,a_pll]
        #self.p = []
        self.p_range = [(0.1,10,15), (1,150,20)]

        # Outputs
        self.y = [xd]

        # Fixed params
        #ksog = 1

         #ksog = np.sqrt(2)*1

        # Initial conditions u
        self.u0 = [self.U0*cos(self.w0*self.t)+0.5*self.U0*cos(self.w0*3*self.t)]

        # Initial conditions x (guess)
        self.x0 = Matrix([self.U0*cos(self.w0*self.t),
                   self.U0/self.w0*sin(self.w0*self.t),
                   0, 0])

        # Internal expressions
        kp = 2 * a_pll / self.U0
        ki = 2 * a_pll ** 2 / self.U0

        th_pll = xd + self.w0*self.t
        wpll = 1/(1 - kp*xb*cos(th_pll)) * (xpll + self.w0 + kp * (-sin(th_pll) * xa))
        uq = (-sin(th_pll) * xa + cos(th_pll) * xb * wpll)

        # Differential equations
        fa = ksog*(ug-xa)*wpll-xb*wpll**2
        fb = xa
        fpll = ki*uq
        fd = xpll+kp*uq
        self.f = [fa, fb, fpll, fd]

        #print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)


class SimpleSys(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 1
        super().__init__()
        self.A = Matrix(self.f).jacobian(self.x)
        print(self.A)

    def setup(self):
        #print('Setup symbolic HSS')

        # States
        xa, xb = symbols('xa xb')
        self.x = [xa, xb]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        self.y = [xa]                                # Outputs
        ksog = 2
        kp = 1
        ki = 1

        self.p = [ksog]
        self.x0 = Matrix([self.U0*cos(self.w0*self.t),
                   self.U0*sin(self.w0*self.t)])
        self.u0 = [self.U0*cos(self.w0*self.t)+0.5*self.U0*cos(self.w0*3*self.t)]
        fa = (ksog*(ug-xa)-xb)*self.w0
        fb = xa*self.w0

        self.f = [fa, fb]             # Differential eqs

        #print('x: {}'.format(self.x))

def main():
    s = SogiPll()
    #s.find_pss()
    s.parametric_sweep()
    #cx = np.reshape(s.cx,(s.Nx,-1),'F')  # Reshape to 2-D format
    #y = s.idft_reduced(cx)
    #print(cx)
    #cProfile.run('s.parametric_sweep()',sort = 'cumulative')
    #s.calc_eigs()
    #print(s.eigen)
    from numpy import sum
    #print(sum(np.imag(s.eigen)))
    #s2 = SimpleSys()
    #s2.find_pss()

if __name__ == '__main__':
    main()