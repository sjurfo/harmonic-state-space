import numpy as np
from numpy.linalg import solve
from sympy import symbols, sin, cos, Matrix, lambdify
from scipy.linalg import toeplitz, eig
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class HSS:
    """Base class containing methods and descriptive attributes for a Harmonic State Space model

    A particular system can be modelled with HSS as a child class. The child class overrides the setup method.
    """
    def __init__(self, N=4, w0=100*np.pi):
        """
        The init class calls self.setup in which each model is defined
        :param N: Harmonic order
        :param w0: Fundamental frequency
        """
        self.w0 = w0
        self.T = 2 * np.pi / self.w0
        self.t = symbols('t')

        # State space formulation
        self.x = []     # State variables
        self.f = []     # Differential equations
        self.u = []     # Input vector
        self.y = []     # Output vector
        self.p = []     # Parameters (for parametric studies)
        self.p_space = None  # Parameter space
        self.p_value = []  # Inserted values in p
        self.x0 = []    # Initial x values
        self.u0 = []    # Input expression
        self.setup()    # Set the above attributes

        # Linearised state matrices
        self.A = Matrix(self.f).jacobian(self.x)  # State matrix
        self.B = Matrix(self.f).jacobian(self.u)  # State input matrix
        self.C = Matrix(self.y).jacobian(self.x)  # State output matrix
        self.D = Matrix(self.y).jacobian(self.u)  # State feedthrough matrix

        # Sizes
        self.N = N
        self.Nx = len(self.x)
        self.Nu = len(self.u)
        self.Ny = len(self.y)

        # Lambdify symbolic expressions for swift time-domain evaluation
        var_lst = self.x + [self.t] + self.u + self.p   # Inputs to f
        self.A_lam = [lambdify(var_lst, i, 'numpy') for i in self.A]
        self.B_lam = [lambdify(var_lst, i, 'numpy') for i in self.B]
        self.C_lam = [lambdify(var_lst, i, 'numpy') for i in self.C]
        self.D_lam = [lambdify(var_lst, i, 'numpy') for i in self.D]
        self.y_lam = [lambdify(var_lst, i, 'numpy') for i in self.y]
        self.f_lam = [lambdify(var_lst, i, 'numpy') for i in self.f]
        self.x_lam = [lambdify(self.t, i, 'numpy') for i in self.x0]
        self.u_lam = [lambdify([self.t]+self.p, i, 'numpy') for i in self.u0]
        #self.toe_l = None

        # Setting up symbolic toeplitz matrix (only way to construct toeplitz matrix of matrices)
        Tps = symbols('Tp0:{}'.format(2 * self.N + 1))
        Tns = symbols('Tn0:{}'.format(2 * self.N + 1))
        Toep = Matrix(toeplitz(Tps, Tns))
        self.toe_l = lambdify(list(Tps) + list(Tns), Toep, 'numpy')

        # PSS
        self.pss = PSS(self)
        self.calc_init()

    def setup(self):
        """HSS models override this method"""
        pass

    def calc_init(self):
        """
        Calculates initial time domain values for x and u from sympy expressions
        """

        angles = np.outer(self.pss.Nvec, self.pss.t_arr)
        self.pss.exp_arr = np.exp(-1j * self.w0 * angles)

        for ind, xtf in enumerate(self.x_lam):
            y = xtf(self.pss.t_arr)
            if type(y) == np.ndarray:
                self.pss.xt[ind, :] = y
            else:
                self.pss.xt[ind, :] = np.ones(self.pss.Nt) * y
        cx = self.dft(self.pss.xt)
        self.pss.cx = cx.ravel('F')

    def find_pss(self):
        """
        Finds the PSS through the AMG method.

        This method is based on a translated version written for the publication [1]
        [1] An automated method
        """
        err = 100
        u_inp = [self.pss.t_arr] + self.p_value
        for ind, ui in enumerate(self.u_lam):
            y = ui(*u_inp)
            if type(y) == np.ndarray:
                self.pss.ut[ind, :] = y
            else:
                self.pss.ut[ind, :] = np.ones(self.pss.Nt) * y

        for i in range(10):

            # Calculate xt from fft coeffs
            cx = np.reshape(self.pss.cx,(self.Nx,-1),'F')  # Reshape to 2-D format
            self.pss.xt = np.real(self.inv_dft(cx))

            self.calc_td()  # Calc f and A in time domain

            # Calculate fourier coefficients
            cf = self.dft(self.pss.ft)
            self.pss.cA = self.dft(self.pss.At)

            # Reshaping - setup toeplitz A matrix and ravel f
            self.toep_A()
            self.pss.cf = cf.ravel('F')
            tol = 1e-10

            # Iteration equation
            lhs = self.pss.Nblk - self.pss.toeA
            rhs = self.pss.cf - np.dot(self.pss.Nblk, self.pss.cx)
            lhs_sp = csc_matrix(lhs)
            # Iteration step
            dx = spsolve(lhs_sp, rhs)
            dx2 = np.reshape(np.copy(dx), (self.Nx,-1), 'F')
            dx2[:,-1:-self.N-1:-1] = np.conj(dx2[:,0:self.N])
            dx2 = dx2.ravel('F')
            self.pss.cx += dx2

            # Check for convergence
            err_i = np.sqrt(np.dot(dx2, dx2))
            if abs(err_i) < tol:
                print('Converged after {} iterations, err: {:.2e}'.format(i + 1, abs(err_i)))
                break
            elif abs(err_i) > abs(err)*10:
                # In case of divergence
                self.calc_modal_props()
                print('Weakest mode damping: {}'.format(self.pss.modal_props.weak_damp))
                raise RuntimeError('Diverging after {} iterations, err : {:.2e}'.format(i+1, abs(err_i)))
            else:
                print('Err: {:.2e}'.format(abs(err_i)))
                err = err_i

    def dft(self, xin):
        """Computes the DFT for the given harmonic order"""
        return np.dot(xin, np.transpose(self.pss.exp_arr)) / self.pss.Nt

    def inv_dft(self, cs):
        """Computes the Inverse DFT for the given harmonic order"""
        return np.dot(cs, self.pss.exp_arr ** -1)

    def calc_td(self):
        """
        Computes the time domain arrays for f and A
        """
        self.pss.vars_t = [self.pss.xt[i, :] for i in range(self.Nx)]
        self.pss.vars_t.append(self.pss.t_arr)  # append time domain
        u_inp = [self.pss.ut[i, :] for i in range(self.Nu)]
        self.pss.vars_t += u_inp
        self.pss.vars_t += self.p_value

        for ind, f_i in enumerate(self.f_lam):
            y = f_i(*self.pss.vars_t)
            if type(y) == np.ndarray:
                if np.max(np.imag(y)) > 1e-1:
                    print('Something wrong with idft, max imag = {}'.format(np.max(np.imag(y))))
                    plt.plot(self.pss.t_arr,np.imag(self.pss.xt[0,:]))
                    plt.show()

                self.pss.ft[ind,:] = np.real(y)
            else:
                self.pss.ft[ind, :] = np.ones(self.pss.Nt) * y

        # Calculate At here as well
        for ind, A_i in enumerate(self.A_lam):
            y = A_i(*self.pss.vars_t)
            if type(y) == np.ndarray:
                self.pss.At[ind,:] = np.real(y)
            else:
                self.pss.At[ind,:] = np.ones(self.pss.Nt) * np.real(y)

    def toep_A(self):
        """Computes the Toeplitz A matrix"""
        # Numeric values (arrays) for toeplitz elements in state matrix A
        toe_pos = [self.pss.cA[:,i].reshape((self.Nx, self.Nx)) for i in range(self.N,2*self.N+1)]
        toe_neg = [self.pss.cA[:,i].reshape((self.Nx, self.Nx)) for i in range(self.N,-1,-1)]

        toe_neg += [np.zeros((self.Nx, self.Nx))]*self.N
        toe_pos += [np.zeros((self.Nx, self.Nx))] * self.N

        T = toe_pos+toe_neg

        # Calculate toeplitz matrix and reshape to proper size (square Nx*(2N+1) matrix)
        # Correct reshape is not very intuitive

        self.pss.toeA = self.toe_l(*T)
        self.pss.toeA = np.reshape(np.transpose(self.pss.toeA, (0, 2, 1, 3)), (self.Nx * (2 * self.N + 1), -1))

    def toep_BCD(self):
        """Computes the Toeplitz B, C and D matrices"""

        # Calculate time domain array
        for ind, xtB in enumerate(self.B_lam):
            y = xtB(*self.pss.vars_t)
            if type(y) == np.ndarray:
                self.pss.Bt[ind, :] = np.real(y)
            else:
                self.pss.Bt[ind, :] = np.ones(self.pss.Nt) * np.real(y)

        # Fourier coeffs
        self.pss.cB = self.dft(self.pss.Bt)

        # Numeric values (arrays) for toeplitz elements in input matrix B
        toe_pos = [self.pss.cB[:, i].reshape((self.Nx, self.Nu)) for i in range(self.N, 2 * self.N + 1)]
        toe_neg = [self.pss.cB[:, i].reshape((self.Nx, self.Nu)) for i in range(self.N, -1, -1)]
        toe_neg += [np.zeros((self.Nx, self.Nu))] * self.N
        toe_pos += [np.zeros((self.Nx, self.Nu))] * self.N
        T = toe_pos + toe_neg

        # Calculate toeplitz matrix and reshape to proper size: Nx*(2N+1) X Nu*(2N+1)
        self.pss.toeB = self.toe_l(*T)
        self.pss.toeB = np.reshape(np.transpose(self.pss.toeB, (0, 2, 1, 3)), (self.Nx * (2 * self.N + 1), -1))

        # Time domain values for C
        for ind, xtC in enumerate(self.C_lam):
            y = xtC(*self.pss.vars_t)
            if type(y) == np.ndarray:
                self.pss.Ct[ind, :] = np.real(y)
            else:
                self.pss.Ct[ind, :] = np.ones(self.pss.Nt) * np.real(y)
        self.pss.cC = self.dft(self.pss.Ct)

        # Numeric values (arrays) for toeplitz elements in output matrix C
        toe_pos = [self.pss.cC[:, i].reshape((self.Ny, self.Nx)) for i in range(self.N, 2 * self.N + 1)]
        toe_neg = [self.pss.cC[:, i].reshape((self.Ny, self.Nx)) for i in range(self.N, -1, -1)]
        toe_neg += [np.zeros((self.Ny, self.Nx))] * self.N
        toe_pos += [np.zeros((self.Ny, self.Nx))] * self.N
        T = toe_pos + toe_neg

        # Calculate toeplitz matrix and reshape to proper size: Nx*(2N+1) X Nu*(2N+1)
        self.pss.toeC = self.toe_l(*T)
        self.pss.toeC = np.reshape(np.transpose(self.pss.toeC, (0, 2, 1, 3)), (self.Ny * (2 * self.N + 1), -1))

        # Time domain values for D
        for ind, xtD in enumerate(self.D_lam):
            y = xtD(*self.pss.vars_t)
            if type(y) == np.ndarray:
                self.pss.Dt[ind, :] = np.real(y)
            else:
                self.pss.Dt[ind, :] = np.ones(self.pss.Nt) * np.real(y)
        self.pss.cD = self.dft(self.pss.Dt)

        # Numeric values (arrays) for toeplitz elements in feedthrough matrix D
        toe_pos = [self.pss.cD[:, i].reshape((self.Ny, self.Nu)) for i in range(self.N, 2 * self.N + 1)]
        toe_neg = [self.pss.cD[:, i].reshape((self.Ny, self.Nu)) for i in range(self.N, -1, -1)]
        toe_neg += [np.zeros((self.Ny, self.Nu))] * self.N
        toe_pos += [np.zeros((self.Ny, self.Nu))] * self.N
        T = toe_pos + toe_neg

        # Calculate toeplitz matrix and reshape to proper size: Nx*(2N+1) X Nu*(2N+1)
        self.pss.toeD = self.toe_l(*T)
        self.pss.toeD = np.reshape(np.transpose(self.pss.toeD, (0, 2, 1, 3)), (self.Ny * (2 * self.N + 1), -1))

    def calc_modal_props(self):
        """Computes the modal properties for the current PSS"""

        # Recompute the Toeplitz A matrix for the current PSS
        cx = np.reshape(self.pss.cx, (self.Nx, -1), 'F')  # Reshape to 2-D format
        self.pss.xt = np.real(self.inv_dft(cx))
        self.calc_td()

        # Compute eigenvalues and vectors of the HSS
        eigs, rev = eig(self.pss.toeA - self.pss.Nblk)
        lev = np.linalg.inv(rev)

        # Modes in the fundamental strip
        trc_inds = np.abs(np.imag(eigs)) <= (self.N-1)*self.w0
        self.pss.modal_props.eig_fstrip = eigs[trc_inds]
        self.pss.modal_props.weak_damp = np.max(np.real(self.pss.modal_props.eig_fstrip))

        # Modes with high participation from zero-shift states
        p_f = np.multiply(lev.T, rev)
        self.pss.modal_props.p_f = np.abs(p_f) / np.abs(p_f).max(axis=0)
        pf_n0 = self.pss.modal_props.p_f[self.N*self.Nx:self.N*self.Nx+self.Nx]
        col_idxs = np.where(pf_n0 > 0.9999999999)[1]
        col_idxs = np.unique(col_idxs)
        self.pss.modal_props.pf_filt = pf_n0[:,col_idxs]
        self.pss.modal_props.eig_filt = eigs[col_idxs]
        self.pss.modal_props.npf = np.zeros((self.Nx, len(col_idxs)), dtype=np.int8)

        for i in range(self.Nx):
            rowidxs = np.argmax(self.pss.modal_props.p_f[i::self.Nx, col_idxs],axis=0)
            self.pss.modal_props.pf_filt[i, :] = self.pss.modal_props.p_f[rowidxs*self.Nx+i, col_idxs]
            self.pss.modal_props.npf[i,:] = rowidxs-self.N

    def plot_states(self):
        """Plots the states in time domain"""
        ax = plt.subplot()

        self.calc_td()
        for ind, xx in enumerate(self.pss.xt):
            ax.plot(self.pss.t_arr, xx, label='{}'.format(self.x[ind]))
        plt.legend()
        return ax


class PSS:
    """Dataclass for attributes associated with a Periodic Steady State"""
    def __init__(self, hss_model, nt=1000):
        """

        :param hss_model: The HSS model to which this PSS pertains
        :param n_t: Number of samples in time domain
        """
        self.model = hss_model
        self.modal_props = ModalProps()
        self.Nt = nt

        # Shorthand for readability
        nx = self.model.Nx
        nu = self.model.Nu
        ny = self.model.Ny
        n = self.model.N
        m = 2*n+1

        # Time domain arrays
        self.ht = self.model.T / nt  # Sampling rate time domain signal
        self.t_arr = np.linspace(0, self.model.T - self.ht, nt)
        self.xt = np.empty((nx, nt))
        self.ut = np.empty((nu, nt))
        self.yt = np.empty((ny, nt))
        self.ft = np.empty((nx, nt))
        self.At = np.empty((nx*nx, nt))
        self.Bt = np.empty((nx*nu, nt))
        self.Ct = np.empty((ny*nx, nt))
        self.Dt = np.empty((ny*nu, nt))
        self.vars_t = []  # Vars in time domain for lambdified functions

        # Frequency domain
        # Fourier coefficient arrays
        self.cx = 1j * np.empty(nx * m)
        self.cf = 1j*np.empty(nx*m)
        self.cA = 1j*np.empty((nx*nx, m))
        self.cB = 1j*np.empty((nx*nu, m))
        self.cC = 1j*np.empty((nx*ny, m))
        self.cD = 1j*np.empty((ny*nu, m))
        # Toeplitz arrays
        self.toeA = 1j * np.empty((nx*m, nx*m))
        self.toeB = 1j * np.empty((nx*m, nu*m))
        self.toeC = 1j * np.empty((ny*m, nx*m))
        self.toeD = 1j * np.empty((ny*m, nu*m))

        # NOTE: The state matrix arrays above are flattened to 2 dimensions for compatibility with lambdify. This causes
        # readability issues and may be addressed in future updates.

        # Miscellaneous
        self.Nvec = np.linspace(-n,n,m)      # Harmonics vector
        diags = np.asarray([[1j*i*self.model.w0]*nx for i in range(-n, n+1)])
        self.Nblk = np.diag(diags.ravel())                      # N block diagonal matrix
        self.exp_arr = None                                     # complex phasors for reduced DFT


class ModalProps:
    """Dataclass for the modal properties of a Periodic Steady State"""
    def __init__(self):
        # Modal analysis
        self.eig_fstrip = []
        self.eig_filt = []
        self.pf_filt = np.empty(0)
        self.p_f = []
        self.weak_damp = 0

        # 2-param damping plot
        self.XYmesh = (None, None)
        self.damps = np.empty(0)