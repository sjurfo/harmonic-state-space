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
        self.h = []     # Change of variable
        self.u = []     # Input vector
        self.y = []     # Output vector
        self.p = []     # Parameters (for parametric studies)
        self.p_space = None  # Parameter space
        self.p_value = []  # Inserted values in p
        self.x0 = []    # Initial x values
        self.u0 = []    # Input expression
        self.setup()    # Set the above attributes

        # Symbolic Jacobian matrices
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

        # Lambdify for ivp scipy solver
        f = [i.subs(zip(self.u+self.p, self.u0+self.p_value)) for i in self.f]
        self.f_lam_ivp = lambdify((self.t, self.x), f)

        # Setting up symbolic toeplitz matrix (crude way to construct toeplitz matrix of matrices)
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

    def change_of_variable(self):
        # Change of variables
        var_lst = self.x + [self.t] + self.u + self.p   # Inputs to f
        if self.h:
            self.H = Matrix(self.h).jacobian(self.x)  # State matrix
            self.H_inv = self.H.inv()  # State matrix
            self.H_lam = [lambdify(var_lst, i, 'numpy') for i in self.H]
            self.H_inv_lam = [lambdify(var_lst, i, 'numpy') for i in self.H_inv]

    def calc_init(self):
        """
        Calculates initial time domain values for x and u from sympy expressions
        """

        angles = np.outer(self.pss.Nvec, self.pss.t_arr)
        self.pss.exp_arr = np.exp(-1j * self.w0 * angles)

        self.pss.xt = self.calc_lam_td(self.x_lam, [self.pss.t_arr])
        cx = self.dft(self.pss.xt)
        self.pss.cx = cx.ravel('F')

    def find_pss(self):
        """
        Finds the PSS through the AMG method.

        This method is based on a translated version written for the publication [1]
        [1] An Integrated Method for Generating VSCs’ Periodical Steady-State Conditions and HSS-Based Impedance Model
        """
        err = 1e10
        u_inp = [self.pss.t_arr] + self.p_value
        self.pss.ut = self.calc_lam_td(self.u_lam, u_inp)
        for i in range(20):
            self.calc_var_td()
            self.pss.ft = self.calc_lam_td(self.f_lam, self.pss.vars_t)  # Calc f in time domain
            self.pss.At = self.calc_lam_td(self.A_lam, self.pss.vars_t)  # Calc A in time domain

            # Calculate fourier coefficients
            cf = self.dft(self.pss.ft)
            self.pss.cA = self.dft(self.pss.At)

            # Reshaping - setup toeplitz A matrix and ravel f
            self.pss.toeA = self.toeplitz_setup(self.pss.cA, (self.Nx, self.Nx))
            self.pss.cf = cf.ravel('F')
            tol = 1e-10

            # RHS of frequency domain iteration equation
            rhs = self.pss.cf - np.dot(self.pss.Nblk, self.pss.cx)

            # Check for convergence
            err_i = np.sqrt(np.dot(rhs, rhs))

            if abs(err_i) < tol:

                #print(f'\rConverged in iteration {i+1}, err: {abs(err_i):.2e}', end='')
                return (i, abs(err_i))
                #break

            elif abs(err_i) > abs(err)*10:
                # In case of divergence
                self.calc_modal_props()
                print('Weakest mode damping: {}'.format(self.pss.modal_props.weak_damp))
                raise RuntimeError('Diverging in the {}\'th iteration, err : {:.2e}'.format(i+1, abs(err_i)))

            else:
                # Continue iteration
                #print(f'\rErr: {abs(err_i):.2e}', end="")
                err = err_i
                lhs = self.pss.Nblk - self.pss.toeA
                lhs_sp = csc_matrix(lhs)
                # Iteration step
                dx = spsolve(lhs_sp, rhs)
                dx2 = np.reshape(np.copy(dx), (self.Nx, -1), 'F')
                dx2[:, -1:-self.N - 1:-1] = np.conj(dx2[:, 0:self.N])
                dx2 = dx2.ravel('F')
                self.pss.cx += dx2
                # Calculate xt from fft coeffs
                cx = np.reshape(self.pss.cx, (self.Nx, -1), 'F')  # Reshape to 2-D format
                self.pss.xt = np.real(self.inv_dft(cx))
        raise RuntimeError('Reached maximum number of iterations')

    def dft(self, xin):
        """Computes the DFT for the given harmonic order"""
        return np.dot(xin, np.transpose(self.pss.exp_arr)) / self.pss.Nt

    def inv_dft(self, cs):
        """Computes the Inverse DFT for the given harmonic order"""
        return np.dot(cs, self.pss.exp_arr ** -1)

    def calc_var_td(self):
        # Construct time domain arrays for vars_t = [x t u p]
        self.pss.vars_t = [self.pss.xt[i, :] for i in range(self.Nx)]
        self.pss.vars_t.append(self.pss.t_arr)  # append time domain
        u_inp = [self.pss.ut[i, :] for i in range(self.Nu)]
        self.pss.vars_t += u_inp
        self.pss.vars_t += self.p_value

    def calc_lam_td(self, lamfun, vars_t):
        """
        Computes the time domain arrays for a lambdified function
        """
        fun_t = np.empty((len(lamfun), self.pss.Nt))
        # Compute state derivatives f in time domain
        for ind, f_i in enumerate(lamfun):
            y = f_i(*vars_t)
            if type(y) == np.ndarray:
                #if np.max(np.imag(y)) > 1e-1:
                #    print('Something wrong with idft, max imag = {}'.format(np.max(np.imag(y))))
                #    plt.plot(self.pss.t_arr,np.imag(self.pss.xt[0,:]))
                #    plt.show()

                fun_t[ind,:] = np.real(y)
            else: # If f_i contains no symbolic elements
                fun_t[ind, :] = np.ones(self.pss.Nt) * y
        return fun_t

    def toeplitz_setup(self, c, shp):
        """Sets up a harmonic series Toeplitz matrix"""
        nrows, ncols = shp
        # Numeric values (arrays) for toeplitz elements
        toe_pos = [c[:,i].reshape((nrows, ncols)) for i in range(self.N,2*self.N+1)]
        toe_neg = [c[:,i].reshape((nrows, ncols)) for i in range(self.N,-1,-1)]

        toe_neg += [np.zeros((nrows, ncols))]*self.N
        toe_pos += [np.zeros((nrows, ncols))] * self.N

        T = toe_pos+toe_neg

        # Calculate toeplitz matrix and reshape to proper size
        # Correct reshape is not very intuitive
        toe = self.toe_l(*T)
        toe = np.reshape(np.transpose(toe, (0, 2, 1, 3)), (nrows * (2 * self.N + 1), -1))
        return toe

    def toep_BCD(self):
        """
        Computes the toeplitz matrices B, C and D
        :return:
        """
        self.pss.Bt = self.calc_lam_td(self.B_lam, self.pss.vars_t)
        self.pss.Ct = self.calc_lam_td(self.C_lam, self.pss.vars_t)
        self.pss.Dt = self.calc_lam_td(self.D_lam, self.pss.vars_t)

        self.pss.cB = self.dft(self.pss.Bt)
        self.pss.cC = self.dft(self.pss.Ct)
        self.pss.cD = self.dft(self.pss.Dt)

        self.pss.toeB = self.toeplitz_setup(self.pss.cB, (self.Nx, self.Nu))
        self.pss.toeC = self.toeplitz_setup(self.pss.cC, (self.Ny, self.Nx))
        self.pss.toeD = self.toeplitz_setup(self.pss.cD, (self.Ny, self.Nu))

    def calc_modal_props(self):
        """Computes the modal properties for the current PSS"""

        # Recompute the Toeplitz A matrix for the current PSS
        #cx = np.reshape(self.pss.cx, (self.Nx, -1), 'F')  # Reshape to 2-D format
        #self.pss.xt = np.real(self.inv_dft(cx))
        #self.calc_td()

        # Compute eigenvalues and vectors of the HSS

        if self.h:
            Ht = self.calc_lam_td(self.H_lam, self.pss.vars_t)
            H_inv_t = self.calc_lam_td(self.H_inv_lam, self.pss.vars_t)

            cH = self.dft(Ht)
            cH_inv = self.dft(H_inv_t)

            #toeA = self.toeplitz_setup(cA, (self.Nx, self.Nx))
            toeH = self.toeplitz_setup(cH, (self.Nx, self.Nx))
            toeH_inv = self.toeplitz_setup(cH_inv, (self.Nx, self.Nx))

            toeA = np.dot(np.dot(self.pss.Nblk,toeH)-np.dot(toeH, self.pss.Nblk) + np.dot(toeH, self.pss.toeA),toeH_inv)
            #toeA = np.dot(toeH_inv, self.pss.toeA + np.dot(toeH, self.pss.Nblk)-np.dot(self.pss.Nblk, toeH))
            #toeA = np.dot(toeH, self.pss.toeA + np.dot(toeH_inv, self.pss.Nblk)-np.dot(self.pss.Nblk, toeH_inv))

        else:
            toeA = self.pss.toeA

        eigs, rev = eig(toeA - self.pss.Nblk)
        self.pss.modal_props.eigs = eigs
        lev = np.linalg.inv(rev)
        # Modes in the fundamental strip
        trc_inds = np.abs(np.imag(eigs)) <= 0.5*self.w0
        self.pss.modal_props.eig_fstrip = eigs[trc_inds]

        # Modes with high participation from zero-shift states
        p_f = np.multiply(lev.T, rev)
        p_f = np.abs(p_f) / np.abs(p_f).max(axis=0)
        pf_n0 = p_f[self.N*self.Nx:self.N*self.Nx+self.Nx]
        col_idxs = np.where(pf_n0 > 0.9999999)[1]
        col_idxs = np.unique(col_idxs)
        self.pss.modal_props.pf_x0 = pf_n0[:, col_idxs]
        self.pss.modal_props.eig_x0 = eigs[col_idxs]
        if col_idxs.size==0:
            raise ValueError('Eigenvalue selection failed')
        self.pss.modal_props.weak_damp = np.max(np.real(self.pss.modal_props.eig_x0))
        self.pss.modal_props.npf = np.zeros((self.Nx, len(col_idxs)), dtype=np.int8)
        for i in range(self.Nx):
            rowidxs = np.argmax(p_f[i::self.Nx, col_idxs],axis=0)
            self.pss.modal_props.pf_x0[i, :] = p_f[rowidxs * self.Nx + i, col_idxs]
            self.pss.modal_props.npf[i, :] = rowidxs - self.N

class PSS:
    """Dataclass for attributes associated with a Periodic Steady State"""
    def __init__(self, hss_model:HSS):
        """

        :param hss_model: The HSS model to which this PSS pertains
        :param n_t: Number of samples in time domain
        """
        nt = hss_model.N*2
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
        # NOTE: The state-space matrices above are flattened to 2 dimensions for compatibility with lambdify.
        # This causes readability issues and may be addressed in future updates.

        # Toeplitz arrays
        self.toeA = 1j * np.empty((nx*m, nx*m))
        self.toeB = 1j * np.empty((nx*m, nu*m))
        self.toeC = 1j * np.empty((ny*m, nx*m))
        self.toeD = 1j * np.empty((ny*m, nu*m))

        # Miscellaneous
        self.Nvec = np.linspace(-n,n,m)  # Harmonics vector
        diags = np.asarray([[1j*i*self.model.w0]*nx for i in range(-n, n+1)])
        self.Nblk = np.diag(diags.ravel())
        self.exp_arr = None  # complex phasors for reduced DFT

    def plot_states(self, ax):
        """Plots the states in time domain"""
        cx = np.reshape(self.cx, (self.model.Nx, -1), 'F')  # Reshape to 2-D format
        self.xt = np.real(self.model.inv_dft(cx))
        for ind, xx in enumerate(self.xt):
            ax.plot(self.t_arr, xx, label='{}'.format(self.model.x[ind]))
        plt.legend()
        return ax


class ModalProps:
    """Dataclass for the modal properties of a HSS"""
    def __init__(self):
        # Modal analysis
        self.eigs = [] # All frequency-shifted eigenvalues
        self.eig_fstrip = []  # In the fundamental strip
        self.eig_x0 = []  # Highest participation in the 0-shift states
        self.pf_x0 = np.empty(0)
        self.p_f = []
        self.weak_damp = 0
        self.npf = np.empty(0)
        # 2-param damping plot
        self.XYmesh = (None, None)
        self.damps = np.empty(0)

    def plot_eigs(self, ax):
        """
        Plots the
        :param ax:
        :return:
        """
        ax.scatter(np.real(self.eigs), np.imag(self.eigs), marker='x', color='k', alpha=0.5, linewidths=0.5, s=9, label=r'All $\lambda$')
        ax.plot(np.real(self.eig_x0), np.imag(self.eig_x0), 's', fillstyle='none', color='k', markersize=3, label=r'$\lambda_{x0}$')
        ax.plot(np.real(self.eig_fstrip), np.imag(self.eig_fstrip), 'o', fillstyle='none', color='k', markersize=3, label=r'$\lambda_{fstrip}$')

        plt.legend()
        plt.xlabel('Real', fontsize=10)
        plt.ylabel('Imag', fontsize=10)
        ax.grid('on')
