import numpy as np
import matplotlib.pyplot as plt


class HTF:
    def __init__(self, hss_model, freqs):
        """
        :param htm: numpy 3-dim array, shape: (# frequency points, # htm rows, #htm cols)
        """
        self.model = hss_model
        self.N = hss_model.N
        self.freqs = freqs
        self.htf = self.hss_to_htf(freqs)
        shp = np.shape(self.htf)
        self.nf = len(freqs)
        assert shp[1]==shp[2]

    def hss_to_htf(self, tol=1e-10):
        # Supports only first output and input variable
        sI = 2 * np.pi * 1j * np.eye((2 * self.model.N + 1) * self.model.Nx)
        sI = np.array([sI * k for k in self.freqs])
        return self.model.pss.toeC @ (np.linalg.inv(sI + self.model.pss.Nblk - self.model.pss.toeA) @ self.model.pss.toeB)

    def plot(self, Nplot=None, axs = np.empty(0), xscale = 'linear', yscale = 'linear'):
        if not Nplot or Nplot>self.N:
            Nplot = self.N
        if axs.size==0:
            fig, axs = plt.subplots(Nplot*2+1, Nplot*2+1, sharex=True)
        idxs = range(self.N-Nplot, self.N+Nplot+1)
        for n, row in enumerate(idxs):
            for m, col in enumerate(idxs):
                amp = np.abs(self.htf[:,row,col])
                if np.max(amp)>1e-10:
                    axs[n,m].plot(self.freqs, np.abs(self.htf[:,row,col]))
                    axs[n,m].set_yscale(yscale)
                    axs[n,m].set_xscale(xscale)

        return axs

    def get_mid_col(self, tol= 1e-2, N = 6):
        htfs = []
        idxs = []
        midcol = self.htf[:,:,self.N]
        maxhtf = np.max(abs(midcol), axis=1)
        for i in range(self.N * 2 + 1):
            if max(abs(self.htf[:, i, self.N])/maxhtf > tol) and abs(i-self.N)<N+1:
                htfs.append(np.array(self.htf[:, i, self.N]))
                idxs.append(i-self.N)
        return MidColHTF(self.freqs, htfs, idxs, self.N)

    def siso_eq(self, other):
        """

        :param other: Grid impedance (HTM)
        :return:
        """
        a = self.htf[:,:,self.N]
        a = np.reshape(np.delete(a, (self.N), axis=1),(len(self.freqs),1,2*self.N))
        b = self.htf[:,self.N,:]
        b = np.reshape(np.delete(b, (self.N), axis=1),(len(self.freqs),2*self.N,1))
        alph = other.htf[:,:,self.N]
        bet = other.htf[:,self.N,:]
        H0 = self.htf[:,self.N,self.N]
        Q = np.delete(self.htf, (self.N), axis=1)
        Q = np.delete(Q, (self.N), axis=2)
        otherQ = np.delete(other.htf, (self.N), axis=1)
        otherQ = np.delete(otherQ, (self.N), axis=2)
        fbk = np.linalg.inv(np.eye(self.N*2)+np.matmul(otherQ,Q))
        Hsiso = np.reshape(H0,(len(self.freqs),1,1))-np.matmul(np.matmul(a,fbk),np.matmul(otherQ,b))
        Hsiso = np.reshape(Hsiso, (len(self.freqs)))

        return SISOeq(self.freqs, Hsiso)

    def siso_rm0(self):
        return SISOeq(self.freqs, self.htf[:,self.N, self.N])

    def __invert__(self):
        return HTF(self.freqs, np.linalg.inv(self.htf))

    def __mul__(self, other):
        return HTF(self.freqs, np.einsum('nij,njk->nik',self.htf,other.htf))


class MidColHTF:
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
