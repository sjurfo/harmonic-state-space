from hss import HSS
from sympy import symbols, sin, cos, Matrix, atan2, atan
import numpy as np
from power_flow.read_data_with_trafos import read_data
from power_flow.nr_loadflow import newton_raphson, printsys


class LCLGrid(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.3
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(3)

    def setup(self):
        #print('Setup symbolic HSS')
        bus_data, line_data = read_data('../data/voc_statcom.xlsx')
        print('Newton-Raphson load flow started')
        newton_raphson(bus_data)
        printsys(bus_data)
        gridbus = bus_data[0]
        vocbus = bus_data[1]
        xf = line_data[0].xser
        rf = line_data[0].rser
        xg = line_data[1].xser
        rg = line_data[1].rser

        # Declare symbolic variables and parameters
        # Circuit
        uin = symbols('uin')
        ia = symbols('ia')
        ig, ucap = symbols('ig ucap')
        ug = gridbus.V*cos(self.w0 * self.t)*np.sqrt(2)
        uc = vocbus.V*cos(self.w0 * self.t+vocbus.angle)*np.sqrt(2)
        # dVOC

        #  Fixed and derived parameters
        #  Circuit
        L = xf / self.w0
        L1 = L * 2 / 3
        L2 = L * 1 / 3
        R = rf
        Lg = xg / self.w0
        Rg = rg

        flcl = 2000
        #C = (L+Lg)/(L1*(Lg+L2)*(2*np.pi*flcl)**2)
        #C = C*3.966/3
        C = 3e-6*self.zb
        print(C)

        # State space definition
        self.x = [ia, ig, ucap]
        self.u = [uin]           # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q/vocbus.V
        ia0 = I*np.sqrt(2)*sin(self.w0 * self.t-vocbus.angle)
        ig0 = ia0
        ucap0 = vocbus.V*np.sqrt(2)*cos(self.w0*self.t+vocbus.angle)
        self.x0 = Matrix([ia0, ig0, ucap0])

        # DAEs
        # Algebraic
        #yd = xd
        #yq = xq

        # Differential
        # Circuit
        fig = 1 / (Lg) * (ucap - (R+Rg) * ig - uin-ug)
        fia = 1/L*(uc-ucap)
        fucap = 1/C*(ia-ig)

        self.y = [-ig]                              # Outputs
        self.f = [fia, fig, fucap]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class Mathieu(HSS):
    def __init__(self):
        self.w0 = 50*2*np.pi #*2*np.pi
        self.vb = 0.2
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.zb = self.vb/self.ib
        super().__init__(13)

    def setup(self):
        #print('Setup symbolic HSS')

        x1,x2 = symbols('x1 x2')
        #q = 10
        #a = 15.5
        q = 1
        a = 2
        mu = 5
        #Parametric study
        #Lg = symbols('Lg')
        #Rg = symbols('Rg')
        #q = symbols('eta')
        q = symbols('q')
        #a = symbols('a')
        #a = q + bet
        #ksog = symbols('ksog')
        self.p = [q]
        p1 = np.linspace(1.6,1.7,20)
        #p2 = np.linspace(0.01,4, 20)
        #self.p = [q,a]
        self.paramvals = [p1]

        self.x = [x1,x2]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        # Outputs
        self.y = [100*(x1+x2)]

        # Initial conditions u
        self.u0 = [1e-15*sin(self.w0 * self.t)]

        # Initial conditions x (guess)
        x10 = 1e-15*sin(self.w0*self.t)
        x20 = 1e-15*sin(self.w0*self.t)
        self.x0 = Matrix([x10,x20])

        # DAEs
        # Algebraic
        # Differential
        fx1 = x2
        fx2 = -self.w0**2*(a-2*q*cos(self.w0*self.t))*x1-2*mu*x2+ug
        self.f = [fx1,fx2]

        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class MathieuLP(HSS):
    def __init__(self):
        self.w0 = 50*2*np.pi #*2*np.pi
        self.vb = 0.2
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.zb = self.vb/self.ib
        super().__init__(7)

    def setup(self):
        #print('Setup symbolic HSS')

        x1,x2,xlp = symbols('x1 x2 xlp')
        #q = 10
        #a = 15.5
        q = 1
        a = 2
        mu = 5
        T = 0.01
        #Parametric study
        #Lg = symbols('Lg')
        #Rg = symbols('Rg')
        #q = symbols('eta')
        #q = symbols('q')
        #a = symbols('a')
        #a = q + bet
        #ksog = symbols('ksog')
        #self.p = [q, alph]
        p1 = np.linspace(0.01,2,20)
        p2 = np.linspace(0.01,4, 20)
        #self.p = [q,a]
        #self.paramvals = [p1,p2]

        self.x = [x1,x2,xlp]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        # Outputs
        self.y = [xlp*100]

        # Initial conditions u
        self.u0 = [1e-15*sin(self.w0 * self.t)]

        # Initial conditions x (guess)
        x10 = 1e-15*sin(self.w0*self.t)
        x20 = 1e-15*sin(self.w0*self.t)
        xlp0 = 1e-15*sin(self.w0*self.t)
        self.x0 = Matrix([x10, x20, xlp0])

        # DAEs
        # Algebraic
        # Differential
        fx1 = x2
        fx2 = -self.w0**2*(a-2*q*cos(self.w0*self.t))*x1-2*mu*x2+ug
        fxlp = 1/T*(x1+x2-xlp)
        self.f = [fx1,fx2, fxlp]

        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class Lowpass(HSS):
    def __init__(self):
        self.w0 = 50*2*np.pi #*2*np.pi
        self.vb = 0.2
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.zb = self.vb/self.ib
        super().__init__(7)

    def setup(self):
        #print('Setup symbolic HSS')

        x1 = symbols('x1')
        T = 0.01
        self.x = [x1]

        # Inputs
        u = symbols('u')
        self.u = [u]

        # Outputs
        self.y = [x1]

        # Initial conditions u
        self.u0 = [1e-15*sin(self.w0 * self.t)]

        # Initial conditions x (guess)
        x10 = 1e-15*sin(self.w0*self.t)
        self.x0 = Matrix([x10])

        # DAEs
        # Algebraic
        # Differential
        fx1 = 1/T*(u-x1)
        self.f = [fx1]

        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


def compare_with_chirp():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('ytick', labelsize=15)
    df = pd.read_csv('data/mathieuchirp.csv', header=None)
    df2 = pd.read_csv('data/mathieu_chirp2.csv', header=None)

    """
    t = df2[0].values[::-1]
    y = df2[1].values*100
    
    fig, ax = plt.subplots()
    ax.plot(t,y)
    axin1 = ax.inset_axes([0.5, 0.6, 0.47, 0.37])
    x1 = int(134.75/200*2000001)
    x2 = int(135.25/200*2000001)
    axin1.plot(t[x1:x2],y[x1:x2])
    #ax.indicate_inset_zoom(axin1)
    ax.set_xlabel('time[s]')
    ax.set_ylabel('output y(t)')
    plt.show()
    """
    idx0 = df.index[abs(df[1]-200)<1e-2][0]
    f = df.loc[idx0:,1]
    ch_htfs = df.loc[idx0:,2:]
    t = df.loc[idx0:,0]
    k = np.diff(f)
    hss_sys = Mathieu()
    #hss_sys.parametric_sweep()
    #hss_sys.plot_parametric_study3d()
    #hss_sys.plot_parametric_study2d(levels=np.linspace(np.min(hss_sys.damps),0.01,50))
    #hss_sys.plot_parametric_study2d()
    hss_sys.find_pss()
    hss_sys.calc_eigs()
    hss_sys.pf_table()
    pfs = hss_sys.pf_filtered()
    import matplotlib.pyplot as plt
    #fig, axs = plt.subplots(2, sharex=True)
    hss_sys.toepBCD()
    lpsys = Lowpass()
    lpsys.find_pss()
    hss_sys.calc_eigs()
    lpsys.toepBCD()

    freqs = np.linspace(0,250,1000, endpoint=False)
    #freqs = np.append(-np.flip(freqs),freqs)
    H = hss_sys.hss_to_htf(freqs)
    #lphtf = lpsys.hss_to_htf(freqs)
    #axs = H.plot(Nplot=4)
    #lphtf.plot(Nplot=4, axs=axs)
    #H = lphtf*H
    #H.plot(Nplot=4, axs=axs)

    htfs, idxs = H.get_mid_col(tol=1e-15)
    fig, axs = plt.subplots(len(idxs)-1,1, sharex=True)
    for i, htf in enumerate(htfs):
        if i != 0 and i != 14:
            axs[i-1].plot(freqs,np.angle(htf))
            #axs[i-1].plot(f, ch_htfs.iloc[:,i], '--', alpha=0.6, lw=3)
            axs[i-1].text(0.9,0.4,f'n={idxs[i]}', transform=axs[i-1].transAxes, size='large')
    axs[-1].plot(f[1:],-k*1e4)
    axs[-1].text(0.9,0.4,f'rate k', transform=axs[i-1].transAxes, size='large',color='tab:blue')
    fig.text(0.5, 0.04, 'frequency[Hz]', ha='center', va='center', size='xx-large')
    fig.text(0.06, 0.5, 'magnitude [absolute value]', ha='center', va='center', rotation='vertical', size='xx-large')
    plt.xticks(size=15)
    plt.show()
    """
    #H.plot(Nplot=4)
    """
    #plt.show()
    #hss_sys.parametric_sweep()
    #hss_sys.plot_parametric_study3d()
    #hss_sys.plot_parametric_study2d()
    #hss_sys.find_pss()
    """
    
    #hss_sys.calc_eigs()
    """

lclg = LCLGrid()
#lclg.find_pss()