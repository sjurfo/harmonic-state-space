from hss import HSS
from sympy import symbols, sin, cos, Matrix
import numpy as np
import matplotlib.pyplot as plt
from power_flow.read_data_with_trafos import read_data
from power_flow.nr_loadflow import newton_raphson, printsys


from sympy import Function


class Rotate2d(Function):
    @classmethod
    def eval(cls, x1,x2, th):
        return x1*cos(th)-x2*sin(th), x1*sin(th)+x2*cos(th)


class ComplexMult(Function):
    @classmethod
    def eval(cls, x11,x12,x21,x22):
        return x11*x21-x12*x22, x11*x22+x12*x21


class dVOC_1ph(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.zb = self.vb/self.ib
        super().__init__(13)

    def setup(self):
        #print('Setup symbolic HSS')
        bus_data, line_data = read_data('../data/datavoc.xlsx')
        print('Newton-Raphson load flow started')
        newton_raphson(bus_data)
        printsys(bus_data)
        vocbus = bus_data[1]
        xf = line_data[0].xser
        rf = line_data[0].rser
        xg = line_data[1].xser
        rg = line_data[1].rser
        # Components
        # Circuit
        ia = symbols('ia')
        L = xf/self.w0
        R = rf
        Lg = xg/self.w0
        Rg = rg

        # dVOC
        va, vb = symbols('va vb')
        eta = 0.02  # eta gives the P-w droop, dw=eta*dP  in per unit.
        #eta = 5/(100*np.pi)
        mu = 0.1  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        #mu = 0.05
        qref = vocbus.Q
        vref = vocbus.V
        pref = vocbus.P_spec

        # SOGI
        xa, xb = symbols('xa xb')
        ksog = np.sqrt(2)
        #ksog = 3
        #ksog = 2

        # Virtual z
        xz1, xz2 = symbols('xz1 xz2')
        wh = (150*2*np.pi)#+2*np.pi*20
        wb = 0.3
        Kz = 100
        #Parametric study
        #wb = symbols('wb')
        #Lg = symbols('Lg')
        #Rg = symbols('Rg')
        #eta = symbols('eta')
        #mu = symbols('mu')
        #ksog = symbols('ksog')
        #self.p = [eta]
        #self.p = [ksog]
        #self.p = [kpdc, kidc]
        #self.p_range = [(10,1000,10), (0.0001,10,10)]
        #p1 = np.linspace(0.01/self.w0, 1/self.w0, 15)
        #p1 = 5*np.flip(np.logspace(-4, -2, 15))
        #p1 = np.logspace(-2, 5, 15)
        p1 = np.linspace(0.01,0.5,15)
        #ksog = symbols('ksog')
        #self.p = [wb]
        #p1 = np.sqrt(2)*(0.07+1.5*np.linspace(0,1,30))
        #p2 = np.logspace(-3, -1, 15)
        #self.paramvals = [p1]

        self.x = [ia, va, vb, xa, xb, xz1, xz2]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        # Outputs
        self.y = [-ia]

        # Initial conditions u
        self.u0 = [sin(self.w0 * self.t)]

        # Initial conditions x (guess)
        ia0 = sin(self.w0 * self.t)
        va0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vb0 = -vref*cos(self.w0 * self.t+vocbus.angle)
        xa0 = sin(self.w0 * self.t)
        xb0 = -cos(self.w0 * self.t)
        self.x0 = Matrix([ia0, va0, vb0, xa0, xb0, 0, 0])

        # DAEs
        # Algebraic

        uc = va-Kz*xz1
        phi = (1 - (va ** 2 + vb ** 2) / vref ** 2)/2
        yb = xb
        ya = xa
        unorm = va ** 2 + vb ** 2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fia = 1 / (L+Lg) * (uc - (R+Rg) * ia - ug)
        # VOC
        fva = -self.w0 * vb + eta * self.w0 * (1 / mu * phi * va + 1 / unorm * (va * qref - vb * pref) + yb)
        fvb = self.w0 * va + eta * self.w0 * (1 / mu * phi * vb + 1 / unorm * (pref * va + qref * vb) - ya)

        # SOGI
        ##
        fxa = (ksog * (ia - ya) - yb) * self.w0
        fxb = ya * self.w0

        # Virtual z
        fxz1 = -wh**2*xz2+wb*(ia-xz1)
        fxz2 = xz1

        self.f = [fia, fva, fvb, fxa, fxb, fxz1, fxz2]

        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


    def plotpss(self):
        ax = super().plot_states()
        va = self.xt[1,:]
        vb = self.xt[2,:]
        ia = self.xt[3,:]
        ib = self.xt[4,:]
        p = va*ia+vb*ib
        q = -va*ib+vb*ia
        ax.plot(self.t_arr, p,label='p')
        ax.plot(self.t_arr, q,label='q')
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)


class dVOC_1ph_dq(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.126
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.zb = self.vb/self.ib
        super().__init__(9)

    def setup(self):
        #print('Setup symbolic HSS')
        bus_data, line_data = read_data('../data/datavoc.xlsx')
        print('Newton-Raphson load flow started')
        newton_raphson(bus_data)
        printsys(bus_data)
        vocbus = bus_data[1]
        xf = line_data[0].xser
        rf = line_data[0].rser
        xg = line_data[1].xser
        rg = line_data[1].rser
        # Components
        # Circuit
        ia = symbols('ia')
        L = xf/self.w0
        R = rf
        Lg = xg/self.w0
        Rg = rg

        # dVOC
        vd, vq = symbols('vd vq')
        eta = 0.07  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = 0.1  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        qref = vocbus.Q
        vref = vocbus.V
        pref = vocbus.P_spec

        # SOGI
        xa, xb = symbols('xa xb')
        ksog = np.sqrt(2)*0.5
        #ksog = 0.5
        # CNF
        #ksog = 2

        # Virtual z
        xz1, xz2 = symbols('xz1 xz2')
        wh = (150*2*np.pi)#+2*np.pi*20
        wb = 0.3
        Kz = 100

        #Parametric study
        #ksog = symbols('ksog')
        #self.p = [ksog]
        p1 = np.sqrt(2)*np.linspace(0.2,4,50)
        #p1 = np.linspace(0.5, 1, 50)
        #p1 = np.linspace(0.2,4,50)
        #self.paramvals = [p1]

        # State space definition
        self.x = [ia, vd, vq, xa, xb, xz1, xz2]
        ug = symbols('ug')
        self.u = [ug]                               # Inputs
        self.y = [-ia]                              # Outputs
        self.u0 = [cos(self.w0 * self.t)]           # Initial conditions u
        # Initial conditions x (guess)
        ia0 = 0.1*sin(self.w0 * self.t)
        vd0 = vref*cos(vocbus.angle)
        vq0 = vref*sin(vocbus.angle)
        xa0 = 0.1*sin(self.w0 * self.t)
        xb0 = 0.1*-cos(self.w0 * self.t)
        self.x0 = Matrix([ia0, vd0, vq0, xa0, xb0,0,0])

        # Reference frame
        th = self.w0*self.t#+vocbus.angle

        # DAEs
        # Algebraic
        va = vd*cos(th)-vq*sin(th)
        vb = vd*sin(th)+vq*cos(th)
        #xa = xd * cos(th) - xq * sin(th)
        #xb = xd * sin(th) + xq * cos(th)
        uc = va-Kz*xz1
        phi = (1 - (vd ** 2 + vq ** 2) / vref ** 2)/2
        yb = xb
        ya = xa
        unorm = va ** 2 + vb ** 2
        #unorm = vref ** 2

        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)

        yd = ya * cos(th) + yb * sin(th)
        yq = -ya * sin(th) + yb * cos(th)
        #yd = xd
        #yq = xq

        # Differential
        # Circuit
        fia = 1 / (L+Lg) * (uc - (R+Rg) * ia - ug)
        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd + (yq-iqref))
        fvq = eta * self.w0 * (1 / mu * phi * vq + (idref-yd))
        # Virtual z
        fxz1 = -wh**2*xz2+wb*(ia-xz1)
        fxz2 = xz1
        # SOGI
        fxa = (ksog * (ia - xa) - xb) * self.w0
        fxb = xa * self.w0
        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        self.f = [fia, fvd, fvq, fxa, fxb, fxz1, fxz2]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()
        vd = self.xt[1,:]
        vq = self.xt[2,:]
        ya = self.xt[3,:]
        yb = self.xt[4,:]
        yd = ya * np.cos(self.w0 * self.t_arr) + yb * np.sin(self.w0 * self.t_arr)
        yq = -ya * np.sin(self.w0 * self.t_arr) + yb * np.cos(self.w0 * self.t_arr)
        p = vd*yd+vq*yq
        q = -vd*yq+vq*yd
        ax.plot(self.t_arr, p,label='p')
        ax.plot(self.t_arr, q,label='q')
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)


class dVOC_1ph_dq_LCL(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.126
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.zb = self.vb/self.ib
        super().__init__(9)

    def setup(self):
        #print('Setup symbolic HSS')
        bus_data, line_data = read_data('../data/datavoc.xlsx')
        print('Newton-Raphson load flow started')
        newton_raphson(bus_data)
        printsys(bus_data)
        vocbus = bus_data[1]
        xf = line_data[0].xser
        rf = line_data[0].rser
        xg = line_data[1].xser
        rg = line_data[1].rser
        # Components
        # Circuit
        
        ia = symbols('ia')
        ic, ucap = symbols('ic ucap')
        L = xf/self.w0
        L1 = L*2/3
        L2 = L*1/3
        R = rf
        Lg = xg/self.w0
        Rg = rg
        C = 1e-5

        # dVOC
        vd, vq = symbols('vd vq')
        eta = 0.03  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = 0.1  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        qref = vocbus.Q
        vref = vocbus.V
        pref = vocbus.P_spec

        # SOGI
        xa, xb = symbols('xa xb')
        ksog = np.sqrt(2)*0.1
        #ksog = 0.5
        # CNF
        #ksog = 2

        # Virtual z
        xz1, xz2 = symbols('xz1 xz2')
        wh = (150*2*np.pi)#+2*np.pi*20
        wb = 0.3
        Kz = 100

        #Parametric study
        #ksog = symbols('ksog')
        flcl = symbols('flcl')
        self.p = [flcl]
        C = (L+Lg)/(L1*(Lg+L2)*(2*np.pi*flcl)**2)
        p1 = np.sqrt(2)*np.linspace(0.2,4,50)
        #p1 = np.linspace(0.5, 1, 50)
        #p1 = np.linspace(0.2,4,50)
        p1 = np.logspace(2,4,15)
        self.paramvals = [p1]

        # State space definition
        self.x = [ia, ic, ucap, vd, vq, xa, xb, xz1, xz2]
        ug = symbols('ug')
        self.u = [ug]                               # Inputs
        self.y = [-ia]                              # Outputs
        self.u0 = [cos(self.w0 * self.t)]           # Initial conditions u
        # Initial conditions x (guess)
        ia0 = 0.1*sin(self.w0 * self.t)
        ic0 = 0
        vcap0 = cos(self.w0 * self.t)
        vd0 = vref*cos(vocbus.angle)
        vq0 = vref*sin(vocbus.angle)
        xa0 = 0.1*sin(self.w0 * self.t)
        xb0 = 0.1*-cos(self.w0 * self.t)
        self.x0 = Matrix([ia0, ic0, vcap0, vd0, vq0, xa0, xb0,0,0])

        # Reference frame
        th = self.w0*self.t#+vocbus.angle

        # DAEs
        # Algebraic
        va = vd*cos(th)-vq*sin(th)
        vb = vd*sin(th)+vq*cos(th)
        #xa = xd * cos(th) - xq * sin(th)
        #xb = xd * sin(th) + xq * cos(th)
        uc = va-Kz*xz1
        phi = (1 - (vd ** 2 + vq ** 2) / vref ** 2)/2
        yb = xb
        ya = xa
        unorm = va ** 2 + vb ** 2
        #unorm = vref ** 2

        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)

        yd = ya * cos(th) + yb * sin(th)
        yq = -ya * sin(th) + yb * cos(th)
        #yd = xd
        #yq = xq

        # Differential
        # Circuit
        fia = 1 / (L2+Lg) * (ucap - (R+Rg) * ia - ug)
        fic = 1/L1*(uc-ucap)
        fucap = 1/C*(ic-ia)

        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd + (yq-iqref))
        fvq = eta * self.w0 * (1 / mu * phi * vq + (idref-yd))
        # Virtual z
        fxz1 = -wh**2*xz2+wb*(ia-xz1)
        fxz2 = xz1
        # SOGI
        fxa = (ksog * (ia - xa) - xb) * self.w0
        fxb = xa * self.w0
        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        self.f = [fia, fic, fucap, fvd, fvq, fxa, fxb, fxz1, fxz2]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()
        vd = self.xt[1,:]
        vq = self.xt[2,:]
        ya = self.xt[3,:]
        yb = self.xt[4,:]
        yd = ya * np.cos(self.w0 * self.t_arr) + yb * np.sin(self.w0 * self.t_arr)
        yq = -ya * np.sin(self.w0 * self.t_arr) + yb * np.cos(self.w0 * self.t_arr)
        p = vd*yd+vq*yq
        q = -vd*yq+vq*yd
        ax.plot(self.t_arr, p,label='p')
        ax.plot(self.t_arr, q,label='q')
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)


class dVOC_Statcom(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.3
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(13)

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
        #rg = symbols('rg')
        # Declare symbolic variables and parameters
        # Circuit
        ug = gridbus.V*np.sqrt(2)*cos(self.w0*self.t)
        uin = symbols('uin')
        ia = symbols('ia')
        ig, ucap = symbols('ig ucap')
        udc = symbols('udc')
        flcl = symbols('flcl')  # Resonance LCL filter
        # dVOC
        vd, vq = symbols('vd vq')
        #va, vb = symbols('va vb')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        xa2, xb2 = symbols('xa2 xb2') # DC compensation
        ksog = symbols('ksog')
        # DC control
        xdc = symbols('xdc')
        kpdc = symbols('kpdc')
        Cdc_r = symbols('Cdc_r')
        Tdc = symbols('Tdc')
        Rg = symbols('Rg')
        Rvir = symbols('Rvir')
        kp = symbols('kp')
        # Parametric study
        #self.p = [kp]  # Declare symbolic variables for parametric study
        #p1 = np.logspace(-4,-1,10)
        #p1 = np.linspace(0.005,0.1, 20)
        p1 = np.linspace(0.01,0.1,30)
        self.paramvals = [p1]
        #p1 = 2*np.logspace(2,3,10)
        #p1 = np.linspace(1,10,15)*Cdc0
        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        eta = 0.03
        mu = 0.1
        kpdc = 1
        flcl = 2000
        Cdc_r = 0.2e-3
        Tdc = 0.01
        Rvir = 0.2
        kp = 0

        #  Fixed and derived parameters
        #  Circuit
        L = xf / self.w0
        R = rf
        Lg = xg / self.w0
        Rg = rg
        #C = (L+Lg)/(L1*(Lg+L2)*(2*np.pi*flcl)**2)
        #print(C)
        #C = C*3.96/3
        C = 4.7e-6*self.zb
        dv = np.abs(vocbus.P + 1j * vocbus.Q) * 1e3 / (2 * np.pi * 100 * Cdc_r * self.vdcb * 1e3) / (self.vdcb * 1e3)
        print(f'dv = {dv}')
        Cdc = Cdc_r * self.vdcb ** 2 / self.sb

        #  dVoc
        qref = vocbus.Q
        #pref = vocbus.P_spec
        #xvir = symbols('xvir')
        vref = vocbus.V
        # DC
        kidc = kpdc ** 2 * np.sqrt(2)
        udcref = 1.0

        # State space definition
        self.x = [ia, ig, ucap, vd, vq, xa, xb, udc, xdc, xa2, xb2]
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = I * np.sqrt(2) * sin(self.w0 * self.t - vocbus.angle)
        ig0 = ia0
        xa0 = ia0/np.sqrt(2)
        xb0 = I * cos(self.w0 * self.t - vocbus.angle)
        ucap0 = vref*np.sqrt(2)*cos(self.w0*self.t+vocbus.angle)
        udc0 = udcref-0.1
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vd0 = vref*cos(vocbus.angle)
        vq0 = vref*sin(vocbus.angle)
        self.x0 = Matrix([ia0, ig0, ucap0, vd0, vq0, xa0, xb0, udc0, 0, 0, 0])

        # Reference frame
        th = self.w0*self.t+0.5#+vocbus.angle

        # DAEs
        # Algebraic
        # Rotate state variables
        yd, yq = Rotate2d(xa,xb,-th)
        # DC control
        edc = -1 * (udcref ** 2 - udc ** 2)
        pref = (xdc + kpdc * edc)
        # dVOC inputs
        unorm = vd ** 2 + vq ** 2
        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)
        e_id = idref-yd
        e_iq = iqref-yq
        vyd = vd - e_iq*kp
        vyq = vq + e_id*kp
        va, vb = Rotate2d(vyd,vyq,th)
        #e_ia, e_ib = Rotate2d(e_id,e_iq,th)
        va = va
        mod = va*np.sqrt(2)-Rvir*ia
        #mod = mod/udc  # CM mode
        mod = mod/(udcref+xa2)
        uc = mod*udc

        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fig = 1 / Lg * (ucap - Rg * ig - uin-ug)
        fia = 1/L*(uc-ucap-R*ia)
        fucap = 1/C*(ia-ig)
        fudc = -1 / Cdc * (mod * ia)

        # DC
        fxdc = kidc * edc

        # dVOC

        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd - e_iq)
        fvq = eta * self.w0 * (1 / mu * phi * vq + e_id)

        # SOGI
        fxa = (ksog * (ig/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        # modulation
        fxa2 = (1 / (Tdc * 2 * self.w0) * ((udc - 1) - xa2) - xb2) * 2 * self.w0
        fxb2 = xa2 * 2 * self.w0

        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/0.1*(rvir*ia-xvir)
        self.y = [-ig]                              # Outputs
        self.f = [fia, fig, fucap, fvd, fvq, fxa, fxb, fudc, fxdc, fxa2, fxb2]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()

        vd = self.xt[3,:]
        vq = self.xt[4,:]

        ia = self.xt[5,:]
        ib = self.xt[6,:]
        idq = np.exp(-1j*(self.w0*self.t_arr))*(ia+1j*ib)
        id, iq = idq.real,idq.imag
        p = vd*id+vq*iq
        q = -vd*iq+vq*id
        ax.plot(self.t_arr, p,'--', label='p', lw=2)
        ax.plot(self.t_arr, q,'--',label='q', lw=2)
        #ax.plot(self.t_arr, vd,'--', label='vd2', lw=2)
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)


class dVOC_Statcom_Tsukuba(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.32
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(13)

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
        #rg = symbols('rg')
        # Declare symbolic variables and parameters
        # Circuit
        ug = gridbus.V*np.sqrt(2)*cos(self.w0*self.t)
        uin = symbols('uin')
        ia = symbols('ia')
        udc = symbols('udc')
        # dVOC
        #vd, vq = symbols('vd vq')
        vd, vq = symbols('vd vq')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        xa2, xb2 = symbols('xa2 xb2') # DC compensation
        ksog = symbols('ksog')
        # DC control
        xdc = symbols('xdc')
        kpdc = symbols('kpdc')
        Cdc_r = symbols('Cdc_r')
        Tdc = symbols('Tdc')
        Rg = symbols('Rg')
        Rvir = symbols('Rvir')
        kp = symbols('kp')
        # Parametric study
        #self.p = [Rvir]  # Declare symbolic variables for parametric study
        #p1 = np.logspace(-4,-1,10)
        p1 = np.linspace(0.01,0.1, 20)
        #p1 = np.linspace(0.01,0.3,30)
        #p1 = np.linspace(0.5,4,30)
        self.paramvals = [p1]
        #p1 = 2*np.logspace(2,3,10)
        #p1 = np.linspace(1,10,15)*Cdc0
        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        eta = 0.05
        mu = 0.01
        kpdc = 3
        Cdc_r = 0.2e-3
        Tdc = 0.01
        Rvir = 0.05+0.015
        kp = 0

        #  Fixed and derived parameters
        #  Circuit
        L = xf / self.w0
        R = rf
        Lg = xg / self.w0
        Rg = rg
        Cdc = Cdc_r * self.vdcb ** 2 / self.sb

        #  dVoc
        qref = vocbus.Q
        qref = 0.0
        #pref = vocbus.P_spec
        #xvir = symbols('xvir')
        vref = vocbus.V
        # DC
        kidc = kpdc ** 2 * np.sqrt(2)
        udcref = 1.0

        # State space definition
        self.x = [ia, vd, vq, xa, xb, udc, xdc, xa2, xb2]
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = I * np.sqrt(2) * sin(self.w0 * self.t - vocbus.angle)
        xa0 = ia0/np.sqrt(2)
        xb0 = I * cos(self.w0 * self.t - vocbus.angle)
        udc0 = udcref
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vd0 = vref*cos(vocbus.angle)
        vq0 = vref*sin(vocbus.angle)
        self.x0 = Matrix([ia0, vd0, vq0, xa0, xb0, udc0, 0, 0, 0])

        # Reference frame
        th = self.w0*self.t
        yd, yq = Rotate2d(xa, xb, -th)
        # DAEs
        # Algebraic
        # Rotate state variables
        # DC control
        edc = -1 * (udcref ** 2 - udc ** 2)
        pref = (xdc + kpdc * edc)
        # dVOC inputs
        unorm = vd ** 2 + vq ** 2
        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)
        e_id = idref-yd
        e_iq = iqref-yq
        va, vb = Rotate2d(vd, vq, th)
        mod = va*np.sqrt(2)-Rvir*ia
        #mod = mod/udc  # CM mode
        mod = mod/(udcref+xa2)
        uc = mod*udc

        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fia = 1/L*(uc - R*ia - uin - ug)
        fudc = -1 / Cdc * (mod * ia)

        # DC
        fxdc = kidc * edc

        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd - e_iq)
        fvq = eta * self.w0 * (1 / mu * phi * vq + e_id)

        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        # modulation
        fxa2 = (1 / (Tdc * 2 * self.w0) * ((udc - 1) - xa2) - xb2) * 2 * self.w0
        fxb2 = xa2 * 2 * self.w0

        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/0.1*(rvir*ia-xvir)
        self.y = [-ia]                              # Outputs
        self.f = [fia, fvd, fvq, fxa, fxb, fudc, fxdc, fxa2, fxb2]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()

        vd = self.xt[1,:]
        vq = self.xt[2,:]

        ia = self.xt[3,:]
        ib = self.xt[4,:]
        idq = np.exp(-1j*(self.w0*self.t_arr))*(ia+1j*ib)
        id, iq = idq.real,idq.imag
        p = vd*id+vq*iq
        q = -vd*iq+vq*id
        ax.plot(self.t_arr, p,'--', label='p', lw=2)
        ax.plot(self.t_arr, q,'--',label='q', lw=2)
        #ax.plot(self.t_arr, vd,'--', label='vd2', lw=2)
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)


class dVOC_Statcom_Tsukuba_orig(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.32
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(13)

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
        #rg = symbols('rg')
        # Declare symbolic variables and parameters
        # Circuit
        uin = symbols('uin')
        ia = symbols('ia')
        udc = symbols('udc')
        # dVOC
        #vd, vq = symbols('vd vq')
        vd, vq = symbols('vd vq')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        xa2, xb2 = symbols('xa2 xb2') # DC compensation
        ksog = symbols('ksog')
        # DC control
        xdc = symbols('xdc')
        kpdc = symbols('kpdc')
        Cdc_r = symbols('Cdc_r')
        Tdc = symbols('Tdc')
        Rg = symbols('Rg')
        Rvir = symbols('Rvir')
        kp = symbols('kp')
        thgrid = 0 #np.pi/2
        ug = gridbus.V*np.sqrt(2)*sin(self.w0*self.t+thgrid)
        # Parametric study
        #self.p = [Rvir]  # Declare symbolic variables for parametric study
        #p1 = np.logspace(-4,-1,10)
        #p1 = np.linspace(0.005,0.1, 20)
        p1 = np.linspace(0.01,0.1,30)
        #p1 = np.linspace(0.5,4,30)
        self.paramvals = [p1]
        #p1 = 2*np.logspace(2,3,10)
        #p1 = np.linspace(1,10,15)*Cdc0
        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        eta = 0.05
        mu = 0.01
        kpdc = 3
        Cdc_r = 0.2e-3
        Tdc = 0.01
        Rvir = 0.05+0.015
        kp = 0

        #  Fixed and derived parameters
        #  Circuit
        L = xf / self.w0
        R = rf
        Lg = xg / self.w0
        Rg = rg
        Cdc = Cdc_r * self.vdcb ** 2 / self.sb

        #  dVoc
        qref = vocbus.Q
        qref = 0.0
        #pref = vocbus.P_spec
        #xvir = symbols('xvir')
        vref = vocbus.V
        # DC
        kidc = kpdc ** 2 * np.sqrt(2)
        udcref = 1.0

        # State space definition
        self.x = [ia, vd, vq, xa, xb, udc, xdc, xa2, xb2]
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = -I * np.sqrt(2) * cos(self.w0 * self.t - vocbus.angle+thgrid)
        xa0 = ia0/np.sqrt(2)
        xb0 = I * sin(self.w0 * self.t - vocbus.angle+thgrid)
        udc0 = udcref
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vd0 = vref*sin(vocbus.angle+thgrid)
        vq0 = -vref*cos(vocbus.angle+thgrid)

        self.x0 = Matrix([ia0, vd0, vq0, xa0, xb0, udc0, 0, 0, 0])

        # Reference frame
        th = self.w0*self.t
        yd, yq = Rotate2d(xa, xb, -th)
        # DAEs
        # Algebraic
        # Rotate state variables
        # DC control
        edc = -1 * (udcref ** 2 - udc ** 2)
        pref = (xdc + kpdc * edc)
        # dVOC inputs
        unorm = vd ** 2 + vq ** 2
        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)
        e_id = idref-yd
        e_iq = iqref-yq
        va, vb = Rotate2d(vd, vq, th)
        mod = va*np.sqrt(2)-Rvir*ia
        #mod = mod/udc  # CM mode
        mod = mod/(udcref+xa2)
        uc = mod*udc

        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fia = 1/L*(uc - R*ia - uin - ug)
        fudc = -1 / Cdc * (mod * ia)

        # DC
        fxdc = kidc * edc

        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd - e_iq)
        fvq = eta * self.w0 * (1 / mu * phi * vq + e_id)

        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        # modulation
        fxa2 = (1 / (Tdc * 2 * self.w0) * ((udc - 1) - xa2) - xb2) * 2 * self.w0
        fxb2 = xa2 * 2 * self.w0

        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/0.1*(rvir*ia-xvir)
        self.y = [-ia]                              # Outputs
        self.f = [fia, fvd, fvq, fxa, fxb, fudc, fxdc, fxa2, fxb2]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()

        vd = self.xt[1,:]
        vq = self.xt[2,:]

        ia = self.xt[3,:]
        ib = self.xt[4,:]
        idq = np.exp(-1j*(self.w0*self.t_arr))*(ia+1j*ib)
        id, iq = idq.real,idq.imag
        p = vd*id+vq*iq
        q = -vd*iq+vq*id
        ax.plot(self.t_arr, p,'--', label='p', lw=2)
        ax.plot(self.t_arr, q,'--',label='q', lw=2)
        #ax.plot(self.t_arr, vd,'--', label='vd2', lw=2)
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)



class dVOC_Statcom_Tsukuba_mod(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.32
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(13)

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
        #rg = symbols('rg')
        # Declare symbolic variables and parameters
        # Circuit
        uin = symbols('uin')
        ia = symbols('ia')
        udc = symbols('udc')
        # dVOC
        #vd, vq = symbols('vd vq')
        vd, vq = symbols('vd vq')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        xa2, xb2 = symbols('xa2 xb2') # DC compensation
        ksog = symbols('ksog')
        # DC control
        xdc = symbols('xdc')
        kpdc = symbols('kpdc')
        Cdc_r = symbols('Cdc_r')
        Tdc = symbols('Tdc')
        Rg = symbols('Rg')
        Rvir = symbols('Rvir')
        kp = symbols('kp')
        #vref = symbols('vref')

        thgrid =np.pi/2.
        ug = gridbus.V*np.sqrt(2)*sin(self.w0*self.t+thgrid)
        # Parametric study
        self.p = [mu]  # Declare symbolic variables for parametric study
        #p1 = np.logspace(-3,-1,30)
        p1 = np.linspace(0.01,0.1, 20)
        #p1 = np.linspace(0.01,0.1,30)
        #p1 = np.linspace(0.5,4,30)
        #p1 = np.linspace(0.9, 1, 20)
        self.paramvals = [p1]
        #p1 = 2*np.logspace(2,3,10)
        #p1 = np.linspace(1,10,15)*Cdc0
        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        eta = 0.05
        #mu = 0.01
        kpdc = 3
        Cdc_r = 0.2e-3
        Tdc = 0.01
        Rvir = 0.05 + 0.015
        kp = 0

        #  Fixed and derived parameters
        #  Circuit
        L = xf
        R = rf
        Lg = xg
        Rg = rg
        Cdc = Cdc_r * self.vdcb ** 2 / self.sb * self.w0
        #Cdc = 500

        #  dVoc
        qref = vocbus.Q
        #qref = 0.0
        #pref = vocbus.P_spec
        #xvir = symbols('xvir')
        vref = vocbus.V

        # DC
        kidc = kpdc ** 2 * np.sqrt(2)
        udcref = 1.0

        # State space definition
        self.x = [ia, vd, vq, xa, xb, udc, xdc, xa2, xb2]
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = -I * np.sqrt(2) * cos(self.w0 * self.t - vocbus.angle+thgrid)
        xa0 = ia0/np.sqrt(2)
        xb0 = I * sin(self.w0 * self.t - vocbus.angle+thgrid)
        udc0 = udcref
        vref0 = 1
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vd0 = vref0*sin(vocbus.angle+thgrid)
        vq0 = -vref0*cos(vocbus.angle+thgrid)
        self.x0 = Matrix([ia0, vd0, vq0, xa0, xb0, udc0, 0, 0, 0])

        # Reference frame
        th = self.w0*self.t
        yd, yq = Rotate2d(xa, xb, -th)
        # DAEs
        # Algebraic
        # Rotate state variables
        # DC control
        edc = -1 * (udcref ** 2 - udc ** 2)
        pref = (xdc + kpdc * edc)
        # dVOC inputs
        unorm = vd ** 2 + vq ** 2
        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)
        e_id = idref-yd
        e_iq = iqref-yq
        va, vb = Rotate2d(vd, vq, th)
        mod = va*np.sqrt(2)-Rvir*ia
        #mod = mod/udc  # CM mode
        mod = mod/(udcref+xa2)
        uc = mod*udc

        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fia = 1/(L/self.w0)*(uc - R*ia - uin - ug)
        fudc = -1 / (Cdc/self.w0) * (mod * ia)

        # DC
        fxdc = kidc * edc

        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd - e_iq)
        fvq = eta * self.w0 * (1 / mu * phi * vq + e_id)

        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        # modulation
        fxa2 = (1 / (Tdc * 2 * self.w0) * ((udc - 1) - xa2) - xb2) * 2 * self.w0
        fxb2 = xa2 * 2 * self.w0

        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/0.1*(rvir*ia-xvir)
        self.y = [-ia]                              # Outputs
        self.f = [fia, fvd, fvq, fxa, fxb, fudc, fxdc, fxa2, fxb2]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()

        vd = self.xt[1,:]
        vq = self.xt[2,:]

        ia = self.xt[3,:]
        ib = self.xt[4,:]
        idq = np.exp(-1j*(self.w0*self.t_arr))*(ia+1j*ib)
        id, iq = idq.real,idq.imag
        p = vd*id+vq*iq
        q = -vd*iq+vq*id
        ax.plot(self.t_arr, p,'--', label='p', lw=2)
        ax.plot(self.t_arr, q,'--',label='q', lw=2)
        #ax.plot(self.t_arr, vd,'--', label='vd2', lw=2)
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)



class dVOC_GridCon_Tsukuba(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.3
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(13)

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
        #rg = symbols('rg')
        # Declare symbolic variables and parameters
        # Circuit
        ug = gridbus.V*np.sqrt(2)*cos(self.w0*self.t)
        uin = symbols('uin')
        ia = symbols('ia')
        udc = symbols('udc')
        # dVOC
        #vd, vq = symbols('vd vq')
        va, vb = symbols('va vb')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        xa2, xb2 = symbols('xa2 xb2') # DC compensation
        ksog = symbols('ksog')
        # DC control
        xdc = symbols('xdc')
        kpdc = symbols('kpdc')
        Cdc_r = symbols('Cdc_r')
        Tdc = symbols('Tdc')
        Rg = symbols('Rg')
        Rvir = symbols('Rvir')
        kp = symbols('kp')
        # Parametric study
        self.p = [Rvir]  # Declare symbolic variables for parametric study
        #p1 = np.logspace(-4,-1,10)
        #p1 = np.linspace(0.005,0.1, 20)
        p1 = np.linspace(0.01,0.3,30)
        self.paramvals = [p1]
        #p1 = 2*np.logspace(2,3,10)
        #p1 = np.linspace(1,10,15)*Cdc0
        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        eta = 0.03
        mu = 0.1
        kpdc = 1
        Cdc_r = 0.2e-3
        Tdc = 0.01
        #Rvir = 0.2
        kp = 0

        #  Fixed and derived parameters
        #  Circuit
        L = xf / self.w0
        R = rf
        Lg = xg / self.w0
        Rg = rg
        Cdc = Cdc_r * self.vdcb ** 2 / self.sb

        #  dVoc
        qref = vocbus.Q
        #pref = vocbus.P_spec
        #xvir = symbols('xvir')
        vref = vocbus.V
        # DC
        kidc = kpdc ** 2 * np.sqrt(2)
        udcref = 1.0

        # State space definition
        self.x = [ia, va, vb, xa, xb, udc, xdc, xa2, xb2]
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = I * np.sqrt(2) * sin(self.w0 * self.t - vocbus.angle)
        xa0 = ia0/np.sqrt(2)
        xb0 = I * cos(self.w0 * self.t - vocbus.angle)
        udc0 = udcref-0.2
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        va0 = vref*cos(self.w0*self.t)
        vb0 = vref*sin(self.w0*self.t)
        self.x0 = Matrix([ia0, va0, vb0, xa0, xb0, udc0, 0, 0, 0])

        # Reference frame

        # DAEs
        # Algebraic
        # Rotate state variables
        # DC control
        edc = -1 * (udcref ** 2 - udc ** 2)
        pref = (xdc + kpdc * edc)
        # dVOC inputs
        unorm = va ** 2 + vb ** 2
        iaref = 1 / unorm * (pref*va+qref*vb)
        ibref = 1 / unorm * (pref*vb-qref*va)
        e_ia = iaref-xa
        e_ib = ibref-xb
        #e_ia, e_ib = Rotate2d(e_id,e_iq,th)
        mod = va*np.sqrt(2)-Rvir*ia
        #mod = mod/udc  # CM mode
        mod = mod/(udcref+xa2)
        uc = mod*udc

        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fia = 1/L*(uc - R*ia - uin - ug)
        fudc = -1 / Cdc * (mod * ia)

        # DC
        fxdc = kidc * edc

        # VOC
        fva = -self.w0 * vb + eta * self.w0 * (1 / mu * phi * va - e_ib)
        fvb = self.w0 * va + eta * self.w0 * (1 / mu * phi * vb + e_ia)
        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        # modulation
        fxa2 = (1 / (Tdc * 2 * self.w0) * ((udc - 1) - xa2) - xb2) * 2 * self.w0
        fxb2 = xa2 * 2 * self.w0

        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/0.1*(rvir*ia-xvir)
        self.y = [-ia]                              # Outputs
        self.f = [fia, fva, fvb, fxa, fxb, fudc, fxdc, fxa2, fxb2]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()

        vd = self.xt[3,:]
        vq = self.xt[4,:]

        ia = self.xt[5,:]
        ib = self.xt[6,:]
        idq = np.exp(-1j*(self.w0*self.t_arr))*(ia+1j*ib)
        id, iq = idq.real,idq.imag
        p = vd*id+vq*iq
        q = -vd*iq+vq*id
        ax.plot(self.t_arr, p,'--', label='p', lw=2)
        ax.plot(self.t_arr, q,'--',label='q', lw=2)
        #ax.plot(self.t_arr, vd,'--', label='vd2', lw=2)
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)


class dVOC_Statcom_simple(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.1
        self.ib = 0.01
        self.sb = self.vb*self.ib
        self.zb = self.vb/self.ib
        super().__init__(13)

    def setup(self):
        #print('Setup symbolic HSS')
        bus_data, line_data = read_data('../data/voc_statcom_simple.xlsx')
        print('Newton-Raphson load flow started')
        newton_raphson(bus_data)
        printsys(bus_data)
        gridbus = bus_data[0]
        vocbus = bus_data[1]
        xf = line_data[0].xser
        rf = line_data[0].rser
        #rg = symbols('rg')
        # Declare symbolic variables and parameters
        # Circuit
        ug = gridbus.V*np.sqrt(2)*cos(self.w0*self.t)
        uin = symbols('uin')
        ia = symbols('ia')
        # dVOC
        vd, vq = symbols('vd vq')
        #va, vb = symbols('va vb')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        ksog = symbols('ksog')

        R = symbols('R')

        # Parametric study
        self.p = [R]  # Declare symbolic variables for parametric study
        #p1 = np.logspace(-4,-1,10)
        #p1 = np.linspace(0.005,0.1, 20)
        p1 = np.linspace(0.01,0.1,30)
        self.paramvals = [p1]
        #p1 = 2*np.logspace(2,3,10)
        #p1 = np.linspace(1,10,15)*Cdc0
        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        eta = 0.05
        mu = 0.1


        #  Fixed and derived parameters
        #  Circuit
        L = xf / self.w0
        #R = rf

        #  dVoc
        qref = vocbus.Q
        pref = vocbus.P_spec
        vref = vocbus.V

        # State space definition
        self.x = [ia, vd, vq, xa, xb]
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = I * np.sqrt(2) * sin(self.w0 * self.t - vocbus.angle)
        xa0 = ia0/np.sqrt(2)
        xb0 = I * cos(self.w0 * self.t - vocbus.angle)
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vd0 = vref*cos(vocbus.angle)
        vq0 = vref*sin(vocbus.angle)
        self.x0 = Matrix([ia0, vd0, vq0, xa0, xb0])

        # Reference frame
        th = self.w0*self.t#+vocbus.angle

        # DAEs
        # Algebraic
        va, vb = Rotate2d(vd,vq,th)
        #va = va
        mod = va*np.sqrt(2)
        #mod = mod/udc  # CM mode
        uc = mod

        unorm = va ** 2 + vb ** 2
        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fia = 1 / (L) * (uc - (R) * ia - uin-ug)

        # dVOC
        fva = -self.w0 * vb + eta * self.w0 * (1 / mu * phi * va + 1 / unorm * (va * qref - vb * pref) + xb)
        fvb = self.w0 * va + eta * self.w0 * (1 / mu * phi * vb + 1 / unorm * (pref * va + qref * vb) - xa)
        fvd, fvq = Rotate2d(fva+self.w0*vb, fvb-self.w0*va,-th)

        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/0.1*(rvir*ia-xvir)
        self.y = [-ia]                              # Outputs
        self.f = [fia, fvd, fvq, fxa, fxb]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()

        vd = self.xt[3,:]
        vq = self.xt[4,:]

        ia = self.xt[5,:]
        ib = self.xt[6,:]
        idq = np.exp(-1j*(self.w0*self.t_arr))*(ia+1j*ib)
        id, iq = idq.real,idq.imag
        p = vd*id+vq*iq
        q = -vd*iq+vq*id
        ax.plot(self.t_arr, p,'--', label='p', lw=2)
        ax.plot(self.t_arr, q,'--',label='q', lw=2)
        #ax.plot(self.t_arr, vd,'--', label='vd2', lw=2)
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)


class dVOC_Statcom_vi(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.3
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(13)

    def setup(self):
        #print('Setup symbolic HSS')

        # Declare symbolic variables and parameters
        # Circuit
        Rg = symbols('Rg')
        xg = symbols('xg')
        dVg = 0
        uin = symbols('uin')
        ia = symbols('ia')
        udc = symbols('udc')
        # dVOC
        vd, vq = symbols('vd vq')
        #va, vb = symbols('va vb')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        ksog = symbols('ksog')
        xa3, xb3 = symbols('xa3 xb3')
        xa2, xb2 = symbols('xa2 xb2')
        xa5, xb5 = symbols('xa5 xb5')
        #xvir = symbols('xvir')
        kvir = symbols('kvir')
        rvir = symbols('rvir')
        Ts = symbols('Ts')
        #Ts2 = symbols('Ts2')
        # DC control
        xdc = symbols('xdc')
        kpdc = symbols('kpdc')
        kidc = symbols('kidc')
        Cdc_r = symbols('Cdc_r')

        # Read system and solve power flow
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

        ug = (gridbus.V+dVg)*np.sqrt(2)*cos(self.w0*self.t)

        # Parametric study
        self.p = [kvir]  # Declare symbolic variables for parametric study
        #p1 = np.linspace(0.1,np.sqrt(2),30)

        #p1 = np.linspace(0.1,1,30)
        #p1 = np.linspace(0.01,0.1, 30)
        #p1 = 2*np.logspace(-4, -2, 20)
        p1 = np.flip(np.linspace(0,30,30))

        self.paramvals = [p1]
        #p1 = 2*np.logspace(2,3,10)
        #p1 = np.linspace(1,10,15)*Cdc0
        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        kvir = 0
        Ts = 1
        #Ts2 = 0.02
        eta = 0.02
        mu = 0.1
        kpdc = 1
        Cdc_r = 0.2e-3
        rvir = 0.0

        #  Fixed and derived parameters
        #  Circuit
        L = xf / self.w0
        R = rf
        Lg = xg / self.w0
        Rg = rg

        dv = np.abs(vocbus.P + 1j * vocbus.Q) * 1e3 / (2 * np.pi * 100 * Cdc_r * self.vdcb * 1e3) / (self.vdcb * 1e3)
        print(f'dv = {dv}')
        Cdc = Cdc_r * self.vdcb ** 2 / self.sb
        print(f'c_dc: {Cdc} pu')
        #  dVoc
        qref = vocbus.Q
        #qref = 0
        #pref = vocbus.P_spec
        #xvir = symbols('xvir')
        vref = vocbus.V
        #vref = 1
        # DC
        kidc = kpdc ** 2 * np.sqrt(2)
        udcref = 1.0

        # State space definition
        self.x = [ia, vd, vq, xa, xb, udc, xdc, xa3, xb3, xa2, xb2, xa5, xb5]
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = I * np.sqrt(2) * sin(self.w0 * self.t - vocbus.angle)
        xa0 = ia0/np.sqrt(2)
        xb0 = I * cos(self.w0 * self.t - vocbus.angle)
        udc0 = udcref
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vd0 = vref*cos(vocbus.angle)
        vq0 = vref*sin(vocbus.angle)
        self.x0 = Matrix([ia0, vd0, vq0, xa0, xb0, udc0, 0, 0,  0, 0, 0, 0, 0])

        # Reference frame
        th = self.w0*self.t#+vocbus.angle

        # DAEs
        # Algebraic
        va, vb = Rotate2d(vd,vq,th)
        va = va

        #va = va
        mod = (va-kvir*(xa3+xa5))*np.sqrt(2)-rvir*ia
        #mod = mod/udc  # CM mode
        mod = mod/(udcref+xa2)
        uc = mod*udc
        edc = -1 * (udcref ** 2 - udc ** 2)
        pref = (xdc + kpdc * edc)

        unorm = va ** 2 + vb ** 2
        phi = (1 - unorm / (vref) ** 2)/2
        #unorm = vref ** 2
        # Differential
        # Circuit
        fia = 1/(L+Lg)*(uc-ug-ia*(R+Rg)-uin)
        fudc = -1 / Cdc * (mod * ia)

        # DC
        fxdc = kidc * edc

        # dVOC
        fva = -self.w0 * vb + eta * self.w0 * (1 / mu * phi * va + 1 / unorm * (va * qref - vb * pref) + xb)
        fvb = self.w0 * va + eta * self.w0 * (1 / mu * phi * vb + 1 / unorm * (pref * va + qref * vb) - xa)
        fvd, fvq = Rotate2d(fva+self.w0*vb, fvb-self.w0*va,-th)

        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        # Virtual Impedance
        fxa3 = (1/(Ts*3*self.w0) * ((ia/np.sqrt(2)-xa) - xa3) - xb3) * 3*self.w0
        fxb3 = xa3 * 3*self.w0

        # Virtual Impedance
        fxa5 = (1 / (Ts * 5 * self.w0) * ((ia / np.sqrt(2) - xa) - xa5) - xb5) * 5 * self.w0
        fxb5 = xa5 * 5 * self.w0

        # modulation
        fxa2 = (1 / (0.1 * 2 * self.w0) * ((udc - 1) - xa2) - xb2) * 2 * self.w0
        fxb2 = xa2 * 2 * self.w0

        #fxr = 1/Ts*(ig-xr)

        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/Ts2*(rvir*ia-xvir)

        self.y = [-ia]                              # Outputs
        self.f = [fia, fvd, fvq, fxa, fxb, fudc, fxdc, fxa3, fxb3, fxa2, fxb2, fxa5, fxb5]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()

        vd = self.xt[1,:]
        vq = self.xt[2,:]
        vab = np.exp(1j*(self.w0*self.t_arr))*(vd+1j*vq)
        va = vab.real-25*(self.xt[-2,:]+self.xt[-4,:])
        vb = vab.imag
        mod = va
        uc = mod*self.xt[5,:]*np.sqrt(2)
        pinst = uc*self.xt[0,:]

        plt.plot(self.t_arr, uc, label='uc')
        plt.plot(self.t_arr, pinst, label='pinst')

        vcoeffs = self.dft_reduced(uc)
        ia = self.xt[5,:]
        ib = self.xt[6,:]
        idq = np.exp(-1j*(self.w0*self.t_arr))*(ia+1j*ib)
        p = va*ia+vb*ib  # From IPQ
        q = vb*ia-va*ib  # From IPQ
        self.cx2 = self.dft_reduced(self.xt)
        return vcoeffs


class dVOC_Tsukuba(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.3
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(13)

    def setup(self):
        #print('Setup symbolic HSS')
        bus_data, line_data = read_data('../data/dvoc_tsukuba.xlsx')
        print('Newton-Raphson load flow started')
        newton_raphson(bus_data)
        printsys(bus_data)
        gridbus = bus_data[0]
        vocbus = bus_data[1]
        xf = line_data[0].xser
        rf = line_data[0].rser
        xg = line_data[1].xser
        rg = line_data[1].rser
        #rg = symbols('rg')
        # Declare symbolic variables and parameters
        # Circuit
        ug = gridbus.V*np.sqrt(2)*cos(self.w0*self.t)
        uin = symbols('uin')
        ia = symbols('ia')
        ig, ucap = symbols('ig ucap')
        flcl = symbols('flcl')  # Resonance LCL filter
        # dVOC
        vd, vq = symbols('vd vq')
        #va, vb = symbols('va vb')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        ksog = symbols('ksog')
        # DC control
        Rg = symbols('Rg')
        Rvir = symbols('Rvir')

        # Parametric study
        self.p = [eta]  # Declare symbolic variables for parametric study
        #p1 = np.logspace(-4,-1,10)
        #p1 = np.linspace(0.005,0.1, 20)
        p1 = np.flip(np.linspace(0.005,0.05,30))
        self.paramvals = [p1]
        #p1 = 2*np.logspace(2,3,10)
        #p1 = np.linspace(1,10,15)*Cdc0
        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        #eta = 0.05
        mu = 0.1
        flcl = 2000

        #  Fixed and derived parameters
        #  Circuit
        L = xf / self.w0
        R = rf
        Lg = xg / self.w0
        Rg = rg
        C = 4.7e-6*self.zb
        Rvir = 0.0
        #  dVoc
        qref = vocbus.Q
        pref = vocbus.P_spec
        #xvir = symbols('xvir')
        vref = vocbus.V

        # State space definition
        self.x = [ia, ig, ucap, vd, vq, xa, xb]
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = I * np.sqrt(2) * sin(self.w0 * self.t - vocbus.angle)
        ig0 = ia0
        xa0 = ia0/np.sqrt(2)
        xb0 = I * cos(self.w0 * self.t - vocbus.angle)
        ucap0 = vref*np.sqrt(2)*cos(self.w0*self.t+vocbus.angle)
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vd0 = vref*cos(vocbus.angle)
        vq0 = vref*sin(vocbus.angle)
        self.x0 = Matrix([ia0, ig0, ucap0, vd0, vq0, xa0, xb0])

        # Reference frame
        th = self.w0*self.t #+2*vocbus.angle

        # DAEs
        # Algebraic
        va, vb = Rotate2d(vd,vq,th)

        uc = va*np.sqrt(2)-Rvir*ia
        #mod = mod/udc  # CM mode

        unorm = vd ** 2 + vq ** 2
        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fig = 1 / Lg * (ucap - Rg * ig - uin-ug)
        fia = 1/L*(uc-ucap-R*ia)
        fucap = 1/C*(ia-ig)

        # dVOC
        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)

        yd, yq = Rotate2d(xa,xb,-th)
        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd + (yq-iqref))
        fvq = eta * self.w0 * (1 / mu * phi * vq + (idref-yd))

        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/0.1*(rvir*ia-xvir)
        self.y = [-ig]                              # Outputs
        self.f = [fia, fig, fucap, fvd, fvq, fxa, fxb]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

    def plotpss(self):
        ax = super().plot_states()

        vd = self.xt[3,:]
        vq = self.xt[4,:]

        ia = self.xt[5,:]
        ib = self.xt[6,:]
        idq = np.exp(-1j*(self.w0*self.t_arr))*(ia+1j*ib)
        id, iq = idq.real,idq.imag
        p = vd*id+vq*iq
        q = -vd*iq+vq*id
        ax.plot(self.t_arr, p,'--', label='p', lw=2)
        ax.plot(self.t_arr, q,'--',label='q', lw=2)
        #ax.plot(self.t_arr, vd,'--', label='vd2', lw=2)
        plt.legend()
        self.cx2 = self.dft_reduced(self.xt)


def main():
    voc = dVOC_Statcom_Tsukuba_mod()

    #voc.find_pss()
    #voc.calc_eigs()
    pfs = voc.eigenloci()
    #voc.eigenloci_plot(pfs)
    #voc.save_eigenloci(pfs, ['ia', 'vd', 'vq','xdc'], [r'$i_a$', r'$v_d$',r'$v_q$', r'$x_{dc}$'])
    #voc.save_eigenloci(pfs, ['ia', 'vd','udc', 'vq'], [r'$i_a$', r'$v_d$', r'$u_{dc}$',r'$v_q$'])
    #voc.save_eigenloci(pfs, ['ia', 'va', 'vq','xb'], [r'$i_a$', r'$x_{\alpha}$',r'$v_q$', r'$x_{\beta}$'])
    voc.save_eigenloci(pfs, ['ia', 'udc', 'vq','xa2'], [r'$i_a$', r'$u_{dc}$',r'$v_q$', r'$x_{n,1}$'])
    """
    voc.find_pss()
    voc.calc_eigs()
    voc.toepBCD()
    #freqs = 4*np.logspace(1, 3, 400)
    freqs = np.linspace(-2,450,400, endpoint=False)
    #freqs = np.append(-np.flip(freqs), freqs)
    H = voc.hss_to_htf(freqs)
    #H.plot(Nplot=5,yscale='log')
    #plt.show()
    midcol = H.get_mid_col(tol=0.2)
    midcol.plot(xscale = 'log',yscale='log')
    #midcol.plot()
    plt.show()
    """
    return voc, pfs

if __name__ == '__main__':
    #pfs, pfs2, voc = main()

    #voc = dVOC_Statcom()
    voc, pfs = main()