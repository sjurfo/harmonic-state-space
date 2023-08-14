from hss import HSS
from sympy import symbols, sin, cos, Matrix
import numpy as np
import matplotlib.pyplot as plt
from power_flow.read_data_with_trafos import read_data
from power_flow.nr_loadflow import newton_raphson, printsys
from parametric_studies import ParametricSweep

from sympy import Function


class Rotate2d(Function):
    @classmethod
    def eval(cls, x1,x2, th):
        return x1*cos(th)-x2*sin(th), x1*sin(th)+x2*cos(th)


class ComplexMult(Function):
    @classmethod
    def eval(cls, x11,x12,x21,x22):
        return x11*x21-x12*x22, x11*x22+x12*x21


class dVOC_1ph_dq_LCL(HSS):
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
        # print('Setup symbolic HSS')
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
        # rg = symbols('rg')
        # Declare symbolic variables and parameters
        # Circuit
        # Circuit
        uin = symbols('uin')
        ia, udc, ic, ucap = symbols('ia udc ic ucap')
        ####
        vd, vq = symbols('vd vq')
        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        xa, xb = symbols('xa xb')
        xa2, xb2 = symbols('xa2 xb2')  # DC compensation
        ksog = symbols('ksog')
        # DC control
        xdc = symbols('xdc')
        kpdc = symbols('kpdc')
        Cdc_r = symbols('Cdc_r')
        Tdc = symbols('Tdc')
        Rg = symbols('Rg')
        Rvir, Kz = symbols('Rvir Kz')
        xvir, Tvir = symbols('xvir Tvir')
        kp = symbols('kp')
        # vref = symbols('vref')

        thgrid = 0 # np.pi / 2
        ug = gridbus.V * np.sqrt(2) * sin(self.w0 * self.t + thgrid)

        # Parametric study
        self.p = [Tvir]  # Declare symbolic variables for parametric study
        self.p_value = [0.001]

        #  Overwrite parameter values if not used in parametric study
        ksog = np.sqrt(2)
        eta = 0.05
        mu = 0.01
        kpdc = 3
        Cdc_r = 0.2e-3
        Tdc = 0.01
        Rvir = 0.05 + 0.015
        kp = 0

        #  Fixed and derived parameters
        #  Circuit
        L = xf
        L1 = L*2/3
        L2 = L*1/3
        R = rf
        Lg = 0
        Rg = 0
        C = 5e-4
        Cdc = Cdc_r * self.vdcb ** 2 / self.sb * self.w0
        # Cdc = 500

        #  dVoc
        qref = vocbus.Q
        # qref = 0.0
        # pref = vocbus.P_spec
        # xvir = symbols('xvir')
        vref = vocbus.V

        # DC
        kidc = kpdc ** 2 * np.sqrt(2)
        udcref = 1.0
#####
        # Virtual z
        xz1, xz2 = symbols('xz1 xz2')
        wh = (150*2*np.pi)#+2*np.pi*20
        wb = 0.3
        Kz = 100

        #flcl = symbols('flcl')
        #C = (L+Lg)/(L1*(Lg+L2)*(2*np.pi*flcl)**2)

        ###

        # State space definition
        self.x = [ia, ic, ucap, vd, vq, xa, xb, udc, xdc, xa2, xb2, xz1, xz2, xvir]

        # Reference frame
        th = self.w0 * self.t  # +vocbus.angle

        # State space definition
        self.u = [uin]                               # Inputs
        self.u0 = [0]           # Initial conditions u
        # Initial conditions x (guess)
        I = vocbus.Q / vocbus.V
        ia0 = -I * np.sqrt(2) * sin(self.w0 * self.t - vocbus.angle+thgrid)
        ic0 = ia0
        vcap0 = ug
        xa0 = ia0/np.sqrt(2)
        xb0 = I * sin(self.w0 * self.t - vocbus.angle+thgrid)
        udc0 = udcref
        vref0 = 1
        #va0 = vref*cos(self.w0 * self.t+ vocbus.angle)
        #vb0 = vref*sin(self.w0 * self.t+vocbus.angle)
        vd0 = vref0*sin(vocbus.angle+thgrid)
        vq0 = -vref0*cos(vocbus.angle+thgrid)
        self.x0 = Matrix([ia0, ic0, vcap0, vd0, vq0, xa0, xb0, udc0, 0, 0, 0, 0, 0])

        # Reference frame
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
        mod = va*np.sqrt(2)-Rvir*xvir-Kz*xz1
        #mod = mod/udc  # CM mode
        mod = mod/(udcref+xa2)
        uc = mod*udc

        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential
        # Circuit
        fia = 1/(L2/self.w0)*(ucap-(R+Rg)*ia-ug)
        fic = 1/(L1/self.w0)*(uc-ucap)
        fucap = 1/C*(ic-ia)

        fudc = -1 / (Cdc/self.w0) * (mod * ic)

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

        # Virtual z
        fxz1 = -wh**2*xz2+wb*(ia-xz1)
        fxz2 = xz1
        fxvir = 1/Tvir*(ia-xvir)
        #fxd = (fxa*cos(th)+fxb*sin(th)+self.w0*xq)
        #fxq = (-fxa*sin(th)+fxb*cos(th)-self.w0*xd)

        #fxd, fxq = Rotate2d(fxa+self.w0*xb, fxb-self.w0*xa,-th)
        # DC voltage control
        #fxdc = kidc*edc

        # Virtual impedance
        #fxvir = 1/0.1*(rvir*ia-xvir)
        self.y = [-ia]                              # Outputs

        self.f = [fia, fic, fucap, fvd, fvq, fxa, fxb, fudc, fxdc, fxa2, fxb2, fxz1, fxz2, fxvir]

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


voc = dVOC_1ph_dq_LCL()
voc.find_pss()
voc.calc_modal_props()


fig = plt.figure(figsize=(3.54,3.54), dpi=600)
ax = fig.subplots()

eigs= voc.pss.modal_props.eigs
ax.scatter(np.real(eigs), np.imag(eigs),marker='x', color='k', alpha=0.5, linewidths=0.5, s = 9)
eig_filt = voc.pss.modal_props.eig_x0
ax.plot(np.real(eig_filt), np.imag(eig_filt),'s', fillstyle='none', color='k', markersize=3)

plt.xlabel('Real', fontsize=10)
plt.ylabel('Imag', fontsize=10)
ax.grid('on')
plt.tight_layout()
#ax.plot(t, np.real(u),'k', linewidth=2,alpha=0.5, label=r'$u$')
#ax.plot(t, np.real(u0),'k', linestyle='dotted', label=r'$u_0$')
#ax.plot(t, np.real(du),'k', label=r'$u_{lin}$', linestyle='dashed')

#ax.spines['left'].set_position(('data', 0.0))
#ax.spines['left'].set_color('none')
#ax.spines['bottom'].set_position(('data', 0.0))
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')

#ax.plot((1), (0), ls="", marker=">", ms=5, color="k",
#        transform=ax.get_yaxis_transform(), clip_on=False)
#ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
#ax.get_xaxis().set_visible(False)
#plt.xticks([0])
#ax.get_yaxis().set_visible(False)

axins = ax.inset_axes([0.3, 0.02, 0.3, 0.3])

axins.plot(np.real(eigs), np.imag(eigs),'x', color='k')
axins.plot(np.real(eig_filt), np.imag(eig_filt),'s', fillstyle='none', color='k')

# subregion of the original image
idx1, idx2 = 600, 700
x1, x2, y1, y2 = -130, 10, -1500,1500

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])
plt.legend()
ax.indicate_inset_zoom(axins, edgecolor="black")
plt.show()