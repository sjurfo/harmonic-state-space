import pandas as pd
import numpy as np
from typing import List, Dict


class Bus:
    def __init__(self,Busdata):
        self.num = int(Busdata[0])
        self.type = Busdata[1]
        self.P_spec = Busdata[2]
        self.Q_spec = Busdata[3]
        self.V_spec = Busdata[4]
        self.Qlim = [Busdata[5], Busdata[6]]
        self.Plim = [Busdata[7], Busdata[8]]
        if not np.isnan(Busdata[9]) and Busdata[9] != 0:
            self.comp = -1/Busdata[9]
        else:
            self.comp = 0

        self.connectors = {}  # type: Dict[Bus,Connector]
        self.powerflow = {}  # type: Dict[Bus,float]
        # Load flow internal and result variables
        self.n = 0
        self.angle = 0
        if not np.isnan(self.V_spec):
            self.V = self.V_spec
        else:
            self.V = 1
        self.P = 0
        self.Q = 0
        self.j_ind1 = None
        self.j_ind2 = None

    def __repr__(self):
        return f'Bus("{self.num}","{self.type}")'

    def print(self):

        stri = "{:^6}|{:^6}|{:^8}|{:^8}|{:^8}|{:^8}|".format(self.num+1,self.type,self.P_spec,self.Q_spec,self.V_spec,
                                                            np.round(self.angle,4))
        stri += "{:^8}|{:^8}|{:^8}".format(np.round(self.V,4),np.round(self.P,4),np.round(self.Q,4))
        print(stri)

    def printpf(self):
        for bus, pf in self.powerflow.items():
            print('From {} to {}:  {}'.format(self.num+1, bus.num+1, np.round(pf, 4)))


class Connector:
    i: int
    j: int

    def __init__(self,i, j):
        #print(Linedata)
        self.i = i
        self.j = j

    def other(self,bus: Bus):
        if bus.num != self.j:
            return self.j
        else:
            return self.i

    def get_admittance(self, node):
        pass

class Line(Connector):
    y: complex
    rser: float
    xser: float
    b_c2: float
    ilim: float

    def __init__(self, i, j, R, X, xsh, ilim):
        super(Line, self).__init__(i,j)
        self.rser = R
        self.xser = X
        self.ilim = ilim
        if not np.isnan(xsh):
            self.b_c2 = -1/(2*xsh)
        else: self.b_c2 = 0

    def get_admittance(self, frombus):
        yl = 1/(self.rser+1j*self.xser)
        return yl, 1j*self.b_c2+yl

    def __repr__(self):
        return f'Line("{self.i}","{self.j}")'


class Transformer(Connector):
    rser: float
    xser: float
    aij: float

    def __init__(self, i,j, R, X, tap, alpha):
        super(Transformer, self).__init__(i,j)
        self.rser = R
        self.xser = X
        self.aij = tap*np.exp(1j*alpha*np.pi/180) # alpha in degrees

    def get_admittance(self, frombus):
        yl = 1/(self.rser+1j*self.xser)
        if frombus.num == self.i:
            return yl/self.aij.conjugate(), yl/np.abs(self.aij)**2
        else:
            return yl/self.aij, yl

    def __repr__(self):
        return f'Transformer("{self.i}","{self.j}")'


def read_data(filename):
    #  df = pd.read_excel('lesson1/task1/data.xlsx',skiprows=13,usecols=range(3,20))
    busdf = pd.read_excel(filename, sheet_name='Buses', skiprows=2)
    linedf = pd.read_excel(filename, sheet_name='Lines', skiprows=2)
    trafodf = pd.read_excel(filename, sheet_name='Trafos', skiprows=2)
    bd = {row[1]:Bus(row[1:11]) for row in busdf.itertuples() if not np.isnan(row[1])} # type: Dict[int,Bus]

    connectors = []

    for row in linedf.itertuples():
        row = row[1:]
        i = row[0]
        j = row[1]
        row = tuple(row)
        if bd[j] in bd[i].connectors:
            # Line already exists: create parallell equivalent
            line1 = Line(*row)
            line2 = bd[i].connectors[bd[j]]
            z1 = line1.rser+1j*line1.xser
            z2 = line2.rser+1j*line2.xser
            zpar = 1/(1/z1+1/z2)
            line2.rser = np.real(zpar)
            line2.xser = np.imag(zpar)
            line2.b_c2 += line1.b_c2
        else:
            line = Line(*row)
            bd[i].connectors[bd[j]] = line
            bd[j].connectors[bd[i]] = line
            #a.setdefault("somekey", []).append("bob")
            connectors.append(line)

    for row in trafodf.itertuples():
        row = row[1:]
        i = row[0]
        j = row[1]
        row = tuple(row)
        if bd[j] in bd[i].connectors:
            raise IndexError('Cannot assign transformer and line between the same two buses')
        trafo = Transformer(*row)
        bd[i].connectors[bd[j]] = trafo
        bd[j].connectors[bd[i]] = trafo
        # a.setdefault("somekey", []).append("bob")
        connectors.append(trafo)

    bd_sort = [bd[key] for key in sorted(bd.keys())]
    pqs = []
    pvs = []
    npv = 0
    npq = 0
    for i in bd_sort:

        if i.type == 'SB':
            sb = i
        elif i.type == 'PV':
            npv +=1
            pvs.append(i)
        else:
            npq += 1
            pqs.append(i)

    bd_sort2 = [sb] + pvs + pqs

    #for ind, i  in enumerate(bd_sort2):
    #    i.num = ind

    return bd_sort, connectors


def random_3bus():
    import random
    bus0 = Bus([0,'SB',0,0,1,-100,100,-100,100,0])
    bus1 = Bus([1, 'PV', random.randint(1, 100) / 50, 0,1, -100, 100, -100, 100,0])
    bus2 = Bus([2, 'PQ', -random.randint(1, 100) / 100, -random.randint(1, 100) / 100, 1,-100, 100, -100, 100,0])

    R1 = random.randint(1,10)/100
    X1 = random.randint(15,30)/100
    Line1 =  Line([0,1,R1,X1])
    R2 = random.randint(1,5)/100
    X2 = random.randint(10,20)/100
    Line2 = Line([0,2,R2,X2])
    bus0.connectors[bus1] = Line1
    bus1.connectors[bus0] = Line1
    bus0.connectors[bus2] = Line2
    bus2.connectors[bus0] = Line2

    return [bus0,bus1,bus2]