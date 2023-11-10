from sympy import Function, cos, sin, atan2


class ComplexMult(Function):
    @classmethod
    def eval(cls, x11,x12,x21,x22):
        return x11*x21-x12*x22, x11*x22+x12*x21


class Rotate2d(Function):
    @classmethod
    def eval(cls, x1,x2, th):
        return ComplexMult(x1, x2, cos(th), sin(th))


class c2p(Function):
    @classmethod
    def eval(cls, x,y, th0):
        """
        Cartesian to polar
        :param x: x axis coordinate
        :param y: y axis coordinate
        :param th0: reference angle
        :return: polar magnitude and angle
        """
        xd,xq = Rotate2d(x,y, -th0)
        return (x**2+y**2)**0.5, atan2(xq,xd)


class p2c(Function):
    @classmethod
    def eval(cls, v, th, th0):
        """
        Polar to cartesian
        :param v: Polar amplitude
        :param th: angle
        :param th0: reference angle
        :return: Cartesian coordinates x,y
        """
        return v*cos(th+th0), v*sin(th+th0)
