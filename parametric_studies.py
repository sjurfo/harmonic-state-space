import numpy as np
from matplotlib import cm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd
import matplotlib.pyplot as plt


class ParametricSweep:
    def __init__(self, hss_model):
        """
        Parametric sweep for a HSS model

        :param hss_model: HSS model
        """
        self.model = hss_model

        # XY mesh for weakest dampings
        self.weak_damps = []
        self.XYmesh = []

        # Table of eigenvalues and participation factors for eigenloci
        self.modal_table = pd.DataFrame

    def sweep(self, pspace):
        if len(pspace) == 1:
            self.eigenloci(pspace)
        elif len(pspace) == 2:
            self.two_param_sweep(pspace)
        else:
            raise NotImplementedError('Sweep must be for 1 or 2 dimensions')

    def two_param_sweep(self, pspace):
        """
        Sweep two parameters and store the weakest damping

        :param pspace: List of two numpy 1-d arrays
        :return:
        """
        modal_props = self.model.pss.modal_props  # Shorthand
        num_p = len(pspace)
        if num_p != 2:
            raise AssertionError('Number of parameters is not 2!')
        damps = 2*np.ones((len(pspace[0]),len(pspace[1])))
        for idx_i, ival in enumerate(pspace[0]):
            if len(pspace) > 1:
                for idx_j, jval in enumerate(pspace[1]):
                    self.model.p_value = [ival, jval]
                    self.model.find_pss()
                    self.model.calc_modal_props()
                    damps[idx_i,idx_j] = modal_props.weak_damp
        modal_props.XYmesh = np.meshgrid(pspace[0], pspace[1])
        modal_props.damps = damps.T

    def eigenloci(self, pspace, p_idx=0):
        """
        Numerically compute the eigenloci for a sweep of one parameter

        :param pspace: Numpy 1-d array
        :param p_idx: Index of parameter to be swept
        """
        modal_props = self.model.pss.modal_props  # Shorthand

        xnames = [x.name for x in self.model.x]
        xharm = [f'{x.name}_n' for x in self.model.x]
        cols = xnames+xharm+['eig_re','eig_im',self.model.p[0].name]
        modal_table = pd.DataFrame(columns=cols)
        for indi, ival in enumerate(pspace):
            self.model.p_value[p_idx] = ival
            self.model.find_pss()
            self.model.calc_modal_props()
            for xi in range(len(modal_props.eig_filt)):
                pfs_i = modal_props.pf_filt[:,xi].tolist()+modal_props.npf[:,xi].tolist()+\
                        [modal_props.eig_filt[xi].real]+[modal_props.eig_filt[xi].imag]+[ival]
                modal_table.loc[len(modal_table)] = pfs_i

        modal_table[xharm] = modal_table[xharm].astype(int)
        self.modal_table = modal_table

    def save_eigenloci(self, params, params_txt):
        """
        Save the eigenloci in publication quality format.
        :param modal_table: table of modal properties
        :param params: List of 4 states for zoomed view
        :param params_txt: List of 4 strings for these states
        """

        df = self.modal_table
        # define variables
        x_name = 'eig_re'
        y_name = 'eig_im'
        param_name = df.keys()[-1]

        x = df[x_name]
        y = df[y_name]/(2*np.pi)
        paramvals = df[param_name].values
        states = list(df.keys())[0:self.model.Nx]

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

    def eigenloci_plot(self):
        """
        Interactive eigenloci plot

        Hover over an eigenvalue to see the participating states. Similarly, hover over the
        states to see which eigenvalues they participate (seen from opacity).

        :param modal_table:
        :return:
        """
        df = self.modal_table
        # define variables
        x_name = 'eig_re'
        y_name = 'eig_im'
        param_name = df.keys()[-1]

        x = df[x_name]
        y = df[y_name]/(2*np.pi)
        paramvals = df[param_name].values
        states = list(df.keys())[0:self.model.Nx]

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

    def plot_parametric_study2d(self, fig=None, **kwargs):
        modal_props = self.model.pss.modal_props
        if not fig:
            fig, ax = plt.subplots()
        else:
            pass
            #TODO get axis
        X = modal_props.XYmesh[0]
        Y = modal_props.XYmesh[1]
        if 'levels' in kwargs:
            levels = kwargs['levels']
            contourf_ = ax.contourf(X,Y,modal_props.damps, levels=levels,extend='max')
            cbar = fig.colorbar(contourf_, ticks=np.linspace(levels[0], levels[-1], 4))

        else:
            contourf_ = ax.contourf(X, Y, modal_props.damps)
            cbar = fig.colorbar(contourf_)

        ax.set_xlabel(self.model.p[0])
        ax.set_ylabel(self.model.p[1])

        # Print trace of highest damping
        #xidxs = np.arange(len(self.paramvals[0]))
        #yidxs = modal_props.damps.argmin(axis=0)
        #colvals = [self.p_space[1][i] for i in yidxs]
        #dampvals = [self.damps[i, j] for i, j in zip(xidxs, yidxs)]
        #ax.plot(self.paramvals[0], np.asarray(colvals),linewidth=3)

        return ax

    def plot_parametric_study3d(self, ax=None, offset=0):
        modal_props = self.model.pss.modal_props
        if not ax:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = modal_props.XYmesh[0]
        Y = modal_props.XYmesh[1]
        Z = modal_props.damps
        surf = ax.plot_surface(X, Y, Z, alpha=0.5, cmap=cm.viridis,
                               linewidth=0, antialiased=False)

        cset = ax.contour(X, Y, Z, zdir='z', offset=offset, cmap=cm.viridis)

        ax.set_xlabel(self.model.p[0])
        ax.set_ylabel(self.model.p[1])
        ax.set_zlabel(r'$Re[\lambda]$')

        plt.show()
        return ax
