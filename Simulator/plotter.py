import numpy as np
import constants_and_units as cu
import matplotlib.pyplot as plt


class subplot_figure():
    '''
    Class that creates a figure with subplots
    '''

    def __init__(self, n_rows, n_cols, figsize = (4,3)):
        '''
        Constructor that creates a figure with subplots
        Arguments
        ------------
        n_rows : int
            number of rows
        n_cols : int
            number of columns
        figsize : tuple
            size of the figure
        '''

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.figsize = figsize
        self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize = figsize,facecolor = "w")
        if n_rows + n_cols >2:
            self.adjust()

    def add_ax(self, row, col, ax):
        '''
        Function that adds an axis to the figure
        Arguments
        ------------
        row : int
            row of the axis
        col : int
            column of the axis
        ax : instance of the axis
            axis to be added
        '''
        self.axs[row,col] = ax

    def adjust(self, top = 0.9, bottom = 0.1, left = 0.1, right = 0.9,
                hspace = 0.2, wspace = 0.2):
        '''
        Function that adjusts the figure
        Arguments
        ------------
        top : float
            top of the figure
        bottom : float
            bottom of the figure
        left : float    
            left of the figure
        right : float   
            right of the figure
        hspace : float  
            height space between subplots
        wspace : float  
            width space between subplots
        '''

        self.fig.subplots_adjust(top = top, bottom = bottom, left = left, right = right,
                            hspace = hspace, wspace = wspace)


    def save(self, name):
        '''
        Function that saves the figure
        Arguments
        ------------
        name : str
            name of the figure
        '''

        self.fig.savefig(name+'.png', dpi = 300)


class plot():
    '''
    Class that generates default plot
    '''

    def __init__(self, ax, x, y, xlabel, ylabel, xlog = False, ylog = False, title = "",
                 xlim = None, ylim = None):
        '''
        Constructor that generates default plot 
        Arguments
        ------------
        ax : instance of the axis
            axis to be added
        x : array
            x axis data
        y : array
            y axis data.
        xlabel : str
            label of the x axis
        ylabel : str    
            label of the y axis
        xlog : bool
            if x axis is log
        ylog : bool
            if y axis is log
        title : str 
            title of the plot
        xlim : tuple
            limits of the x axis
        ylim : tuple
            limits of the y axis
        '''


        self.ax = ax
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlog = xlog
        self.ylog = ylog
        if xlim == None:
            self.xlim = (x[0],x[-1])
        else:
            self.xlim = xlim
        if ylim == None:
            self.ylim = (np.min(y),np.max(y))
        else:
            self.ylim = ylim
        self.title = title
        self.process_ax()

    def process_ax(self):
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        if self.xlog:
            self.ax.xscale('log')
        if self.ylog: 
            self.ax.yscale('log')



class plot2d(plot):
    '''
    Class that plots 2d map
    '''
    def __init__(self, ax, x, y, z, xlabel, ylabel, zlabel, title = "", xlog = False, ylog = False,
                    xlim=None, ylim=None, symmetric_scale = False, cmap = 'viridis', fixed_scale = False):
        '''
        Constructor that plots 2d map
        Arguments
        ------------
        ax : instance of the axis
            axis to be added
        x : array
            x axis data
        y : array
            y axis data. 
        z : array
            z axis data.
        xlabel : str    
            label of the x axis
        ylabel : str
            label of the y axis
        zlabel : str
            label of the z axis
        title : str 
            title of the plot
        xlog : bool
            if x axis is log
        ylog : bool
            if y axis is log
        xlim : tuple
            limits of the x axis
        ylim : tuple
            limits of the y axis
        symmetric_scale : bool  
            if the scale is symmetric
        cmap : str
            colormap
        fixed_scale : tuple 
            fixed scale of the plot
        '''

        super().__init__(ax, x, y, xlabel, ylabel, xlog, ylog, title, xlim, ylim)
        self.Z = z  
        self.cmap = cmap
        self.fixed_scale = fixed_scale
        self.symmetric_scale = symmetric_scale
        plot = self.plot()
        self.add_colorbar(plot,zlabel)
        

    def plot(self):
        '''
        Function that plots the 2d map
        '''

        if self.symmetric_scale:
            self.clim = np.max(np.abs(self.Z))
            p = self.ax.contourf(self.x, self.y, self.Z, cmap = self.cmap, levels = np.linspace(-self.clim, self.clim, 100))
       
        elif self.fixed_scale is not False:
            p = self.ax.contourf(self.x, self.y, self.Z, cmap = self.cmap, 
                                 levels = np.linspace(self.fixed_scale[0], self.fixed_scale[1], 100))
        else:
            p = self.ax.contourf(self.x, self.y, self.Z, cmap = self.cmap)
        return p

    def add_colorbar(self,plot, label = None):
        '''
        Function that adds colorbar to the plot
        Arguments
        ------------
        plot : instance of the plot
            plot to which colorbar is added
        label : str
            label of the colorbar
        '''
        self.fig = self.ax.get_figure()
        self.cbar = self.fig.colorbar(plot, ax = self.ax)
        self.cbar.ax.set_ylabel(label)


class plot1d(plot):
    '''
    Class that plots 1d map
    '''
    def __init__(self, ax, x, y, xlabel, ylabel,title = "", xlog = False, ylog = False,
                    xlim=None, ylim=None, labels = [""], styles = ["-"], colors = ["b"],alphas = [1],
                    zorders = [1],legend=False, grid = True):
    
        '''
        Constructor that plots 1d map
        Arguments
        ------------
        ax : instance of the axis
            axis to be added
        x : array
            x axis data. Typically size (1,Nx)
        y : array
            y axis data. Typically size (Ncurves,Nx), where Ncurves is the number of curves to be plotted
        xlabel : str
            label of the x axis
        ylabel : str
            label of the y axis
        title : str
            title of the plot
        xlog : bool
            if x axis is log
        ylog : bool 
            if y axis is log
        xlim : tuple
            limits of the x axis
        ylim : tuple    
            limits of the y axis
        labels : list   
            list of labels for each curve list(Ncurves), e.g. ["curve1","curve2","curve3"]
        styles : list
            list of styles for each curve list(Ncurves), e.g. ["-","--","o"]
        colors : list
            list of colors for each curve list(Ncurves), e.g. ["b","r","g"]
        alphas : list
            list of alphas for each curve list(Ncurves), e.g. [0.5,1,1]
        zorders : list
            list of zorders for each curve list(Ncurves), e.g. [1,2,3]
        legend : bool
            if legend is added
        grid : bool
            if grid is added
        '''

        
        super().__init__(ax, x, y, xlabel, ylabel, xlog, ylog, title, xlim, ylim)
        self.plot(colors, styles, labels, alphas, zorders)
        if legend:
            
            if len(labels)<4:
                ax.legend(ncol = len(labels))
            else:
                ax.legend(ncol = len(labels)//4)
        if grid:
            ax.grid()

    def plot(self, colors, styles, labels, alphas, zorders):
        for yn,y in enumerate(self.y):
            self.ax.plot(self.x, y, styles[yn],color=colors[yn], 
                         label = labels[yn], alpha = alphas[yn], zorder = zorders[yn])
  

class plotViolin(plot):
    """
    Class used to plot the violin plots
    """
    def __init__(self, ax, which_x, data, separation_factor, xlabel, ylabel, labels, xlog = False, ylog= False, title = "", xlim = None, ylim = None, legend = False):
        super().__init__(ax = ax, x = which_x, y = data, xlabel = xlabel, ylabel = ylabel, xlog = xlog, ylog = ylog, title = title, xlim = xlim, ylim = ylim)
        self.violins = len(self.y)
        self.separation = (which_x[2]-which_x[1])/self.violins/2*separation_factor
        self.labels = labels
        self.legend = legend
        self.plot_vliolin() 
        self.process_ax()


    def plot_vliolin(self):
        sets = []
        for vn in range(self.violins):
            sets.append(self.ax.violinplot(self.y[vn,:,self.x].T, 
                               positions=self.x+(self.separation*vn-self.violins/2*self.separation+self.separation), 
                               widths=2, 
                               showmeans=True, 
                               showextrema=False, 
                               showmedians=False, 
                               points=100, 
                               bw_method=0.5))
        if self.legend:
            self.ax.legend([sets[k]['bodies'][0] for k in range(self.violins)], [str(self.labels[k]) for k in range(self.violins)], ncol = 1)
    


#create subplots
def plot_pdf_insets(data_tracker, tracker, method, params):
    pdfs_shots = np.zeros((params["N_realisations"],params["N_shots"],101))
    oms = np.zeros((params["N_realisations"],params["N_shots"]))
    times = np.zeros(params["N_realisations"])
    tracker = np.array(tracker)
    for r in range(params["N_realisations"]):
        for i in range(params["N_shots"]):
            oms[r][i] = data_tracker[r][0][i][-2]
            times[r] = tracker[r][0]
            for j in range(101):
                pdfs_shots[r][i][j] = data_tracker[r][0][i][-1][j]
                
    plt.figure(figsize=(6,4))
    plt.plot(times, np.abs(oms[:,-1]), "o-")
    # draw an inset with no axes in which we plot pdfs_shots
    plt.xlabel("Time (ns)")
    plt.ylabel("Frequency (MHz)")
    if "Linear" in str(method):
        name = str(method)
    else:
        name = str(method)+r"  $c=$"+str(method.coeff)
    plt.title(name)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    xl = xlim[1]-xlim[0]
    yl = ylim[1]-ylim[0]
    for r in range(params["N_realisations"]):
        # put the inset at x = 0.1 and y = 10
        x = times[r]
        y = np.abs(oms[r][-1])
        ax2 = plt.axes([(x-xlim[0])/xl*0.9, 0.8*(y-ylim[0])/yl-0.01, 0.1, 0.1])
        ax2.contourf(pdfs_shots[r].T, levels = 100, cmap="binary")
        ax2.set_ylim(0,100)
        ax2.set_xticks([])
        ax2.plot(np.abs(oms[r]),color="red", lw=0.6)
        ax2.set_yticks([])
    plt.savefig("../Notes/Figures/method"+str(name)+".png", dpi=300)