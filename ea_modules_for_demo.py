'''
E. Araya module of useful functions for demos.

'''


############ Modules and Libraries #################
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from numpy import loadtxt
from pylab import *
import numpy as np
from scipy import *
import os
from pylab import *
from scipy import *
from scipy import optimize
#from lmfit import minimize, Parameters
from scipy.misc import derivative
from scipy import signal

from IPython.display import HTML
from IPython.display import Image
from IPython.display import IFrame
from IPython.core.display import display






##### Begin: loadascii load spectrum from ASCII file ###############
def loadascii(inputfile,column_x,column_y,skiprows=0):
    spectrum = loadtxt(inputfile,dtype="float",skiprows=skiprows)
    x = spectrum[:,column_x-1]
    y = spectrum[:,column_y-1]
    x = np.array(x)
    y = np.array(y)
    return [x,y]
#################### End: loadascii #########################


#################### Begin: RMS #########################
def RMS(spec, xmin=None,xmax=None, ylabel='Flux Density (mJy)'):
    #Get RMS of spectrum
    vel_array = np.array(spec[0])
    Snu_array = np.array(spec[1])
    if xmin==None: xmin = vel_array[0]
    if xmax==None: xmax = vel_array[-1]
    spec_subdata = getsubdata(vel_array,Snu_array,xmin,xmax)

    x = spec_subdata[0]
    y = spec_subdata[1]
    rms = np.std(y)
    print('\t RMS = %1.3e (%s)' % (rms,ylabel))

    plt.title('Spectrum + 3$\sigma$ Error',fontsize=15)
    plt.xlabel(r'Velocity (km s$^{-1}$)',fontsize=15)
    plt.ylabel(ylabel,fontsize=15)

    # x and y axis
    labelx = getp(gca(), 'xticklabels')
    setp(labelx, color='black', fontweight='normal',fontsize=12)

    labely = getp(gca(), 'yticklabels')
    setp(labely, color='black', fontweight='normal',fontsize=12)
    #plot the line and the dots
    x_for_step = x-(x[10]-x[11])/2.0
    line1 = plt.step(x_for_step,y)
    #line1 = plt.plot(x,y,'r-',linewidth=1.0,)
    #line2 = plt.plot(x,y,'s',markersize = 5,color='g')
    line2 = scatter(x,y,marker='s',s=2,c='b')

    #draw 3sigma lines
    plt.plot((x[0], x[-1]), (3.0*rms, 3.0*rms), 'k-.')
    plt.plot((x[0], x[-1]), (-3.0*rms, -3.0*rms), 'k-.')

    #legend
    #plt.legend((line1,line2), ('Fit', 'Dots'),loc = 'upper right', numpoints=1)

    # x and y limits
    #delta = 10.0
    #xmin = min(x)-(max(x)-min(x))/(5*delta)
    #xmax = max(x)+(max(x)-min(x))/(5*delta)
    #ymin = min(y)-(max(y)-min(y))/delta
    #ymax = max(y)+(max(y)-min(y))/delta

    #set limits
    xmin,xmax,ymin,ymax=getlimits(x,y)
    xlim(xmin,xmax)
    ylim(ymin,ymax)

    #show figure and save in eps format
    show()
    #savefig(filename)
    return rms

#################### END: RMS #########################


#################### Begin: plspec #########################
# programs to plot spectra

# see http://matplotlib.org/users/event_handling.html
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


class GetPixelVal:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        
    def __call__(self, event):
        #print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        #self.line.set_data(self.xs, self.ys)
        #self.line.figure.canvas.draw()
        print('Clicked value :', event.xdata,event.ydata)


def getlimits(x,y,xmin=None,xmax=None,ymin=None,ymax=None):
    delta = 10.0
    x = np.array(x)
    y = np.array(y)
    if xmin != None:
        if ymin == None:
            index_xmin = (np.abs(x-xmin)).argmin()
            index_xmax = (np.abs(x-xmax)).argmin()
            if index_xmin>index_xmax:
                a=index_xmin
                index_xmin=index_xmax
                index_xmax=a
            ymin = np.min(y[index_xmin:index_xmax])-(np.max(y[index_xmin:index_xmax])-np.min(y[index_xmin:index_xmax]))/delta
            ymax = np.max(y[index_xmin:index_xmax])+(np.max(y[index_xmin:index_xmax])-np.min(y[index_xmin:index_xmax]))/delta
        return [xmin,xmax,ymin,ymax]
    elif ymin != None:
        index_ymin = (np.abs(y-ymin)).argmin()
        index_ymax = (np.abs(y-ymax)).argmin()
        if index_ymin>index_ymax:
            a=index_ymin
            index_ymin=index_ymax
            index_ymax=a
        xmin = np.min(x[index_ymin:index_ymax])-(np.max(x[index_ymin:index_ymax])-np.min(x[index_ymin:index_ymax]))/delta
        xmax = np.max(x[index_ymin:index_ymax])+(np.max(x[index_ymin:index_ymax])-np.min(x[index_ymin:index_ymax]))/delta
        return [xmin,xmax,ymin,ymax]
    else:
        return [xmin,xmax,ymin,ymax]    

def plspec(spec,xmin=None,xmax=None,ymin=None,ymax=None,ylabel='Flux Density (mJy)',xlabel='Velocity (km s$^{-1}$)', title='Spectrum', horizontal=None, add_patch = None):
    #plot spectrum function
    # No output
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    x = spec[0]
    y = spec[1]
    #plt.clf() 
    plt.title(title,fontsize=15)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel,fontsize=15)

    # x and y axis
    labelx = getp(gca(), 'xticklabels')
    setp(labelx, color='black', fontweight='normal',fontsize=12)

    labely = getp(gca(), 'yticklabels')
    setp(labely, color='black', fontweight='normal',fontsize=12)
    #plot the line and the dots
    x_for_step = x-(x[10]-x[11])/2.0
    line1 = plt.step(x_for_step,y)
    #line1 = plt.plot(x,y,'r-',linewidth=1.0,)
    #line2 = plt.plot(x,y,'s',markersize = 5,color='g')
    line2 = scatter(x,y,marker='s',s=2,c='b')

    if horizontal != None:
        line3 = plt.plot((x[0], x[-1]), (horizontal, horizontal), 'k-.')

    #legend
    #plt.legend((line1,line2), ('Fit', 'Dots'),loc = 'upper right', numpoints=1)

    # x and y limits
    #delta = 10.0
    #xmin = min(x)-(max(x)-min(x))/(5*delta)
    #xmax = max(x)+(max(x)-min(x))/(5*delta)
    #ymin = min(y)-(max(y)-min(y))/delta
    #ymax = max(y)+(max(y)-min(y))/delta

    #set limits
    xmin,xmax,ymin,ymax=getlimits(x,y,xmin,xmax,ymin,ymax)
    xlim(xmin,xmax)
    ylim(ymin,ymax)


    #get pixel info with mouse
    #useranswer = int(input('*** 0: get pix info, 1: Draw line *** '))
    #if useranswer == 1:
    #    line, = plot([0], [0])  # empty line
    #    linebuilder = LineBuilder(line)
    #elif useranswer == 0:
    line, = plot([-60], [0])  # empty line
    pixval = GetPixelVal(line)


    #If patch is given, then plot them
    # for example, in the code that calls plspec() one can have:
    #rect1 = matplotlib.patches.Rectangle((0,10), 4, 2, color='yellow', alpha = 0.4)
    #circle1 = matplotlib.patches.Circle((-20,-2), radius=3, color='#EB70AA')
    #add_patch = [rect1, rect2, circle1]
    #plspec(spec=spec_ave_ba,xmin=None,xmax=None,ymin=None,ymax=None,
    #   ylabel='Flux Density (Jy)',xlabel='Velocity (km s$^{-1}$)',
    #   title='No Cont', horizontal=None, add_patch = add_patch)

    if add_patch != None:
        for item in add_patch:
            ax.add_patch(item)

    
    #Show figure and save in eps format
    #draw()
    show()
    
    #savefig(filename)



#################### END: plspec #########################


#################### Begin: pl2spec #########################
def pl2spec(spec1,spec2,xmin=None,xmax=None,ymin=None,ymax=None,xlabel=r'Velocity (km s$^{-1}$)', ylabel='Flux Density (mJy)', title='Spectra', horizontal=None):

    x = spec1[0]
    y = spec1[1]
    x1 = spec2[0]
    y1 = spec2[1]

    plt.clf() 
    plt.title(title,fontsize=15)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel,fontsize=15)

    # x and y axis
    labelx = getp(gca(), 'xticklabels')
    setp(labelx, color='black', fontweight='normal',fontsize=12)

    labely = getp(gca(), 'yticklabels')
    setp(labely, color='black', fontweight='normal',fontsize=12)
    #plot the line and the dots
    x_for_step = x-(x[10]-x[11])/2.0
    line1 = plt.step(x_for_step,y)
    line2 = scatter(x,y,marker='s',s=2,c='b')

    x1_for_step = x1-(x1[10]-x1[11])/2.0
    line3 = plt.step(x1_for_step,y1,c='r')
    line4 = scatter(x1,y1,marker='s',s=2,c='r')

    if horizontal != None:
        line5 = plt.plot((x[0], x[-1]), (horizontal, horizontal), 'k-.')


    #set limits
    xmin,xmax,ymin,ymax=getlimits(x,y,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
    xlim(xmin,xmax)
    ylim(ymin,ymax)

    line, = plot([0], [0])  # empty line
    pixval = GetPixelVal(line)

    show()
    #savefig(filename)
#################### END: pl2spec #########################



#################### Begin: Baseline #########################

def getsubdata(x,y,xmin=None,xmax=None):
    #get a subset of data from x and y based on x range.
    if xmin != None:
        index_xmin = (np.abs(x-xmin)).argmin()
        index_xmax = (np.abs(x-xmax)).argmin()
        if index_xmin>index_xmax:
            a=index_xmin
            index_xmin=index_xmax
            index_xmax=a
        return [x[index_xmin:index_xmax], y[index_xmin:index_xmax]]
    else:
        return [x,y]


def Baseline(spec,x1,x2,x3,x4,poly_order=1,title='Spectrum',xmin=None,xmax=None,ymin=None,ymax=None,xlabel=r'Velocity (km s$^{-1}$)',ylabel='Flux Density (mJy)'):
    print('\n\t *******************************')
    print('\t Substract baseline')

    spec_subdata1 = getsubdata(x=spec[0],y=spec[1],xmin=x1,xmax=x2)
    spec_subdata2 = getsubdata(x=spec[0],y=spec[1],xmin=x3,xmax=x4)

    spec_subdata = np.concatenate((spec_subdata1,spec_subdata2),axis=1)

    #print spec_subdata

    fit_coefficients = np.polyfit(spec_subdata[0],spec_subdata[1],poly_order)

    #y_fit = np.sum((spec[0]**(power-len(fit_coefficients)+1.0))* coeff for power, coeff in enumerate(fit_coefficients))
    y_fit = np.sum((spec[0]**(len(fit_coefficients)-power-1.0)) * coeff for power, coeff in enumerate(fit_coefficients))

    #print fit_coefficients
    #print y_fit
    baseline=[spec[0],y_fit]
    pl2spec(spec,baseline,
            xmin=xmin,xmax=xmax,
            ymin=ymin,ymax=ymax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title)
    spec_ba = [spec[0],spec[1]-y_fit]
    return spec_ba


#################### End: Baseline #########################



#################### Begin: smooth #########################
def smooth_box(spec, box_pts,decimate=False):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(spec[1], box, mode='same')
    if decimate:
        spec_smo = [spec[0][int(box_pts/2)::box_pts],y_smooth[int(box_pts/2)::box_pts]]
    else:
        spec_smo = [spec[0],y_smooth]
    return spec_smo


#################### END: smooth #########################



####### OTHERS ##############

class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

