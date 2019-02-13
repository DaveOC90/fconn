import h5py
import time
import os, sys
import glob
from functools import reduce

import numpy as np
import pandas as pd

import matplotlib
if (os.name == 'posix' and "DISPLAY" in os.environ) or (os.name == 'nt'):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib.animation as animation

elif os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('agg')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib.animation as animation


## Plotting

def plot_tsne(ipdct,hue_name=''):

    nplots=len(ipdct)
    nrows=int(np.ceil(nplots/4))

    titles=list(ipdct.keys())

    if nplots < 4:
        ncols=nplots
    else:
        ncols=4

    for i,title in enumerate(titles):
        plt.subplot(nrows,ncols,i+1)
        plt.title(title)
        if hue_name:
            ncolours=len(ipdct[title][hue_name].unique())
            cmap=sns.color_palette("coolwarm", n_colors=ncolours)
            sns.scatterplot(data=ipdct[title],x='x',y='y',hue=hue_name,palette=cmap,legend=False,alpha=0.7)
        else:
            npoints=ipdct[title].shape[0]
            cmap=sns.color_palette("coolwarm", n_colors=npoints)
            sns.scatterplot(data=ipdct[title],x='x',y='y',hue=np.arange(0,npoints),palette=cmap,legend=False,alpha=0.7)
            #plt.show()



def run_animation(data,anisave=False,savename=''):

    '''
    Refs:
    https://stackoverflow.com/questions/46236902/redrawing-seaborn-figures-for-animations
    https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/
    https://scipy-cookbook.readthedocs.io/items/Matplotlib_Animations.html
    '''

    def init():
        g=sns.heatmap(np.zeros((nx, ny)), vmax=1,vmin=-1,cmap=cmap)
        return g,

    def animate(frame):
        plt.clf()
        g=sns.heatmap(data[:,:,frame], vmax=1, vmin=-1,cmap=cmap)
        return g,

    nx,ny,frames=data.shape
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, repeat = False, blit=True, interval=100)


    if anisave:
        anim.save(savename, dpi=80, writer='imagemagick')
    else:
        plt.show()

def plot_3d(ipdata,labels=None,dsply=False,sve=False,savename=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(ipdata[0,:],ipdata[1,:],ipdata[2,:],s=10,c=labels)
    if dsply:
        plt.show()
    if sve and savename:
        plt.savefig(savename)
    elif sve and not savename:
        raise Exception("Must specifiy savename if sve == True")


def encircle(x,y, ax=None, **kw):
    '''
    from: https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot
    '''
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)

def scatter_and_encircle(x,y,cols=None):

    plt.scatter(x,y,c=cols)
    encircle(x,y,fc="none")

def encircle2(x,y, ax=None, **kw):
    '''
    from: https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot
    '''
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    mean = np.mean(p, axis=0)
    d = p-mean
    r = np.max(np.sqrt(d[:,0]**2+d[:,1]**2 ))
    circ = plt.Circle(mean, radius=1.05*r,**kw)
    ax.add_patch(circ)

