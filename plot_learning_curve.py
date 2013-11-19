'''
simple code to reproduce (hopefully pretty) plots.

> import plot_learning_curve
> import supervised_learner
> X,y = supervised_learner.learning_curve()
> plot_learning_curve.plot_curve(X,y)
'''

import pylab

def plot_curve(X, y, xlabel="number of labeled abstracts", 
                ylabel="F2 (sample size)", out_f="learning_curve.pdf"):
    pylab.clf()
    linewidth = 3.5
    pylab.plot(X,y,lw=linewidth,marker='o',ms=linewidth*3)
    ax1 = pylab.axes()
    pylab.setp(ax1.get_xticklabels(), color='black', size=14)
    pylab.setp(ax1.get_yticklabels(), color='black', size=14)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Helvetica']


    for x in ["right", "top"]:
        ax1.spines[x].set_visible(False)

    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.xlim(0,max(X)+5)
    pylab.axes().xaxis
    pylab.axes().yaxis

    pylab.savefig(out_f)