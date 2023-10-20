""" Created: 13.10.2023  \\  Updated: 13.10.2023  \\   Author: Robert Sales """

#==============================================================================
'''
Please note: Below, N(mu, sigma) and GGD(beta, loc, scale) represent Normal and
Generalised Gaussian distributions respectively. When beta = 2 and scale = root
2 times sigma, then these are exactly the same. This has been incorporated into
the labels for each plot.

Since the beta shape parameter is almost 12 for weights these are approximately
uniformly distributed. Since this is about 1 for biases these are approximately
normally or Laplace distributed.

https://en.wikipedia.org/wiki/Generalized_normal_distribution
'''

#==============================================================================

import os, json, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import gennorm

#==============================================================================

plt.style.use("mystyle.mplstyle")  
params_plot = {'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}','axes.grid': True,"figure.figsize":(4,4),'axes.axisbelow': True}
matplotlib.rcParams.update(params_plot) 

SaveFiguresFlag = True

#==============================================================================
# Plot histograms of weight matrices (removing first and last for clarity)

def PlotWeightDistribution(num_bins):
        
    for index1,test_number in enumerate([1,2,3]):
    
        weight_files = sorted(glob.glob("/home/rms221/Documents/Compressive_Parameter_Quantisation/test_for_quantisation_{}/weights/*.kernel.npy".format(test_number)),key=SortLayerNames)
        
        fig = plt.figure(figsize=(6,6),constrained_layout=False)
        gridspec = fig.add_gridspec(nrows=1,ncols=1,height_ratios=[1],width_ratios=[1])
        ax = fig.add_subplot(gridspec[0,0])
        
        sum_of_caps = np.zeros(int(num_bins-1))
        all_weights = []
        
        histogram_bins = np.linspace(-0.5,+0.5,num_bins)
        scatter_bins = histogram_bins[:-1] + (np.ptp(histogram_bins[-2:])/2)
            
        for index2,weight_file in enumerate(weight_files[1:-1]):
                        
            weights = np.load(weight_file)
            
            if not index2:
                label = "Per-Matrix Weight Distribution"
            else:
                label = None
            ##
            
            caps,bins,patches = ax.hist(x=weights.ravel(),bins=histogram_bins,density=True,alpha=0.0)
    
            ax.scatter(scatter_bins,caps,color="blue",alpha=0.25,label=label,zorder=1)
            
            sum_of_caps = sum_of_caps + caps
            
            all_weights = all_weights + weights.ravel().tolist()
            
        average_caps = sum_of_caps / len(weight_files[1:-1])
        ax.hlines(average_caps,histogram_bins[:-1],histogram_bins[1:],color="red",linewidth=4.0,alpha=1.0,zorder=2,label=r"Matrix Averaged Distribution")
                   
        norm_parameters = norm.fit(all_weights)       
        x = np.linspace(-0.5,+0.5,501)
        y = norm.pdf(x,*norm_parameters)
        plt.plot(x,y,color="magenta",linestyle="dashed",linewidth=1.8,alpha=0.5,zorder=0,label=r"N$(\mu = {:.3f}, \sigma = {:.3f})$".format(*norm_parameters))
        
        gennorm_parameters = gennorm.fit(all_weights)       
        x = np.linspace(-0.5,+0.5,501)
        y = gennorm.pdf(x,*gennorm_parameters)
        mu = gennorm_parameters[1]
        sigma = gennorm_parameters[2] / np.sqrt(2)
        beta = gennorm_parameters[0] 
        plt.plot(x,y,color="lime",linestyle="dashdot",linewidth=1.8,alpha=0.5,zorder=0,label=r"GGD$(\mu = {:.3f}, \sigma = {:.3f}, \beta = {:.3f})$".format(mu,sigma,beta))
        
        plt.legend(ncols=1,loc="upper left")
        
        ax.set_xlim(-0.5,+0.5)
        ax.set_ylim(+0.0,+3.25)
        
        if SaveFiguresFlag: 
            plt.savefig("plots/test_for_quantisation_weights_{}.svg".format(test_number),bbox_inches="tight",dpi=600)  
            plt.savefig("plots/test_for_quantisation_weights_{}.png".format(test_number),bbox_inches="tight",dpi=600)  
        ## 
        
        plt.show()
    ##    
##

#==============================================================================
# Plot histograms of bias matrices (removing first and last for clarity)

def PlotBiasDistribution(num_bins):
        
    for index1,test_number in enumerate([1,2,3]):
    
        weight_files = sorted(glob.glob("/home/rms221/Documents/Compressive_Parameter_Quantisation/test_for_quantisation_{}/weights/*.bias.npy".format(test_number)),key=SortLayerNames)
        
        fig = plt.figure(figsize=(6,6),constrained_layout=False)
        gridspec = fig.add_gridspec(nrows=1,ncols=1,height_ratios=[1],width_ratios=[1])
        ax = fig.add_subplot(gridspec[0,0])
        
        sum_of_caps = np.zeros(int(num_bins-1))
        all_weights = []
        
        histogram_bins = np.linspace(-0.2,+0.2,num_bins)
        scatter_bins = histogram_bins[:-1] + (np.ptp(histogram_bins[-2:])/2)
            
        for index2,weight_file in enumerate(weight_files[1:-1]):
                        
            weights = np.load(weight_file)
            
            if not index2:
                label = "Per-Matrix Weight Distribution"
            else:
                label = None
            ##
            
            caps,bins,patches = ax.hist(x=weights.ravel(),bins=histogram_bins,density=True,alpha=0.0)
    
            ax.scatter(scatter_bins,caps,color="blue",alpha=0.25,label=label,zorder=1)
            
            sum_of_caps = sum_of_caps + caps
            
            all_weights = all_weights + weights.ravel().tolist()
            
        average_caps = sum_of_caps / len(weight_files[1:-1])
        ax.hlines(average_caps,histogram_bins[:-1],histogram_bins[1:],color="red",linewidth=4.0,alpha=1.0,zorder=2,label=r"Matrix Averaged Distribution")
                   
        norm_parameters = norm.fit(all_weights)       
        x = np.linspace(-0.2,+0.2,501)
        y = norm.pdf(x,*norm_parameters)
        plt.plot(x,y,color="magenta",linestyle="dashed",linewidth=1.8,alpha=0.5,zorder=0,label=r"N$(\mu = {:.3f}, \sigma = {:.3f})$".format(*norm_parameters))
        
        gennorm_parameters = gennorm.fit(all_weights)       
        x = np.linspace(-0.2,+0.2,501)
        y = gennorm.pdf(x,*gennorm_parameters)
        mu = gennorm_parameters[1]
        sigma = gennorm_parameters[2] / np.sqrt(2)
        beta = gennorm_parameters[0] 
        plt.plot(x,y,color="lime",linestyle="dashdot",linewidth=1.8,alpha=0.5,zorder=0,label=r"GGD$(\mu = {:.3f}, \sigma = {:.3f}, \beta = {:.3f})$".format(mu,sigma,beta))
        
        plt.legend(ncols=1,loc="upper left")
        
        ax.set_xlim(-0.20,+0.20)
        ax.set_ylim(+0.0,+23)
        
        if SaveFiguresFlag: 
            plt.savefig("plots/test_for_quantisation_biases_{}.svg".format(test_number),bbox_inches="tight",dpi=600)  
            plt.savefig("plots/test_for_quantisation_biases_{}.png".format(test_number),bbox_inches="tight",dpi=600)  
        ## 
        
        plt.show()
    ##
##

#==============================================================================

def SortLayerNames(parameter_filename):
    
    layer_name = parameter_filename.split("/")[-1].replace(".npy","")
    layer_index = int(layer_name.split("_")[0][1:])

    if "_a" in layer_name: 
        layer_index = layer_index
    
    if "_b" in layer_name: 
        layer_index = layer_index + 0.50
        
    if ".kernel" in layer_name: 
        layer_index = layer_index
        
    if ".bias" in layer_name: 
        layer_index = layer_index + 0.25   

    return layer_index

#==============================================================================
num_bins = 24
PlotWeightDistribution(num_bins=num_bins)

num_bins = 12
PlotBiasDistribution(num_bins=num_bins)
