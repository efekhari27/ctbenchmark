# -*- coding: utf-8 -*-
"""
@authors: E.Fekhari
File description TO DO 
"""

import numpy as np
import openturns as ot
from matplotlib import cm
import matplotlib.pyplot as plt

###########################################
############ FUNCTIONS VISUALS ############
###########################################
#my_colorbar = cm.Spectral.reversed() #Nice different colorbar
class DrawFunctions:
    def __init__(self):
        dim = 2
        self.grid_size = 100
        lowerbound = [0.] * dim
        upperbound = [1.] * dim
        mesher = ot.IntervalMesher([self.grid_size-1] * dim)
        interval = ot.Interval(lowerbound, upperbound)
        mesh = mesher.build(interval)
        self.nodes = mesh.getVertices()
        self.X0, self.X1 = np.array(self.nodes).T.reshape(2, self.grid_size, self.grid_size)

    def draw_2D_controur(self, title, function=None, distribution=None, colorbar=cm.coolwarm, nb_isocurves=8, contour_values=True):
        fig = plt.figure(figsize=(7, 6))
        if distribution is not None:
            Zpdf = np.array(distribution.computePDF(self.nodes)).reshape(self.grid_size, self.grid_size)
            contours = plt.contour(self.X0, self.X1, Zpdf, nb_isocurves, colors='black', alpha=0.6)
            if contour_values:
                plt.clabel(contours, inline=True, fontsize=8)
        if function is not None:
            Z = np.array(function(self.nodes)).reshape(self.grid_size, self.grid_size)
            plt.contourf(self.X0, self.X1, Z, 20, cmap=colorbar)
            plt.colorbar()
        plt.title(title, fontsize=20)
        plt.xlabel("$x_0$", fontsize=20)
        plt.ylabel("$x_1$", fontsize=20)
        #plt.close()
        return fig

    def draw_3D_controur(self, function, title, colorbar=cm.coolwarm):
        fig = plt.figure(figsize=(8, 8))
        Z = np.array(function(self.nodes)).reshape(self.grid_size, self.grid_size)
        ax = plt.axes(projection='3d')
        ax.contour3D(self.X0, self.X1, Z, 100, cmap=colorbar)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("$x_0$", fontsize=14)
        ax.set_ylabel("$x_1$", fontsize=14)
        ax.set_zlabel("$y$", fontsize=14)
        plt.close()
        return fig

    def draw_3D_surface(self, function, title, colorbar=cm.coolwarm):
        fig = plt.figure(figsize=(8, 8))
        Z = np.array(function(self.nodes)).reshape(self.grid_size, self.grid_size)
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.X0, self.X1, Z, rstride=1, cstride=1,
                        cmap=colorbar, edgecolor='none')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("$x_0$", fontsize=14)
        ax.set_ylabel("$x_1$", fontsize=14)
        ax.set_zlabel("$y$", fontsize=14)
        plt.close()
        return fig

    def draw_full_3D(self, function, title, zoffset, colorbar=cm.coolwarm):
        fig = plt.figure(figsize=(8, 8))
        Z = np.array(function(self.nodes)).reshape(self.grid_size, self.grid_size)
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.X0, self.X1, Z, rstride=1, cstride=1,
                    cmap=colorbar, edgecolor='none')
        ax.contour(self.X0, self.X1, Z, 20, zdir='z', offset=zoffset - 0.1, colors='grey', linewidths=.5)
        cset = ax.contourf(self.X0, self.X1, Z, 20, zdir='z', offset=zoffset - 0.1, cmap=colorbar)
        #ax.set_title(title, fontsize=20)
        ax.set_xlabel("$x_0$", fontsize=18)
        ax.set_ylabel("$x_1$", fontsize=18)
        ax.set_zlabel("$y$", fontsize=20)
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.25, 1.25)
        ax.set_zlim(zoffset, 3)
        plt.close()
        return fig