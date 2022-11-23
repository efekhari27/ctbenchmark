#!/usr/bin/python
# coding:utf-8
# Copyright 2021 EDF
"""
Class to define the GSobol benchmark problem.

@author: efekhari27
"""

from ctbenchmark.CentralTendencyBenchmarkProblem import CentralTendencyBenchmarkProblem
import openturns as ot
import numpy as np


class CentralTendencyGSobolProblem10N(CentralTendencyBenchmarkProblem):
    def __init__(self, a=[2.] * 10):
        """
        Creates a central tendency problem from the g-Sobol function. 

        The model is:

        g(x) = prod_{i=0,..., d-1} g[i](x[i])

        where d is the dimension and

        g[i](x[i]) = (|4 * x[i] - 2.0| + a[i])/ (1 + a[i])

        x[i] = Normal(0.5, 0.15) truncated between 0. and 1.

        for i = 0, ..., d-1.

        The default dimension is equal to 3.

        Parameters
        ----------
        a : sequence of floats
            The coefficients of the linear sum, with length d + 1.
        
        Example
        -------
        problem  = CentralTendencyGSobolProblem()
        """

        dimension = len(a)

        # Define the function

        def GSobolModel(X):
            X = ot.Point(X)
            d = X.getDimension()
            Y = 1.0
            for i in range(d):
                Y *= (abs(4.0 * X[i] - 2.0) + a[i]) / (1.0 + a[i])
            return ot.Point([Y])

        g = ot.PythonFunction(dimension, 1, GSobolModel)

        # Define the distribution
        distributionList = [ot.TruncatedDistribution(ot.Normal(0.5, 0.1), 0., 1.) for i in range(dimension)]
        myDistribution = ot.ComposedDistribution(distributionList)

        name = "GSobol 10D (normal input)"

        inputRandomVector = ot.RandomVector(myDistribution)
        compositeRandomVector = ot.CompositeRandomVector(
            g, inputRandomVector
        )

        mean = 0.07622597230038555
        std = 0.025681478801014707

        super(CentralTendencyGSobolProblem10N, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
