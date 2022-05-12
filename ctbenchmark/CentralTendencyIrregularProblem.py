#!/usr/bin/python
# coding:utf-8
# Copyright 2021 EDF
"""
Class to define the Irregular benchmark problem found in the following reference:
Iooss, Bertrand & Boussouf, Loïc & Feuillard, Vincent & Marrel, Amandine. (2010). 
Numerical studies of the metamodel fitting and validation processes. 
International Journal of Advances in Systems and Measurements. 3.

@author: efekhari27
"""

from ctbenchmark.CentralTendencyBenchmarkProblem import CentralTendencyBenchmarkProblem
import openturns as ot
import numpy as np


class CentralTendencyIrregularProblem(CentralTendencyBenchmarkProblem):
    def __init__(self):
        """
        Creates a central tendency problem from the irregular function found in the following reference:
        Iooss, Bertrand & Boussouf, Loïc & Feuillard, Vincent & Marrel, Amandine. (2010). 
        Numerical studies of the metamodel fitting and validation processes. 
        International Journal of Advances in Systems and Measurements. 3.

        The function is the following:

        g(x1, x2) = exp(x1) / 5 - x2 / 5 + (x2 ** 6) / 3 + 4 * x2 ** 4 - 4 * x2 ** 2 + (7 * x1 ** 2) / 10 + x1 ** 4 + 3 / (4 * x1 ** 2 + 4 * x2 ** 2 + 1)
        We have x1 ~ Normal(0.5, 0.1) and x2 ~ WeibullMin(0.3, 2.0), both truncated between 0. and 1.

        Parameters
        ----------
        mu : sequence of floats
            The list of two items representing the means of the gaussian distributions.

        sigma : float
            The list of two items representing the standard deviations of
            the gaussian distributions.
        
        Example
        -------
        problem  = CentralTendencyIrregularProblem()
        """
        formula = "exp(x1) / 5 - x2 / 5 + (x2 ^ 6) / 3 + 4 * x2 ^ 4 - 4 * x2 ^ 2 + (7 * x1 ^ 2) / 10 + x1 ^ 4 + 3 / (4 * x1 ^ 2 + 4 * x2 ^ 2 + 1)"
        g = ot.SymbolicFunction(["x1", "x2"], [formula])

        #X1 = ot.TruncatedDistribution(ot.Normal(0.5, 0.1), 0., 1.)
        #X1.setDescription(["X1"])
        #X2 = ot.TruncatedDistribution(ot.WeibullMin(0.3, 2.), 0., 1.)
        #X2.setDescription(["X2"])
        #myDistribution = ot.ComposedDistribution([X1, X2])
        modes = [ot.Normal(0.3, 0.12), ot.Normal(0.7, 0.1)]
        weight = [0.4, 1.0]
        mixture = ot.Mixture(modes, weight)
        X1 = ot.TruncatedDistribution(mixture, 0., 1.)
        X1.setDescription(["X1"])
        X2 = ot.TruncatedDistribution(ot.Normal(0.6, 0.15), 0., 1.)
        X2.setDescription(["X2"])

        myDistribution = ot.ComposedDistribution([X1, X2], ot.ClaytonCopula(2.))
        inputRandomVector = ot.RandomVector(myDistribution)
        compositeRandomVector = ot.CompositeRandomVector(
            g, inputRandomVector
        )

        name = "CT_irregular_problem"
        # References computed by a very large MC sample (size 10**8)
        #mean = 1.5559449933705856
        #std = 0.44492631409352396
        mean = 0.802416025496437
        std = 0.39433913279693183

        super(CentralTendencyIrregularProblem, self).__init__(name, compositeRandomVector, 
                                                        mean, std)
        return None
