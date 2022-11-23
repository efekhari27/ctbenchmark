#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (C) EDF 2022

@author: Elias Fekhari
"""

import numpy as np
import openturns as ot
from scipy.special import erfc
from copy import deepcopy
import ctbenchmark as ctb

class AKDA:
    """
    New implementation of the “Adaptive Kriging Damage Assessment” method 
    developped in 2018 by Huchet to incrementally select new design points.

    Parameters
    ----------
    kernel : :class:`openturns.CovarianceModel`
        Covariance kernel used to define potentials.
        By default a product of Matern kernels with smoothness 5/2.
    distribution : :class:`openturns.Distribution`
        Distribution of the set of candidate points.
        If not specified, then *candidate_set* must be specified instead.
        Even if *candidate_set* is specified, can be useful if it allows the use of analytical formulas.
    candidate_set_size : positive int
        Size of the set of all candidate points.
        Unnecessary if *candidate_set* is specified. Otherwise, :math:`2^{12}` by default.

    initial_design : 2-d list of float
        Sample of points that must be included in the design. Empty by default.
    """

    def __init__(
        self,
        kernel=None,
        distribution=None,
        candidate_set=None,
        candidate_set_size=None,
        initial_design=None,
        initial_observations = None
    ):

        self._dimension = distribution.getDimension()

        # Kernel
        if kernel is None:
            supposed_size = 50
            scale = supposed_size ** (-1 / self._dimension)
            ker_list = [ot.MaternModel([scale], 2.5)] * self._dimension
            self._kernel = ot.ProductCovarianceModel(ker_list)
        else:
            self._set_kernel(kernel)

        # Candidate set
        if candidate_set is not None: 
            self._candidate_set = ot.Sample(candidate_set)
            candidate_set_size = len(candidate_set)
        else :
            if candidate_set_size is None:
                candidate_set_size = 2 ** 12

            sobol = ot.LowDiscrepancyExperiment(
                ot.SobolSequence(), distribution, candidate_set_size, True
            )
            sobol.setRandomize(False)
            self._candidate_set = sobol.generate()

        # Initial design
        self._design_indices = list(
            range(candidate_set_size, candidate_set_size + len(initial_design))
        )
        self._candidate_set.add(initial_design)
        self._candidate_pdf = np.array(distribution.computePDF(self._candidate_set)) 
        self._initial_observations = initial_observations


    def _set_kernel(self, kernel):
        if kernel.getInputDimension() == self._dimension:
            self._kernel = kernel
        else:
            raise ValueError(
                "Incorrect dimension {}, should be {}".format(
                    kernel.getInputDimension(), self._dimension
                )
            )

    @staticmethod
    def build_kriging(x_learning_sample, y_learning_sample):
        """
        Creates an openturns kriging result from an imput and output sample

        Parameters
        ----------
        x_learning_sample: np.array
            Input sample
        y_learning_sample: np.array
            Output sample

        Returns
        -------
        ot.KrigingResult
            Object containing all the Kriging results 

        """
        x_learning_sample = ot.Sample(x_learning_sample)
        y_learning_sample = ot.Sample(y_learning_sample)
        dim = x_learning_sample.getDimension()
        covarianceModel = ot.MaternModel([1.0] * dim, 2.5)
        basis = ot.ConstantBasisFactory(dim).build()
        
        # Define Kriging
        my_kriging = ot.KrigingAlgorithm(x_learning_sample, y_learning_sample, covarianceModel, basis)
        my_kriging.setOptimizeParameters(True)
        my_solver = my_kriging.getOptimizationAlgorithm()
        
        # Optimization bounds
        my_bounds = my_kriging.getOptimizationBounds()
        lbounds = my_bounds.getLowerBound()
        ubounds = my_bounds.getUpperBound()
        
        # Uniform distribution for multistart starting points DoE
        uni_dist = ot.DistributionCollection()
        for i in range(dim):
            uni_dist.add(ot.Uniform(lbounds[i], ubounds[i]))
        uni_dist = ot.ComposedDistribution(uni_dist)
        
        # Multistart starting points DoE
        K = 25 # design size
        LHS = ot.LHSExperiment(uni_dist, K)
        LHS.setAlwaysShuffle(False)
        SA_profile = ot.GeometricProfile(10., 0.95, 20000)
        LHS_optimization_algo = ot.SimulatedAnnealingLHS(LHS, ot.SpaceFillingC2(), SA_profile)
        
        # Generate starting points DoE
        LHS_optimization_algo.generate()
        LHS_design = LHS_optimization_algo.getResult()
        starting_points = LHS_design.getOptimalDesign()
        
        # Define optimization solver
        my_solver.setMaximumIterationNumber(10000)
        multiStartSolver = ot.MultiStart(my_solver, starting_points)
        my_kriging.setOptimizationAlgorithm(multiStartSolver)
        my_kriging.run()
        return my_kriging.getResult()

    def select_design(self, size, function):
        """
        Select a design with kernel herding.

        Parameters
        ----------
        size : positive int
            Number of points to be selected

        Returns
        -------
        design : :class:`openturns.Sample`
            Sample of all selected points
        """
        current_design_indices = deepcopy(self._design_indices)
        #current_design_indices = list(set(current_design_indices))

        current_x = deepcopy(self._candidate_set[current_design_indices])
        current_y = deepcopy(self._initial_observations)
        
        initial_design_size = len(current_design_indices)

        for k in range(size):
            # Build Kriging
            current_kriging_results = self.build_kriging(current_x, current_y)
            # # Get covariance model
            # conditionned_kernel = current_kriging_results.getCovarianceModel()
            # # Compute on candidate points the potential of this kernel
            # covmatrix = np.array(conditionned_kernel.discretize(self._candidate_set))
            # candidate_potentials = covmatrix.mean(axis=0).reshape(-1, 1)
            covmatrix = np.array(current_kriging_results.getConditionalCovariance(self._candidate_set))
            candidate_potentials = covmatrix.mean(axis=0).reshape(-1, 1)
            # Multiply by the pdf
            criterion = candidate_potentials * self._candidate_pdf
            # Get argmax
            iopt = criterion.argmax()
            current_design_indices.append(iopt)
            # Update the current_x and current_y variables
            x_opt = self._candidate_set[iopt]
            current_x.add(x_opt)
            # Compute new observation
            current_y.add(function(x_opt))
        design = current_x[initial_design_size:]
        design_indices = current_design_indices[initial_design_size:]
        observations = current_y[initial_design_size:]

        return design, design_indices, observations
