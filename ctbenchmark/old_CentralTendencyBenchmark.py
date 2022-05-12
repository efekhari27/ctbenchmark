#!/usr/bin/python
# coding:utf-8
# Copyright 2021 EDF
"""
Class to perform a central tendency benchmark for multiple problems, sample sizes and methods.

@author: efekhari27
"""

import openturns as ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CentralTendencyBenchmark:
    """Class to define a central tendency benchmark"""

    def __init__(self, problems, sample_sizes, methods):
        """
        Creates a benchmark for combinations of multiple problems, sample sizes and methods.

        Parameters
        ----------
        problems : list
            List of CentralTendencyProblem 

        sizes : list
            List of integers defining the sampling sizes

        methods : list
            List of methods to be benchmarked among the following strings 
            ['RegularGrid', 'MonteCarlo', 'Sobol', 'LHS']

        Example
        -------
        branin = otb.CentralTendencyBraninProblem()
        irregular = otb.CentralTendencyIrregularProblem()
        bench  = otb.CentralTendencyBenchmark([branin, irregular], [20, 50, 100], ['MonteCarlo', 'Sobol'])
        """
        self.problems = problems
        self.sample_sizes = sample_sizes
        self.methods = methods

        markers = "ovsXD*+"
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        self.markers = markers
        self.colors = colors
        return None

    @staticmethod
    def _runTaylor(problem):
        # Not appropriate for the current problems.
        """
        Runs a Taylor approximation of a central tendency problem.
        
        Parameters
        ----------
        problem: otbenchmark.CentralTendencyProblem
            Central tendency problem from otbenchmark's catalogue.
        
        Return: 
        -------
        apx_mean: float 
            Mean value for the central tendency problem approximated by this method.

        apx_std: float 
            Standard deviation value for the central tendency problem approximated by this method.

        sample_size: 
            Number of function calls effected by the method.

        Example:
        --------
        branin_problem = otb.CentralTendencyBraninProblem()
        apx_mean, apx_std, sample_size = CentralTendencyBenchmark.runTaylor(branin_problem)     

        """
        compositeRV = problem.compositeRandomVector
        function = compositeRV.getFunction()
        taylor = ot.TaylorExpansionMoments(compositeRV)
        apx_mean = taylor.getMeanFirstOrder()[0]
        apx_std = np.sqrt(taylor.getCovariance()[0, 0])
        sample_size = function.getEvaluationCallsNumber()
        return apx_mean, apx_std, sample_size

    @staticmethod
    def runRegularGrid(problem, sample_size):
        """
        Runs a regular grid sampling estimation of a central tendency problem.
        
        Parameters
        ----------
        problem: otbenchmark.CentralTendencyProblem
            Central tendency problem from otbenchmark's catalogue.
        
        sample_size: int
            Maximum number of function calls effected by the method.
        
        Return: 
        -------
        apx_mean: float 
            Mean value for the central tendency problem approximated by this method.

        apx_std: float 
            Standard deviation value for the central tendency problem approximated by this method.

        sample_size: int
            Number of function calls actually effected by the method. Note that depending 
            on the dimension, it can be very different from the sample size required. 
            e.g., in dimension 5, the consecutive admitted sizes are (1^5, 2^5, 3^5, 4^5, 5^5...).

        Example:
        --------
        branin_problem = otb.CentralTendencyBraninProblem()
        apx_mean, apx_std, sample_size = CentralTendencyBenchmark.runRegularGrid(branin_problem, 2**3)[:3] 
        """
        def _meshgrid(xi):
            """ Meshgrid function adapted from the numpy.meshgrid function"""
            ndim = len(xi)
            s0 = (1,) * ndim
            output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
                    for i, x in enumerate(xi)]
            output = np.broadcast_arrays(*output, subok=True)
            output = [x.flatten() for x in output]
            return np.asanyarray(output).T

        dim = problem.getDimension()
        function = problem.getFunction()
        distribution = problem.getDistribution()
        lowerbound = distribution.getRange().getLowerBound()[0]
        upperbound = distribution.getRange().getUpperBound()[0]
        # Generate a factorial design of experiment
        x_size = round(sample_size ** (1./float(dim)), 10)
        # If the hyper-cube is not complete, round to the next integer,
        # and truncate the sample later
        if x_size%1 != 0.:
            x_size += 1
        x = np.linspace(lowerbound, upperbound, int(x_size))
        input_sample = _meshgrid((x,) * dim)[:sample_size]
        output_sample = function(input_sample)
        f_x = distribution.computePDF(input_sample)
        weighted_output_sample = np.array(output_sample) * np.array(f_x)
        # Compute statistics
        apx_mean = weighted_output_sample.mean()
        apx_std = weighted_output_sample.std()
        return apx_mean, apx_std, sample_size

    @staticmethod
    def runMonteCarlo(problem, sample_size):
        """
        Runs a Monte Carlo sampling estimation of a central tendency problem.
        
        Parameters
        ----------
        problem: otbenchmark.CentralTendencyProblem
            Central tendency problem from otbenchmark's catalogue.
        
        sample_size: int
            Number of function calls effected by the method.
        
        Return: 
        -------
        apx_mean: float 
            Mean value for the central tendency problem approximated by this method.

        apx_std: float 
            Standard deviation value for the central tendency problem approximated by this method.

        sample_size: int
            Number of function calls effected by the method.

        Example:
        --------
        branin_problem = otb.CentralTendencyBraninProblem()
        apx_mean, apx_std, sample_size = CentralTendencyBenchmark.runMonteCarlo(branin_problem, 1000)[:3] 
        """
        function = problem.getFunction()
        distribution = problem.getDistribution()
        # Generate a MC sample of the input distribution and compute it
        input_sample = distribution.getSample(sample_size)
        output_sample = function(input_sample)
        # Compute statistics
        apx_mean = output_sample.computeMean()[0]
        apx_std = output_sample.computeStandardDeviation()[0]
        #TODO: get the 25% and 75% quantiles from the mean distribution  
        return apx_mean, apx_std, sample_size

    @staticmethod
    def runSobol(problem, sample_size, is_scrambled=False):
        """
        Runs a Sobol sequence estimation of a central tendency problem.
        
        Parameters
        ----------
        problem: otbenchmark.CentralTendencyProblem
            Central tendency problem from otbenchmark's catalogue.

        sample_size: int
            Number of function calls effected by the method.

        is_scrambled: boolean
            If True, the Sobol sequence will be randomized (also called scrambled). 
            Which means that the whole low discrepancy sequence is translated by a 
            random vector modulo 1.
        
        Return: 
        -------
        apx_mean: float 
            Mean value for the central tendency problem approximated by this method. 

        apx_std: float 
            Standard deviation value for the central tendency problem approximated by this method.

        sample_size: 
            Number of function calls effected by the method.

        Example:
        --------
        branin_problem = otb.CentralTendencyBraninProblem()
        apx_mean, apx_std, sample_size = CentralTendencyBenchmark.runSobol(branin_problem, 1000)[:3]  
        """
        dim = problem.getDimension()
        function = problem.getFunction()
        distribution = problem.getDistribution()
        # Define the low discrepancy sample
        sequence = ot.SobolSequence(dim)
        experiment = ot.LowDiscrepancyExperiment(sequence, distribution, sample_size)
        experiment.setRandomize(is_scrambled)
        # Sample on the low discrepancy sequence
        input_sample = experiment.generate()
        output_sample = function(input_sample)
        # Compute statistics
        apx_mean = output_sample.computeMean()[0]
        apx_std = output_sample.computeStandardDeviation()[0]
        return apx_mean, apx_std, sample_size

    @staticmethod
    def runLHS(problem, sample_size, is_optimized=False):
        """
        Runs a LHS sampling estimation of a central tendency problem.
        
        
        Parameters
        ----------
        problem: otbenchmark.CentralTendencyProblem
            Central tendency problem from otbenchmark's catalogue.

        sample_size: int
            Number of function calls effected by the method.

        is_optimized: boolean
            If True, a space-filling metric of the LHS is optimized by SimulatedAnneling method.
            Note that if the LHS is optimized, repetitions will be set to 1.
        
        Return: 
        -------
        apx_mean: float 
            Mean value for the central tendency problem approximated by this method. 

        apx_std: float 
            Standard deviation value for the central tendency problem approximated by this method.

        sample_size: 
            Number of function calls effected by the method for each repetition.
        
        Example:
        --------
        branin_problem = otb.CentralTendencyBraninProblem()
        apx_mean, apx_std, sample_size = 
                                CentralTendencyBenchmark.runLHS(branin_problem, 1000)[:3] 
        """
        function = problem.getFunction()
        distribution = problem.getDistribution()
        # Generate the LHS
        experiment = ot.LHSExperiment(distribution, sample_size)
        experiment.setAlwaysShuffle(True) # randomized
        # Optimized LHS
        if is_optimized:
            l2disc_metric = ot.SpaceFillingC2()
            temperature_profile = ot.GeometricProfile(10.0, 0.95, 1000)
            experiment = ot.SimulatedAnnealingLHS(experiment, l2disc_metric, temperature_profile)
        # Generate LHS
        input_sample = experiment.generate()
        output_sample = function(input_sample)
        # Compute statistics
        apx_mean = output_sample.computeMean()[0]
        apx_std = output_sample.computeStandardDeviation()[0]
        return apx_mean, apx_std, sample_size

    def run(self, is_scrambled=False, is_optimized=False):

        multi_index = pd.MultiIndex.from_product([self.methods, self.sample_sizes], names=['methods', 'sizes'])
        pb_names = [pb.getName() for pb in self.problems]
        df_mean = pd.DataFrame([], index=multi_index, columns=pb_names)
        df_std = pd.DataFrame([], index=multi_index, columns=pb_names)
        for problem, method, size in pd.MultiIndex.from_product([self.problems, self.methods, self.sample_sizes]):
            size = int(size)
            if method=="RegularGrid":
                mean, std = self.runRegularGrid(problem, size)[:2]
            elif method=="MonteCarlo":
                mean, std = self.runMonteCarlo(problem, size)[:2]
            elif method=="Sobol":
                    mean, std = self.runSobol(problem, size, is_scrambled)[:2]
            elif method=="LHS":
                mean, std = self.runLHS(problem, size, is_optimized)[:2]
            df_mean.loc[(method, size), problem.getName()] = mean
            df_std.loc[(method, size), problem.getName()] = std
        return df_mean, df_std

    def repeat(self, method, problem, sample_size, reps=51, is_scrambled=False, is_optimized=False, alpha=0.05):
        """
        TODO : UPDATE docstring
        Method to repeat multiple times a sampling method for central tendency estimation.

        Parameters
        ----------
        problem: otbenchmark.CentralTendencyProblem
            Central tendency problem from otbenchmark's catalogue.

        sample_size: int
            Number of function calls effected by the method.

        reps: int
            Number of times the design experiment is repeated to take into account the 
            variability of the method. If 

        is_scrambled: boolean
            For the "runSobol" method only.If True, the Sobol sequence will be randomized
            (also called scrambled). Which means that the whole low discrepancy sequence 
            is translated by a random vector modulo 1.

        Return: 
        -------
        apx_mean: float 
            Mean value for the central tendency problem approximated by this method. 
            Actually the median of the mean values of the central tendency problem if 
            the method is repeated reps times.

        apx_std: float 
            Standard deviation value for the central tendency problem approximated by this method.
            Actually the median of the std values of the central tendency problem if 
            the method is repeated reps times.

        apx_mean_reps: np.array
            List of mean values for all the repetitions sorted.
        
        apx_std_reps: np.array
            List of std values for all the repetitions sorted.
        """
        mean_reps, std_reps = np.zeros(reps), np.zeros(reps)
        if method=="RegularGrid":
            mean, std = self.runRegularGrid(problem, sample_size)[:2]
            mean_reps = [mean] * reps
            std_reps = [std] * reps
        elif method=="MonteCarlo":
            for rep in range(reps):
                mean, std = self.runMonteCarlo(problem, sample_size)[:2]
                mean_reps[rep] = mean
                std_reps[rep] = std
        elif method=="Sobol":
            if is_scrambled:
                for rep in range(reps):
                    mean, std = self.runSobol(problem, sample_size, is_scrambled)[:2]
                    mean_reps[rep] = mean
                    std_reps[rep] = std
            else :
                mean, std = self.runSobol(problem, sample_size)[:2]
                mean_reps = [mean] * reps
                std_reps = [std] * reps
        elif method=="LHS":
            if is_optimized:
                mean, std = self.runLHS(problem, sample_size, is_optimized)[:2]
                mean_reps = [mean] * reps
                std_reps = [std] * reps
            else :
                for rep in range(reps):
                    mean, std = self.runLHS(problem, sample_size)[:2]
                    mean_reps[rep] = mean
                    std_reps[rep] = std
        
        mean_stats = np.median(mean_reps), np.quantile(mean_reps, 1 - alpha), np.quantile(mean_reps, alpha)
        std_stats = np.median(mean_reps), np.quantile(std_reps, 1 - alpha), np.quantile(std_reps, alpha)
        return mean_stats, std_stats

    def repeated_run(self, reps=51):

        multi_index = pd.MultiIndex.from_product([self.methods, self.sample_sizes], names=['methods', 'sizes'])
        pb_names = [pb.getName() for pb in self.problems]
        df_med_mean, df_up_mean, df_low_mean = pd.DataFrame([], index=multi_index, columns=pb_names), \
                                                pd.DataFrame([], index=multi_index, columns=pb_names), \
                                                pd.DataFrame([], index=multi_index, columns=pb_names)
        df_med_std, df_up_std, df_low_std = pd.DataFrame([], index=multi_index, columns=pb_names), \
                                                pd.DataFrame([], index=multi_index, columns=pb_names), \
                                                pd.DataFrame([], index=multi_index, columns=pb_names)

        for problem, method, size in pd.MultiIndex.from_product([self.problems, self.methods, self.sample_sizes]):
            size = int(size)
            mean_stats, std_stats = self.repeat(method, problem, size, reps, is_scrambled=True)
            idx = (method, size)
            col = problem.getName()
            # Fill up mean df
            df_med_mean.loc[idx, col] = mean_stats[0]
            df_up_mean.loc[idx, col] = mean_stats[1]
            df_low_mean.loc[idx, col] = mean_stats[2]
            # Fill up std df
            df_med_std.loc[idx, col] = std_stats[0]
            df_up_std.loc[idx, col] = std_stats[1]
            df_low_std.loc[idx, col] = std_stats[2]
        return df_med_mean, df_up_mean, df_low_mean, df_med_std, df_up_std, df_low_std

    def draw_mean(self, problem, df_med_mean, df_up_mean, df_low_mean, reps=51):
        function_name = problem.getName()
        fig = plt.figure(figsize=(9, 6))
        for idx, method in enumerate(self.methods):
            df = df_med_mean.loc[(method, ), function_name]
            df_up = df_up_mean.loc[(method, ), function_name]
            df_low = df_low_mean.loc[(method, ), function_name]
            plt.plot(df.index, df.values, self.markers[idx] + '--', label=method, color=self.colors[idx])
            plt.fill_between(df.index, df_low.values.tolist(), df_up.values.tolist(), alpha=0.2, color=self.colors[idx])
        plt.axhline(y=problem.getMean(), color='black', label='Reference mean - MC ($10^8$)')
        plt.xlabel("Sample size", fontsize=12)
        plt.ylabel("Estimated mean", fontsize=12)
        plt.title("Mean estimation for {} ({} reps)".format(function_name, reps), fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.close()
        return fig

    

