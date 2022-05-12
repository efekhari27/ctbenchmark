# -*- coding: utf-8 -*-
"""
@authors: E.Fekhari 

This file provides various static methods to generate design of experiments
"""
from random import sample
import numpy as np
import pandas as pd
import openturns as ot
from seaborn import pairplot
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import otkerneldesign as otkd

##Non openturn-based methods are imported from another .py files
#from fssf import fssfFr
import ctbenchmark as ctb
import otkerneldesign as otkd
#
##Using rpy2 in order to benefit of R's MaxPro and Support Point implementations
#from rpy2.robjects.packages import importr
#from rpy2.robjects import numpy2ri
#from rpy2 import robjects

class DesignOfExperiments:
    def __init__(self):
        ## So far this class is a simple collection of Doe Methods but Guillaume's work could be poured into it to add benchmark methods
        return None
    ###########################################
    ######### NON ITERATIVE METHODS ###########
    ###########################################
    @staticmethod
    def LHS_sample(size, distribution, my_seed=1):
        """
        Generate LHS samples
        
        Parameters
        ----------
        size: intoutput_sample
            Sample size
        distribution: ot.Distribution
            Input random distribution, its dimension determines the dimension of the input domain
        my_seed: int 
            Pseudo-random seed used to generate the design

        Returns
        -------
        return an LHS samples with the specified sample size and dimension.

        """
        ot.RandomGenerator.SetSeed(my_seed)
        LHS_experiment = ot.LHSExperiment(distribution, size, True, True)
        sample = LHS_experiment.generate()
        return np.array(sample)
                        
    @staticmethod
    def LHS_opt_sample(size, distribution, metric, my_seed=1):
        """
        Generate LHS samples optimized by simulated annealing, with a given optimization metric
        
        Parameters
        ----------
        size: int
            Sample size
        distribution: ot.Distribution
            Input random distribution, its dimension determines the dimension of the input domain
        metric: ot.SpaceFilling
            Metric to optimize by permuting points from a LHS. Can either be ot.SpaceFillingC2() or ot.SpaceFillingMinDist()
        my_seed: int 
            Pseudo-random seed used to generate the design

        Returns
        -------
        return a reps-sized array of LHS samples optimised by simulated annealing with the specified sample size, dimension and metric to minimise.
        """
        ot.RandomGenerator.SetSeed(my_seed)
        LHS_experiment = ot.LHSExperiment(distribution, size, True, True)
        SA_profile = ot.GeometricProfile(10., 0.95, 20000)
        LHS_optimization_algo = ot.SimulatedAnnealingLHS(LHS_experiment, metric, SA_profile)
        LHS_optimization_algo.generate()
        sample = LHS_optimization_algo.getResult().getOptimalDesign()
        return np.array(sample) 
    
    @staticmethod
    def sobol_sample(size, distribution):
        """
        Generate a Sobol sample

        Parameters
        ----------
        size: int
            Sample size
        distribution: ot.Distribution
            Input random distribution, its dimension determines the dimension of the input domain

        Returns
        -------
        return a Sobol sample with a given sample size representing the input distribution
        """

        dim = distribution.getDimension()
        seq = ot.SobolSequence(dim)
        sobol_experiment = ot.LowDiscrepancyExperiment(seq, distribution, size, False)
        sample = sobol_experiment.generate()
        return np.array(sample)
      
    @staticmethod
    def max_pro_sample(size, dimension, my_seed=1):
        """
        Generate a MaxPro sample

        Parameters
        ----------
        size: int
            Sample size
        dimension: int
            Dimension value
        my_seed: int 
            Pseudo-random seed used to generate the design

        Returns
        -------
        return a uniform MaxPro sample with a given sample size and dimension.
        """       
        importr('MaxPro')
        robjects.r('set.seed({})'.format(my_seed))
        # Define a Python function MaxProLHD from the R package
        MaxProLHD = robjects.r['MaxProLHD']
        XMP = MaxProLHD(size, dimension, itermax=100, total_iter=1e4)
        sample = np.array(XMP.rx2('Design'))
        return sample
    
    ###########################################
    ########### ITERATIVE METHODS #############
    ###########################################
    @staticmethod
    def FSSFfr_sample(size, distribution, initial_design=None, candidate_points=None, is_uniform=True):
        """
        Generate a FSSF-fr sample from Thomas Bittar's developpement.
        The transform does not take into account dependant input distributions.

        Parameters
        ----------
        size: int
            Sample size
        distribution: ot.ComposedDistribution()
            Input random distribution including its dimension
            Marginals are supposed to be independant
        initial_design: np.array()
            Numpy array with the shape size x dimension
        candiate_points: np.array()
            Numpy array with the shape candidate_point_size x dimension
        is_uniform: Boolean
            If True, all the marginals of the input distribution are uniform between 0 and 1.
            If False an inverse cdf marginal by marginal is applied to transform the sample.

        Returns
        -------
        sample: np.array
            FSSF-fr sample with a given sample size and dimension
        candidate_points: np.array
            Numpy array with the shape candidate_point_size x dimension

        """
        # Uniform sample
        dim = distribution.getDimension()
        if not is_uniform:
            marginals = distribution.getDistributionCollection()
            cdf_sample = ot.MarginalTransformationEvaluation(marginals, 0)
            candidate_points = np.array(cdf_sample(candidate_points))
            initial_design = np.array(cdf_sample(initial_design))

        sample, candidate_points = fssfFr(dim, size, initDes=initial_design, candidate_points=candidate_points)
        if not is_uniform:
            # Inverse transform marginal by marginal
            inverse_sample = ot.MarginalTransformationEvaluation(marginals, 1)
            sample, candidate_points = np.array(inverse_sample(sample)), np.array(inverse_sample(candidate_points))
        return sample, candidate_points

    @staticmethod
    def kernel_herding_sample(size, distribution, initial_design=None, candidate_points=None, scale_coeff=0.1, nu_coeff=2.5, my_seed=1):
        """
        Generate a KernelHerding sample

        Parameters
        ----------
        size: int
            Sample size
        distribution: ot.ComposedDistribution()
            Input random distribution including its dimension. If None then Uniform(0, 1).
        initial_design: np.array()
            Numpy array with the shape size x dimension
        candiate_points: np.array()
            Numpy array with the shape candidate_point_size x dimension
        scale_coeff: float 
            Scale coefficient of a ot.MaternModel
        nu_coef: float
            Nu coefficient of a ot.MaternModel
        my_seed: int 
            Pseudo-random seed used to generate the design

        Returns
        -------
        return a KernelHerding sample with a given sample size and dimension.
        """
        ot.RandomGenerator.SetSeed(my_seed)
        if distribution is not None:
            dim = distribution.getDimension()
        else :
            dim = candidate_points.shape[1]
        #scale = [scale_coeff] * dim
        #kernel = ot.MaternModel(scale, nu_coeff)
        ker_list = [ot.MaternModel([scale_coeff], [1.0], nu_coeff)] * dim
        kernel = ot.ProductCovarianceModel(ker_list)

        kh = otkd.KernelHerding(
            kernel=kernel,
            candidate_set_size=2 ** 12,
            distribution=distribution
            )
        kh_design, _ = kh.select_design(size)

        return np.array(kh_design)
    
    @staticmethod
    def support_points_sample(size, distribution, initial_design=None, candidate_points=None, my_seed=1):
        """
        Generate a SupportPoints sample

        Parameters
        ----------
        size: int
            Sample size
        distribution: ot.ComposedDistribution()
            Input random distribution including its dimension. If None then Uniform(0, 1).
        initial_design: np.array()
            Numpy array with the shape size x dimension
        candiate_points: np.array()
            Numpy array with the shape candidate_point_size x dimension
        my_seed: int 
            Pseudo-random seed used to generate the design

        Returns
        -------
        return a SupportPoints sample with a given sample size and dimension.

        """
        if distribution is not None:
            dim = distribution.getDimension()
        else :
            dim = candidate_points.shape[1]
        if candidate_points is None :
            ot.RandomGenerator.SetSeed(my_seed)
            dim = distribution.getDimension()
            candidate_size = 5000 * dim
            sobol_sequence = ot.LowDiscrepancyExperiment(ot.SobolSequence(), distribution, candidate_size, True)
            sobol_sequence.setRandomize(False)
            candidate_points = np.array(sobol_sequence.generate())
        importr('support')
        r = robjects.r
        r('graphics.off()')
        r('sink(NULL)')
        r('sink("/dev/null")') #Dispose every console output.
        r('set.seed({})'.format(int(my_seed)))
        ## NON ITERATIVE IMPLEMENTATION
        numpy2ri.activate()
        if initial_design is None:
            XSP = r['sp'](size, dim, dist_samp=candidate_points)
            sample = np.array(XSP.rx2('sp'))
        else:
            XSP = r['sp_seq'](initial_design, size, dist_samp=candidate_points)
            sample = np.array(XSP.rx2('seq'))
        numpy2ri.deactivate()
        ## SUBOPTIMAL ITERATIVE IMPLEMENTATION
        #numpy2ri.activate()
        ####### Case 1 ######
        #if initial_design is None:
        #    # First point using sp function
        #    XSP = r['sp'](1, dim, dist_samp=candidate_points)
        #    first_sp_point = np.array(XSP.rx2('sp'))
        #    sample = first_sp_point
        #    # Following points one by one using sp_seq function
        #    for i in range(size):
        #        XSP = r['sp_seq'](sample, 1, dist_samp=candidate_points, iter_max=200)
        #        sp_point = np.array(XSP.rx2('seq'))
        #        sample = np.vstack([sample, sp_point])
        ####### Case 2 ######
        #else:
        #    # Force support points to pick points one by one 
        #    sample = np.zeros(dim)
        #    for i in range(size):
        #        numpy2ri.activate()
        #        XSP = r['sp_seq'](initial_design, 1, dist_samp=candidate_points, iter_max=200)
        #        sp_point = np.array(XSP.rx2('seq'))
        #        sample = np.vstack([sample, sp_point])
        #        initial_design = np.vstack([initial_design, sp_point])
        #    sample = sample[1:]
        #numpy2ri.deactivate()
        return sample, candidate_points

    @staticmethod
    def greedy_support_points_sample(size, distribution, initial_design=None, candidate_points=None, my_seed=1):
        """
        Generate a SupportPoints sample

        Parameters
        ----------
        size: int
            Sample size
        distribution: ot.ComposedDistribution()
            Input random distribution including its dimension. If None then Uniform(0, 1).
        initial_design: np.array()
            Numpy array with the shape size x dimension
        candiate_points: np.array()
            Numpy array with the shape candidate_point_size x dimension
        my_seed: int 
            Pseudo-random seed used to generate the design

        Returns
        -------
        return a SupportPoints sample with a given sample size and dimension.

        """
        if candidate_points is None :
            ot.RandomGenerator.SetSeed(my_seed)
            dim = distribution.getDimension()
            candidate_size = 5000 * dim
            sobol_sequence = ot.LowDiscrepancyExperiment(ot.SobolSequence(), distribution, candidate_size, True)
            sobol_sequence.setRandomize(False)
            candidate_points = np.array(sobol_sequence.generate())
        
        candidate_size = candidate_points.shape[0]
        # Add initial design to the candidate points
        if initial_design is not None:
            candidate_points = np.vstack([candidate_points, initial_design])
            initial_size = initial_design.shape[0]
            design_indexes = np.arange(candidate_size, candidate_size + initial_size)
        else :
            initial_size = 0
            design_indexes = np.array([], dtype='int32')
        # Divide the candidate points in batches
        batch_nb = 8
        batch_size = candidate_size // batch_nb
        batches = []
        for batch_index in range(batch_nb):
            if batch_index==batch_nb - 1:
                batches.append(candidate_points[batch_size * batch_index : ])
            else:    
                batches.append(candidate_points[batch_size * batch_index : batch_size * (batch_index + 1)])
        # Build matrix of distances between all the couples of candidate points
        # Built block by block to avoid filling up memory
        distances = np.zeros([candidate_points.shape[0]] * 2, dtype='float16')
        for i, _ in enumerate(batches):
            for j in range(i+1):
                # raw i column j in the distances matrix
                batch_dist = distance_matrix(batches[i], batches[j])
                # Lower right corner block of the matrix has a different shape
                if (i==batch_nb - 1) and (j==batch_nb - 1):
                    distances[batch_size * i : , batch_size * j : ] = batch_dist
                # Lower left corner block of the matrix has a different shape
                elif (i==batch_nb - 1):
                    distances[batch_size * i : , 
                            batch_size * j : batch_size * (j + 1)] = batch_dist
                # Squared block
                else:
                    distances[batch_size * i : batch_size * (i + 1), 
                            batch_size * j : batch_size * (j + 1)] = batch_dist
        distances = np.tril(distances)
        distances += distances.T
        term1 = distances.mean(axis=0)
        for _ in range(size):
            if len(design_indexes)==0:
                criteria = term1
            else:
                distances_to_design = distances[:, design_indexes]
                term2 = distances_to_design.mean(axis=1)
                term2 *= len(design_indexes) / (len(design_indexes) + 1)
                criteria = term1 - term2
            next_index = np.argmin(criteria)
            design_indexes = np.append(design_indexes, next_index)
        sample = candidate_points[design_indexes[initial_size:]]
        return sample, candidate_points

###########################################
######### ADAPTIVE DESIGNS VISUALS ########
###########################################
def draw_initial_2D_design(x_sample, candidate_sample=None, title='Design of experiments'):
    fig = plt.figure(figsize=(8,8))
    plt.title(title, fontsize=20)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x_1$", fontsize=20)
    if candidate_sample is not None :
        plt.scatter(candidate_sample.T[0], candidate_sample.T[1], alpha=0.1, label='Candidate points ($N={}$)'.format(len(candidate_sample)), color='C7')
    plt.scatter(x_sample.T[0], x_sample.T[1], label='Initial design ($m={}$)'.format(len(x_sample)), marker='x', color='C3')
    #plt.legend(bbox_to_anchor=(0.5, -0.01), loc='upper center', ncol=4, columnspacing=1.2)
    plt.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center')
    return fig

def draw_2D_validation(x_learn, x_test, x_candidate=None, method_label='Kernel herding', marker_size=None, save_file=None):
    fig = draw_initial_2D_design(x_learn, x_candidate, method_label)
    test_label = 'Sequential design ' + ' ($n={}$)'.format(len(x_test))
    plt.scatter(x_test.T[0], x_test.T[1], color='C2', label=test_label, s=marker_size)
    for i in range(len(x_test)):
        fsize = np.max((22 - i, 5))
        plt.text(x_test[i][0], x_test[i][1], r"$\textbf{}$".format(i + 1), weight="bold", fontsize=fsize)
    legend = plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=1, fontsize=12)
    if save_file is not None:
        fig.savefig(save_file, bbox_extra_artists=(legend,), bbox_inches='tight')
    return fig

def draw_pairs_validation(x_learn, x_test, x_candidate, method_label='Kernel herding', save_file=None):
    dim = len(x_learn.T)
    learn_size = len(x_learn)
    test_size = len(x_test)

    columns_labels = ["X{}".format(i) for i in range(dim)]
    X_learn = pd.DataFrame(x_learn, columns=columns_labels)
    X_learn['type'] = ["learn"] * learn_size
    X_val = pd.DataFrame(x_test, columns=columns_labels)
    X_val['type'] = ["validation"] * test_size
    X_candidates = pd.DataFrame(x_candidate, columns=columns_labels)
    X_candidates['type'] = ["candidate"] * len(x_candidate)

    data = X_candidates.append(X_val).append(X_learn).reset_index(drop=True)
    label_order = ["learn", "validation", "candidate"]
    colors = {"learn":'C3', "validation":'C2', "candidate":'0.8'}
    marker_list = [".", "o", "X"]
    pp = pairplot(data, hue="type", hue_order=label_order, palette=colors, markers=marker_list, diag_kind="hist")
    if save_file is not None:
        pp.fig.savefig(save_file, bbox_inches='tight')
    return pp

