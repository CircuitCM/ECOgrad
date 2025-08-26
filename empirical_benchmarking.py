import sampling as sp
import grad_estimators as ge
import utils as ut
import benchmark_functions as bf
import numpy as np
import random as rd
rdn=np.random


def dev_test():
    ut.set_seed(42_069)
    d=500
    s=2000
    A=bf.random_gauss_sqrmat(d)
    x= rdn.normal(0,1,d) #np.empty((d,),dtype=np.float64)
    S=np.empty((s,d),dtype=np.float64)
    sp.sample_sphere(S)

    g_true=bf.grad_sqrmat(x,A)
    c=sp.gradcross_sample(S,g_true) #replace with 1 side or 2 side function eval FD stencils
    #c=sp.side1_sample(x,S,lambda p:bf.val_sqrmat(p,A),2**(-52/2))
    #c=sp.side2_sample(x,S,lambda p:bf.val_sqrmat(p,A),2**(-52/3))
    #because we pregenerate our sample set for building our approximator, we don't need x.
    c,S,g_true
    nfo1=ge.simple_averager(c,S,g_true)
    nfo2=ge.simple_leastchg(c,S,g_true)

    def end_str(nfo, i): return f'Final - {nfo[nfo.shape[0] - 1, i]:.3f}'

    ut.plot_info([
        [nfo1, f'Elliptic Mean', end_str],
        [nfo2, 'LMS Norm', end_str],])




