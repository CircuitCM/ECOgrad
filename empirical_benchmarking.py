import sampling as sp
import grad_estimators as ge
import utils as ut
import benchmark_functions as bf
import numpy as np
import random as rd

def dev_test(d=100,s=300,biastype=1,seed=None):
    sp.set_seed(seed)
    d=d
    s=s
    A=bf.random_gauss_sqrmat(d)
    x= sp.rng.normal(0,1,d) #np.empty((d,),dtype=np.float64)
    S=np.empty((s,d),dtype=np.float64)
    sp.sample_sphere(S)

    g_true=bf.grad_sqrmat(x,A)
    #We utilize the exact (within f64 representation) directional gradient so that we can validate results meaningfully with noise std=0.
    #otherwise we'd need an accurate estimate of STD[f_smooth - f_round] a good estimate is probably the uniform distribution's std, or you can just use \epsilon_{mach} as the largest possible.
    c=sp.gradcross_sample(S,g_true) #replace with 1 side or 2 side function eval FD stencils
    #c=sp.side1_sample(x,S,lambda p:bf.val_sqrmat(p,A),2**(-52/2))
    #c=sp.side2_sample(x,S,lambda p:bf.val_sqrmat(p,A),2**(-52/3))
    #because we pregenerate our sample set for building our approximator, we don't need x.
    #c,S,g_true
    nfo1=ge.simple_averager(c,S,g_true,biastype)
    nfo2=ge.simple_leastchg(c,S,g_true,biastype)

    def end_str(nfo, i): return f'Final - {nfo[nfo.shape[0] - 1, i]:.3f}'

    ut.plot_gradest_info([
        [nfo1, f'Elliptic Mean', end_str],
        [nfo2, 'LMS Norm', end_str],],d)





