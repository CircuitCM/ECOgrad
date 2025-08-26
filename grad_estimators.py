import matplotlib.pyplot as plt
import numpy as np
import optional_numba as nbu
import math as mt
import utils as ut


@nbu.jtc
def simple_averager(c,S,truegrad, bias_type=0):
    """ A method that uses Monte Carlo averaging to converge to the true gradient by sampling stochastic gaussian approximations
    of the gradient. Unlike directional derivatives that utilize a norm=1 projection. Gaussian smoothing theory uses
    norm=sqrt(d) projections which average out in mean to the true gradient.

    :param c: 1D array of slopes or approximate slopes of truegrad along S.
    :param S: The directional sample vectors, (# samples, # parameters). Scaled to be at least proportional to ||S[i]||_2 = 1.
    :param truegrad: The true gradient used to benchmark the .
    :param bias_type: 0 - MSE/RMSE minimization adjustment, 1 - Matching norm adjustment, 2 - unadjusted: norm is biased large.
    :return: info_array
    """
    dms = S.shape[1]
    smps=S.shape[0]
    gn2=np.dot(truegrad,truegrad)
    # We translate directional derivative samples into stochastic approximators, easiest way:
    c *= dms

    g_est = np.zeros((dms,),dtype=np.float64)# np.zeros_like(S[0])  # grad update mem
    g_tmp = np.zeros((dms,), dtype=np.float64)
    info_array = np.empty((smps, 3), dtype=np.float64)  # info array

    for i in range(smps):
        # We translate directional derivative samples into stochastic approximators, so dim scale:
        #g_est += c[i] * S[i]*dms
        #It would be a bit faster in numba to write our own loops here, but that would be prohibitively slow in the python interpreter.
        g_est[:]*=(1 - (1/(i+1)))
        np.dot(c[i]*dms*(1/(i+1)),S[i],out=g_tmp)
        g_est[:]+=g_tmp
        g_tmp[:]=g_est
        bcorrect = ((i + 1) / (i + dms + 2)) if bias_type == 0 else mt.sqrt(((i + 1) / (i + dms + 2))) if bias_type == 1 else 1.
        g_tmp[:]*=bcorrect
        ut.calc_info(g_tmp,truegrad, gn2,info_array[i])
    return info_array

@nbu.jtc
def simple_leastchg(c,S,truegrad, bias_type=0):
    """

    :param c: 1D array of slopes or approximate slopes of truegrad along S.
    :param S: The directional sample vectors, (# samples, # parameters). Scaled to be at least proportional to ||S[i]||_2 = 1.
    :param truegrad: The true gradient used to benchmark the .
    :param bias_type: 0 - MSE/RMSE minimization (no adjustment), 1 - Matching norm adjustment, 2 - unadjusted: norm is biased large (to match monte carlo everager).
    :return: info_array
    """
    dms = S.shape[1]
    cm=1. - 1./dms
    smps=S.shape[0]
    gn2=np.dot(truegrad,truegrad)

    g_est = np.zeros((dms,),dtype=np.float64)# np.zeros_like(S[0])  # grad update mem
    g_tmp = np.zeros((dms,), dtype=np.float64)
    info_array = np.empty((smps, 3), dtype=np.float64)  # info array
    # norm matching adjustment $=\sqrt{1-(1-d^{-1})^k}$ Is the version solved for
    # raw monte carlo everaging adjustment $=1-(1-d^{-1})^k$
    # s=2
    for i in range(smps):
        er = c[i] - S[i].dot(g_est)
        g_tmp[:]=S[i]
        g_tmp[:]*=(er / 1.0)#for true directional samples S[i].dot(S[i]) = 1.0 always, however we might use samples such that S[i].dot(S[i]) is = 1 only on average, therefore we might want to see the degradation in the raw LMS update which is why it's removed from the denominator. You can use the block least change update with block size = 1 to compare.
        g_est[:] += g_tmp
        g_tmp[:] = g_est
        if bias_type != 0:
            bc=1. / (1. - cm ** (i + 1))
            if bias_type == 1: bc=mt.sqrt(bc)
            g_tmp[:]*= bc
        ut.calc_info(g_tmp,truegrad,gn2, info_array[i])
    return info_array


from scipy.stats import norm, t


def ext_marks_ratio(ugh, ngh, v, c, d, a=3., sig=0., b=1., m=1., q=1.):
    """
    Extended Marks ratio, includes a noise factor and onset buffer for initial eigen value asymmetry. We also use the arithmetic form, so > 0 implies anomaly instead of >1.

    :param ugh:  u^T·ĝ direction vector dot gradient estimator
    :param ngh: ||ĝ|| estimator norm
    :param v: Our true gradient directional scalar, if we knew g it would be u^T g, however v can be noisy if sig!=0.
    :param c: Previous estimator expectation \sin \theta, or "Expected Normalized Root Mean Square Error" of ĝ to g.
    :param a: 2-sided confidence interval that rejects anomalous v. Dependent on ||g||/\sqrt{d} because of how unit isotropic u is distributed as the dimension count increases. Use significance level to select `a` that "expects n false positives per m samples" 1-n/m.
    :param d: Dimensions or # parameters of our problem (u/g/x).shape[0]
    :param b: Confidence Interval of independent noise in v.
    :param sig: Constant noise std expected from v, derived for static whitenoise process.
    :return:
    """
    o = a
    c2 = c * c
    # For your knowledge,
    if m is not None:
        s = max(mt.log(c2) / mt.log1p(-1 / d),
                2.) - 1.  # in production we will only calculate this after a reset occurs
        # s = max(round(s,0),1)
        nsig = 2 * (1 - norm.cdf(a))

        o = t.ppf(1 - nsig / 2, df=s)  # /.81
        o = a + (o - a) / ((1 - 1 / m) ** 2)
        # o=(1+m/(s*a))**q
        if s < 10:
            print(o, a * (1 / (1 / m + (1 - c2))) ** q - a)
        # print('not hur')
    if ngh == 0.: ngh = v
    gamult = ((ngh * o) ** 2) * c2 / (d * (1 - c2))  # gamma/gradient expectation interval
    nf = 0. if sig is None else b * sig  # noise factor/interval

    return (ugh - v) ** 2 - gamult - nf * nf  # -1e-15#R(c)


from utils import marks_ratio, marks_shrinkage_reset_solution, shrink_gradestimate


def stationary_lms_eco(x, rng_func, sample_func, use_delta=False, bias_type=0, eps=2 ** (-51 / 2), truegrad=None,
                       alpha=3.,
                       b=1., sig=0., partial_reset=False, t_buffer=True,
                       ):
    """
    Error Correcting Optimization framework for broyden updated gradients. Stationary testing. Uses Marks Ratio

    :param x: array x[0] holds the global point, x[1:] holds samples and determines how many samples we will use.
    :param rng_func: to generate the directional perturbations, takes x.
    :param sample_func: samples the function values.
    :param use_delta: Do we delta normalize or norm normalize from perturbation. See `s1_sample`.
    :param bias_type: 0 - MSE/RMSE minimization adjustment (this is what it is already) , 1 - Matching norm adjustment, 2 - un-adjusted (now adjusted to match the initial norm of averaging).
    :param eps: Base relative machine epsilon to be used for scaling the point's proximal region.
    :param truegrad: None - it will generate the true grad from the assigned `grad` func.
    :return: info_array
    """
    x = rng_func(x)  # pre-init samples
    x, d = make_prox(x, eps)
    U, v = sample_func(x, d if use_delta else None)
    dms = U.shape[1]
    if alpha is None or alpha > (dms ** .5): alpha = dms ** .5

    g = np.zeros_like(x[0])  # grad update mem
    info_array = np.empty((U.shape[0], 3), dtype=np.float64)  # info array

    if truegrad == None:
        truegrad = grad(x[0])  # needed for measurements
    ct = 1
    # cbase=((1-(1/dms))**.5)
    l = 1.
    cmult = (1 - (1 / dms)) ** .5
    oc = c = 1  # *cmult
    # c2=(1-(1/dms))**ct
    rm = 0
    sig2 = sig ** 2
    for i in range(U.shape[0]):
        ngh2 = g.dot(g)
        if True:
            if c == 1:
                l = 1.
                cmult = (1 - (1 / dms)) ** .5
            else:
                u = ((c * c) * ngh2) / (dms * (1 - c * c))
                l = u / (u + sig2)
                # if i%100==0:print(i,l)
                cmult = (1 - (l / dms)) ** .5
        ngh = ngh2 ** .5
        ugh = U[i].dot(g)
        # c=c*cmult
        r = marks_ratio(ugh, ngh, v[i], c, dms, alpha, sig, b, t_buffer, False)

        if r > 0:
            if rm < r: rm = r
            ct += 1
            if partial_reset:
                oc = c
                # c=noisymarksratio_rootestimator(ugh,ngh,v[i],c,dms,alpha,sig,b,)
                c, s = marks_shrinkage_reset_solution(ugh, ngh, v[i], c, dms, alpha, sig, b, r, t_buffer)
                # print('Partial Reset Occurred: ',i,'oc',oc,'c',c,'r',r,'t_buffer',t_buffer)
                sd = mt.sqrt((1 - c ** 2) / (1 - oc ** 2))
                g *= sd
                ugh *= sd
            else:
                g[:] = 0
                ugh = 0.
                c = 1  # *cmult
        else:
            c = c * cmult
        er = v[i] - ugh
        # if i%250==1:
        #     print(i,'oc',oc,'c',c,'r',r)
        g = g + (er * l / 1.0) * U[i]  # for true directional samples U[i].dot(U[i]) = 1.0 always.
        bcorrect = 1 if bias_type == 0 else 1 / mt.sqrt(1 - c ** 2) if bias_type == 1 else 1 / (1 - c ** 2)
        gt = g * bcorrect
        calc_info(truegrad, gt, info_array[i])
    print('max r', rm, 'total resets', ct, ', False Positives Rate:', ct / U.shape[0])
    return info_array

