
import math as mt
from typing import Callable

import numpy as np

import scipy.special as sp
import math
import importlib.util
from t_model import FastT_V1
import optional_numba as nbu

def alpha_to_sigma(alpha: float,side2=True) -> float:
    """
    For a two‐sided significance level α (e.g. α=0.05 for 95% confidence),
    returns the corresponding ±kσ threshold.
    E.g. α=0.05 → k ≈ 1.96
    """
    # Invert the two‐sided confidence:
    #   erf(k/√2) = 1 – α
    # ⇒ k = √2 · erfinv(1 – α)
    return math.sqrt(2) * sp.erfinv(1 - (1 if side2 else 2)*alpha)

@nbu.rgic
def marks_reset_factor(cn,ugh, ngh, v, c, d, a=3., sig=0., b=1.,t_buffer=True,r=-2.):
    """
    
    :param cn: The N-RMSE that we optimize/estimate to accommodate the new directional derivative. 
    :param ugh:  u^T·ĝ direction vector dot gradient estimator
    :param ngh: ||ĝ|| estimator norm
    :param v: Our true gradient directional scalar, if we knew g it would be u^T g, however v can be noisy if sig!=0.
    :param c: Previous estimator expectation \sin \theta, or "Expected Normalized Root Mean Square Error" of ĝ to g.
    :param a: Typical range 2 to 4 z_sigma. 2-sided confidence interval that rejects anomalous v. Dependent on ||g||/\sqrt{d} because of how unit isotropic u is distributed as the dimension count increases, meaning a < \sqrt{d} always but this is only relevant at small dimension count. Use a z (normal) significance level to select `a` that "expects n false positives per m samples" 1-n/m.
    :param d: Dimensions or # parameters of our problem (u/g/x).shape[0]
    :param b: Confidence Interval of independent noise in v.
    :param sig: Constant noise std expected from v, derived for static whitenoise process.
    :return: 
    """
    di2=mt.sqrt(d)
    a=min(a,di2)
    c2=cn**2
    co2=c**2
    ratio =(1 - c2) / (1 - co2)
    scaling = mt.sqrt(ratio)
    dev_rate = (ugh * scaling - v)**2
    lam2_tol=(ngh*ngh)*c2/(d*(1-co2))
    sigma2=(b*sig)**2
    #lam2_tol=((norm_ghat*Tfast_V1(z_sig,s))**2)*c2/(d*(1-co2))
    #denominator = norm_ghat**2 * ratio * c2 + sigma2
    #we could cut down on the calculation cost a bit more by saying:

    cr=dev_rate -a*a*lam2_tol - sigma2
    #return cr
    if not t_buffer or cr<-1e-8:
        return cr
    else:
        lgdi=mt.log1p(-1/d)
        lcn=2*mt.log(cn)
        s = max(lcn / lgdi, 2.) - 1  # DoF or sample size for t-dist
        at= FastT_V1(a, s) ** 2.
        #at = min(at, di2)  # protect by hard lower bound
        cr=dev_rate -at*lam2_tol - sigma2
        return max(cr,r/3)

@nbu.rgic
def marks_ratio(ugh, ngh, v, c, d, a=3., sig=0., b=1., t_buffer=True,ratio_form=False):
    """
    Extended Marks ratio, includes the noise factor and student's t interval buffer for initial eigen value/predictor asymmetry. We also use the arithmetic form, so > 0 implies anomaly instead of >1.

    :param ugh:  u^T·ĝ direction vector dot gradient estimator
    :param ngh: ||ĝ|| estimator norm
    :param v: Our true gradient directional scalar, if we knew g it would be u^T g, however v can be noisy if sig!=0.
    :param c: Previous estimator expectation \sin \theta, or "Expected Normalized Root Mean Square Error" of ĝ to g.
    :param a: Typical range 2 to 4 z_sigma. 2-sided confidence interval that rejects anomalous v. Dependent on ||g||/\sqrt{d} because of how unit isotropic u is distributed as the dimension count increases, meaning a < \sqrt{d} always but this is only relevant at small dimension count. Use a z (normal) significance level to select `a` that "expects n false positives per m samples" 1-n/m.
    :param d: Dimensions or # parameters of our problem (u/g/x).shape[0]
    :param b: Confidence Interval of independent noise in v.
    :param sig: Constant noise std expected from v, derived for static whitenoise process.
    :return: 
    """
    #if ngh == 0.: ngh = v #It's the first sample vector
    a = min(a, mt.sqrt(d))  # protect gaussian upper bound, however the approximator may still have onset asymmetry so we t-dist using protected gauss and DoF.
    if c==1.:return -1. #also the first sample vector
    if t_buffer:
        s = max(2.*mt.log(c) / mt.log1p(-1 / d), 2.) - 1.  # in production we will only calculate this after a reset occurs, and increment otherwise.
        a= FastT_V1(a, s)
    if ngh == 0.: ngh = v
    c2 = c * c
    gamult = ((ngh * a) ** 2) * c2 / (d * (1 - c2))  # gamma/gradient expectation interval
    nf = 0. if sig is None else b * sig  # noise factor/interval
    
    if not ratio_form:
        return (ugh - v) ** 2 - gamult - nf * nf  # IF > 0 implies violation
    else:
        return abs(ugh - v)/mt.sqrt(gamult + nf * nf) # IF > 1 implies violation
    

@nbu.jtc
def marks_shrinkage_reset_solution(ugh, ngh, v, c, d, a=3., sig=0., b=1.,r=-2,t_buffer=True,max_iters=12,co_tol=1e-6):
    if ngh==0.: c,max(2 * mt.log(c) / mt.log1p(-1 / d), 2.)
    m_op=(marks_reset_factor,ugh,ngh,v,c,d,a,sig,b,t_buffer,-abs(r))
    mnc =1.#max(1. + min(-1 / d + 1e-15, 0.), .9999)
    if (ugh+v)<abs(ugh-v):br_rate=2/3 #Tends to be more well behaved because shrinkage will be more effective so boosted bracketing rate.
    else:br_rate=4/9 #otherwise there is an increased chance of multiple roots so take it slower
    #print(f' Solution Args: ', ", ".join(str(i) for i in (ugh, ngh, v, c, d, a, sig, b, t_buffer, -abs(r))))
    cn,lcn,hcn,reasn=signseeking_secant_v2(m_op,c,mnc,br_rate=br_rate,er_tol=co_tol,max_iters=max_iters,sign=-1)
    #print('positive curvature solution. c:',c,'lcn:',lcn,'hcn:',hcn,'reason:',reasn)

    # ##Note it turns out this third root scenario is so rare empirically, that I've never seen it actually happen, so I'm commenting out.
    # if t_buffer and (mnc - (lcn + hcn) / 2.) < .01 and reasn < 2:
    #     #We need to check that this didn't land on the 3rd t-limit left root
    #     #print('Launching on t_buffer c before:',cn,c,lcn,'marks ratio',marks_ratio(ugh,ngh,v,hcn,d,a,sig,b,t_buffer),'ugh',ugh,'v',v,'ngh',ngh,)
    #     cn2, lcn2, hcn2, reasn2 = signseeking_secant_v2(m_op, c, lcn, br_rate=2 / 3, er_tol=1e-6, max_iters=max_iters,sign=-1)
    #     if reasn2<2:#then it succeeded
    #         cn=cn2
    #         reasn=reasn2
    #         #print('Second Search succeeded.')
    #     else:pass
    #         #print('Second Search Failed.')
    #     #else: #it didnt succeed we rely on the t-limit result
    #     #print('Launching on t_buffer c after:', cn,'new mr',marks_ratio(ugh,ngh,v,cn,d,a,sig,b,t_buffer))
    lstc,ls=1-1/d,2.
    if reasn == 2: #reason 2 happens because no roots were found between 0 and 1.
        return lstc, ls
    s = max(2 * mt.log(cn) / mt.log1p(-1 / d), 2.,)
    return min(cn,lstc), max(s,ls) #it's barely meaningful to reset as a fraction less than 2 DoF, this also acts as a generic shrinkage smoothing bound for partial resets. In the t_buffered case this bound is basically never hit which confirms the cutoff. At very low dimension count <10, you can experiment with loosing these cutoffs.

def shrink_gradestimate(g,cn,c):
    g[:]*=mt.sqrt((1.-cn**2)/(1-c**2))
    return g

@nbu.rg
def signseeking_secant_v2(f_op, lo, hi,br_rate=.5, er_tol=1e-8, max_iters=20, sign=1):
    """A bracketed secant method that achieves (empirically) faster convergence by knowing the sign of the function to the left and right of the root.
    It also allows us to select if the slope of our root is positive or negative when there are multiple roots.

    The secant method uses the two most recent points, instead of the updated lo hi brackets, this typically gets the most out of the secant method,
    while still guaranteeing convergence with bisection bracketing. Assume we know only which lo or hi has a positive sign with regards to
    the general problem, if left side is positive we are seeking a negatively sloped root sign:=-1 vice versa for right side and positive slope root.
    Then until the first time sign(value)==-1, we only take a bracketing step; this strategy allows us to converge to a root that has a sign congruent
    slope in a multi root situation. Eg in a convex problem to a -(slope) root this will always be the left root.

    Other Notes: Convergence is only guaranteed when there is a single root with a congruent slope in the bracket.
    However, the likelihood of converging to a congruent root, is still very high due to the initial side rejection strategy explained above,
    by decreasing the bracketing increment to a range that guarantees sampling a basin br_rate <.5, you once more recover guaranteed
    convergence to the signed root.

    Variable calculations are all f64.

    :param f_op: Can be a function or a function operator (tuple) that includes its arguments. It receives a single scalar value for the point estimate. 
    :param br_rate: (0,1). The bracket increment, at .5 it's classic bisection, if you expect roots to be clustered on the right then >.5 might be suitable.
    Left clustered <.5. But a smaller br_rate should always have more definite convergence.
    :param sign: =1 we expect to have f(lo)<f(root)<f(hi). if -1 we expect f(lo)>f(root)>f(hi). If this expectation is unknown,
    it controls the bracketing bias eg if f is all positives and sign=1, then the bracket will reduce from right to left at (1 - br_rate) until
    reaching hi, if negatives and sign=1 then left to right at br_rate. Note: If both sides are wrong then convergence will not occur in the single root case.

    """
    fo, f = nbu.op_call_args(f_op, lo), nbu.op_call_args(f_op, hi)
    if sign == -1:
        op_bracket = f < 0.
        fo,f=-fo,-f
        fo, f = f, fo
        lamo, lam = hi, lo  #We know lo is positive, so we are more confident in giving it the step 2 interpolation point.
        lrt, hrt = 1 - br_rate, br_rate  #we want eagerness away from known side. so smaller=more conservative.
    else:
        op_bracket = fo < 0.
        lamo, lam = lo, hi
        lrt, hrt = br_rate, 1-br_rate
    #We init op_bracket by checking if the unknown point is negative, for this algo we assume that we know either lo or hi always has a positive sign for the general problem, my choice was due to the typical format of boundary solutions. If we seek a negative sloped root then we know our left side is positive, but the right side may have a basin or multiple roots (positive or negative areas), therefore we check if our right side has a negative bracketing location.
    

    #we flip our problems sign for negative roots so that lo bracket is always -, and hi always +.
    ict = int(max_iters)
    while ict > 0:
        fd = (f - fo)
        fo = f
        if abs(fd)<1e-15:
            lamo = lam
            lam = (lo*lrt + hi*hrt)
        else:
            lamn = lam - f * (lam - lamo) / fd
            lamo = lam
            lam = lamn
            lamb=(lo*lrt + hi*hrt)
            ll,lh=((lo,lamb) if sign==-1 else (lamb,hi)) if not op_bracket else (lo,hi)
            if not (ll<lam< lh): lam = lamb

        f = nbu.op_call_args(f_op,lam)
        op_bracket= op_bracket or f < 0.
        if sign == -1: f = -f #possible we don't need this and can replace with a single sign branch for the lo hi assignment.
        ict -= 1

        if f > 0.:
            hi = lam
        else:lo = lam

        if abs(lamo - lam) < er_tol: break

    return lam,lo,hi,2 if not op_bracket else 1 if ict==0 else 0

@nbu.rgic
def calc_info(g_est: np.ndarray, g: np.ndarray,gn2:float, infor: np.ndarray) -> None:
    """
    Compute metrics between true vector x and estimator y.
    Results are stored in outp as follows
      0: Cosine similarity between x and y
      1: Ratio of norms (||y||/||x||)
      2: Normalized MSE (MSE / ||x||^2)

    :param g_est: Gradient estimator vector, in a temporary/copied array where memory can be edited.
    :param g: True gradient.
    :param gn2: Use the precalculated gradient norm.
    :param infor: 3 element info array of floats.
    """


    nx2=np.dot(g_est,g_est)
    ny2=gn2
    # ny2=0
    # for v in g: ny2+= v * v
    infor[0] = np.dot(g_est, g) / (nx2 * ny2) #Cosine sim
    infor[1] = mt.sqrt(ny2 / nx2) #norm ratio
    #we will assume g_est can be edited, so we can calculate this efficiently, hopefully without array copies
    g_est[:]-=g
    g_est[:]*=g_est
    rs=np.dot(g_est,g_est)
    infor[2] = rs/nx2 #MSE, can do rmse as well.

import random as rd

def set_seed(sd=None):
    np.random.seed(sd)
    rd.seed(sd)
    _set_seed(sd)

@nbu.jtc
def _set_seed(sd=None):
    np.random.seed(sd)
    rd.seed(sd)


#Move this to a separate plotting.py later if more additions happen later

import matplotlib.pyplot as plt

def plot_gradest_info(info_list, dims, mnlook=None, mxlook=None,
              choose_plots=(0, 1, 2),
              pscale_ratios=(1., 1., 1.),
              plot_names=("Cosine Similarity", r"$\|\hat{g}_k\|/\|\nabla f(x)\|$",
                          r"Normalized RMSE $\|\hat{g}_k -\nabla f(x)\|/\|\nabla f(x)\|$"),
              vstack=True,
              figsize=(12, 12),
              minorxaxes=False,
              minoryaxes=True
              ):
    """
    info_list: list of (series, label) where series shape = (T, 3)
               columns: [cosine_sim, norm_ratio, norm_rmse]
    dims:      dimensionality used for expectation bounds
    mnlook: min sample range.
    mxlook:    max sample range
    """

    # ---------- prep ----------
    ss = info_list[0][0].shape[0]
    mnlook = 0 if mnlook is None else mnlook
    mxlook = ss if mxlook is None else min(mxlook, ss)

    plt.rc('grid', linestyle="-", color='white')
    if not vstack: figsize = figsize[::-1]
    fig = plt.figure(figsize=figsize)
    cp = choose_plots
    lps = len(cp)
    if vstack:
        gs = fig.add_gridspec(lps, 1, height_ratios=pscale_ratios, hspace=0.3, wspace=0.01)
    else:
        gs = fig.add_gridspec(1, lps, width_ratios=pscale_ratios, hspace=0.01, wspace=0.19)

    c = 0

    def _as(sharex=None, sharey=None):
        nonlocal c

        ax = fig.add_subplot(gs[c, 0] if vstack else gs[0, c], sharex=sharex, sharey=sharey)
        c += 1
        return ax

    ax0 = _as()

    x_vals = np.arange(mnlook, mxlook)

    # ---------- plot raw series ----------
    axes = {choose_plots[0]: ax0}
    for i in choose_plots[1:]:
        ax = _as(ax0)
        axes[i] = ax
    for i, ax in axes.items():
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.9, color='lightgray')
        if minorxaxes:
            ax.minorticks_on()
            ax.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.7, color='lightgray')
        if minoryaxes:
            ax.minorticks_on()
            ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.7, color='lightgray')
        for series, label, label_op in info_list:
            S = series[mnlook:mxlook]  # raw
            ax.plot(x_vals, S[:, i],
                    label=f"{label}, {label_op(S, i)}")
        ax.set_title(plot_names[i])
        ax.set_xlabel("Samples")

    if 0 in cp or 2 in cp:
        # ---------- LMS MSE bound ----------
        msebound = np.empty(ss, dtype=np.float64)
        for i in range(ss):
            msebound[i] = (1 - (1 / dims)) ** (i)

    if 0 in cp:
        ax0 = axes[0]
        ax0.set_ylabel('Feasible Range')
        ax0.set_ylim(0., 1.)
        cosbound = np.sqrt(1 - msebound)
        ax0.plot(x_vals, cosbound[mnlook:mxlook],
                 label=r"$\mathbb{E}(\text{LMS})$, " + f"Final: {cosbound[mxlook - 1]:.3f}")
    if 1 in cp:
        axes[1].margins(y=0.05)
        axes[1].set_ylabel('Norm Range')
        # axes[1].set_ylim(0.01,5)

    if 2 in cp:
        ax2 = axes[2]
        ax2.set_yscale('log')
        ax2.set_ylabel("Log10")
        ax2.margins(y=0.05)
        ax2.plot(x_vals, msebound[mnlook:mxlook],
                 label=r"$\mathbb{E}(\text{LMS})$, " + f"Final: {msebound[mxlook - 1]:.3f}")

    if minoryaxes:
        for v in axes.values():
            v.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5)

    # Legends
    if len(info_list) > 1:
        for v in axes.values():
            v.legend()

    return fig, axes


