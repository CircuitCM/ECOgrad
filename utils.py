
import math as mt
from typing import Callable

import numpy as np

import mpmath as mpm
import math

def alpha_to_sigma(alpha: float,side2=True) -> float:
    """
    For a two‐sided significance level α (e.g. α=0.05 for 95% confidence),
    returns the corresponding ±kσ threshold.
    E.g. α=0.05 → k ≈ 1.96
    """
    # Invert the two‐sided confidence:
    #   erf(k/√2) = 1 – α
    # ⇒ k = √2 · erfinv(1 – α)
    return math.sqrt(2) * mpm.erfinv(1 - (1 if side2 else 2)*alpha)

_2ri= 1./mt.sqrt(2.0)

"""
-[FastT-V1]- A precise polynomial approximation of Student-t significance levels, interpolated from normal/gaussian significance levels and DoF.
At DoF>3 and Z in [.01, 6], the maximum error between confidence intervals/tail expectation : |p_true - p_est| is ~ 5.78 * 10^-4 or 0.058% .
The evaluation of this polynomial is many times less expensive than a direct numerical solution.

Deg3Poly represents the DoF where we reject the z-score at (t-z)/z (MaxAPE max absolute percentage) error.
From the cubic poly bound to DoF 3 the t-score is approximated by a 21 parameter rational, optimized for minimization of Max + Mean : abs(t_true-t_est)/t_true.
Tradeoff weights are chosen so Max and Mean are within .02% of their best when optimized as lone policies.
In some sense this is a more realistic and ideal error metric than abs(p_true - p_est) where p are the confidence intervals.
For DoF=1,2 we use exact solutions to the t-score as a function of math.erf, a much cheaper calculation than the general T.
However if Dof<3 and it's fractional, it will round down to 1 or 2; meaning `t_est` is only an upper bound in this range.
RootMAPE: sqrt(mean(((t_true-t_est)/t_true)^2))

╔════════════════════════════════════════════════════╗
║ Asset: Deg3Poly                                    ║
║ Datatype: float64  |  #params: 4                   ║
║ Info: Z-MaxAPE : 0.10%, Zmin: 0.01, Zmax: 6.0      ║
╚════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Asset: T-Model-4o21                                                                                                        ║
║ Datatype: float64  |  #params: 21                                                                                          ║
║ Info: estT-MaxAPE: 0.2564%, estT-RootMAPE: 0.1024% Sample Size: 1000000, Sample Density ~ 1000000*Deg3Poly(z)/(v*z_#lines) ║
║ This is a 21 parameter 2D rational polynomial expansion to the 4th order/degree.                                           ║
║ Note: Errors are from the training sample. Optimized using augmented LShade-CnEpSin. RootMAPE ~.05% with uniform sampling. ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

def FastT_V1(z_sig, dof):
    az =abs(z_sig)
    if az <.01 :return z_sig
    if dof < 3.:
        # if az > 8.2590072: return 0. #erf will round to zero after this range with f64, though assumption is user will select a reasonable z-score range.
        err =mt.erf(z_sig *_2ri)
        if dof < 2.:
            return mt.tan(( mt.pi / 2.0) * err)  # when dof is nearly 2. there will be significant error when z>5-ish, however its a fractional DoF and upper bounded so live with it.
        e = err * err
        den = 1.0 - e
        return mt.copysign(mt.sqrt(2. * e / den), err)

    # then poly
    p4, p3, p2, p1 = 0.004491152418700, 250.1508589461631, 0.243789262458383, 250.0952851341867
    if dof > p1:
        if dof > p1 + az * (p2 + az * (p3 + az * p4)): return z_sig

    return mt.copysign(tmv1_t1(az, dof), z_sig)

p4o_t1=(22.72573914371964, -13.570479991819441, -0.338998884352377, 2.486243732743884, 0.207331149646325, 0.013561147304741, -1.628487578121468, 0.017894429239572, 0.000042785084111, 0.000035858284938, 1.000549459621945, 20.903260486632245, -11.339740516942792, -0.786224682733048, 2.085934258154098, 0.789626406029434, -0.002451118780528, -1.792855924513853, -0.263394466232209, 0.002740111828052, -0.00001488653939, 1.)


def tmv1_t1(z: float, nu: float) -> float:
    #NOTE: This model is optimized for z .01 to 6, no guarantees outside of this range, make your own for a larger range but know that after 
    #z>8 you'll run into floating point precision issues.
    p = z*z #same thing but it's a tuple
    r = 1.0/nu
    # Numerator
    C4 = p4o_t1[0]
    C3 = p4o_t1[1] + p * p4o_t1[2]
    C2 = p4o_t1[3] + p * (p4o_t1[4] + p * p4o_t1[5])
    C1 = p4o_t1[6] + p * (p4o_t1[7] + p * (p4o_t1[8] + p * p4o_t1[9]))
    C0 = p4o_t1[10]
    num = (((C4 * r + C3) * r + C2) * r + C1) * r + C0
    D4 = p4o_t1[11]
    D3 = p4o_t1[12] + p * p4o_t1[13]
    D2 = p4o_t1[14] + p * (p4o_t1[15] + p * p4o_t1[16])
    D1 = p4o_t1[17] + p * (p4o_t1[18] + p * (p4o_t1[19] + p * p4o_t1[20]))
    D0 = p4o_t1[21]
    den = (((D4 * r + D3) * r + D2) * r + D1) * r + D0
    return z * (num / den)

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
        at = min(at, di2)  # protect by hard lower bound
        cr=dev_rate -at*lam2_tol - sigma2
        return max(cr,r/3)


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
    if c==1.:return -1. #also the first sample vector
    if t_buffer:
        s = max(2.*mt.log(c) / mt.log1p(-1 / d), 2.) - 1.  # in production we will only calculate this after a reset occurs only, and increment otherwise.
        a= FastT_V1(a, s)
    a = min(a, mt.sqrt(d)) #protect by hard lower bound
    if ngh == 0.: ngh = v
    c2 = c * c
    gamult = ((ngh * a) ** 2) * c2 / (d * (1 - c2))  # gamma/gradient expectation interval
    nf = 0. if sig is None else b * sig  # noise factor/interval
    
    if not ratio_form:
        return (ugh - v) ** 2 - gamult - nf * nf  # IF > 0 implies violation
    else:
        return abs(ugh - v)/mt.sqrt(gamult + nf * nf) # IF > 1 implies violation
    

def marks_shrinkage_reset_solution(ugh, ngh, v, c, d, a=3., sig=0., b=1.,r=-2,t_buffer=True,max_iters=12,co_tol=1e-6):
    if ngh==0.: c,max(2 * mt.log(c) / mt.log1p(-1 / d), 2.)
    m_op=(marks_reset_factor,ugh,ngh,v,c,d,a,sig,b,t_buffer,-abs(r))
    mnc =1.#max(1. + min(-1 / d + 1e-15, 0.), .9999)
    if (ugh+v)<abs(ugh-v): #Tends to be more well behaved because shrinkage will be more effective early on so boosted bracketing rate.
        #print(f'Negative Solution Args: ', ", ".join(str(i) for i in (ugh, ngh, v, c, d, a, sig, b, t_buffer, -abs(r))))
        cn,lcn,hcn,reasn=signseeking_secant_v2(m_op,c,mnc,br_rate=2/3,er_tol=co_tol,max_iters=max_iters,sign=-1)
        #print('negative curvature solution. c:',c,'lcn:',lcn,'hcn:',hcn,'reason:',reasn)
    else: #otherwise there could be two or even three roots if t_buffer is true. So take it slower. Maybe even 1/3
        
        #print(f'Positive Solution Args: ',", ".join(str(i) for i in (ugh,ngh,v,c,d,a,sig,b,t_buffer,-abs(r))))
        cn, lcn, hcn, reasn=signseeking_secant_v2(m_op, c, mnc, br_rate=4/9, er_tol=co_tol, max_iters=max_iters,sign=-1)
        #print('positive curvature solution. c:',c,'lcn:',lcn,'hcn:',hcn,'reason:',reasn)
    
        # ##Note it turns out this third root scenario is so rare empirically I've never seen it actually happen, so I'm commenting out.
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


def signseeking_secant_v2(f_op, lo, hi,br_rate=.5, er_tol=1e-8, max_iters=20, sign=1):
    """A bracketed secant method that achieves better performance by knowing the sign of the function to the left and right of the root. It also allows us to choose the slope of our root in a multi root scenario.

     The secant points use the two most recent points, instead of the updated lo hi brackets, this typically gets the most out of the secant method, while still guaranteeing convergence with bisection bracketing. Assume we know only which lo or hi has a positive sign with regards to the general problem, if left side is positive we are seeking a negatively sloped root sign:=-1 vice versa for right side and positive slope root. Then until the first time sign(value)==-1, we only take a bracketing step; this strategy allows us to converge to a root that has a sign congruent slope in a multi root situation. Eg in a convex problem to a -(slope) root this will always be the left root.

    Other Notes: Convergence is always guaranteed when there is a single root with congruent sign in the bracket. However the likelihood of converging to a congruent root, is still very high due to the initial side rejection strategy explained above, by decreasing the bracketing increment to a range that guarantees sampling a basin br_rate <.5, you once more recover guaranteed convergence to the signed root.

    Variable calculations are all f64.

    :param f_op: Can be a function or a function operator (tuple) that includes its arguments. It receives a single scalar value for the point estimate. 
     :param br_rate: (0,1). The bracket increment, at .5 it's classic bisection, if you expect roots to be clustered on the right then >.5 might be suitable. Left clustered <.5. But a smaller br_rate should always have more definite convergence.
    :param sign: =1 we expect to have f(lo)<f(root)<f(hi). if -1 we expect f(lo)>f(root)>f(hi). If this expectation is unknown, it controls the bracketing bias eg if f is all positives and sign=1, then the bracket will reduce from right to left at (1 - br_rate) until reaching hi, if negatives and sign=1 then left to right at br_rate. Note: If both sides are wrong then convergence will not occur in the single root case.

    """
    fo, f = op_call_args(f_op, lo), op_call_args(f_op, hi)
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
    ict = np.int64(max_iters)
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

        f = op_call_args(f_op,lam)
        op_bracket= op_bracket or f < 0.
        if sign == -1: f = -f
        ict -= 1

        if f > 0.:
            hi = lam
        else:lo = lam

        if abs(lamo - lam) < er_tol: break

    return lam,lo,hi,2 if not op_bracket else 1 if ict==0 else 0


def op_call_args(cal,args):
    ct=isinstance(cal,Callable) #otherwise tuple|list
    rt=isinstance(args,tuple|list) #otherwise single element.
    if ct and rt:
        return cal(*args)
    if ct and not rt:
        return cal(args)
    if not ct and rt:
        return cal[0](*args,*cal[1:])
    #if not ct and not rt:
    return cal[0](args,*cal[1:])
