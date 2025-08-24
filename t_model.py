import numpy as np
import optional_numba as nbu
import math as mt

"""
-[FastT-V1]- A precise polynomial approximation of Student-t significance, interpolated from normal/gaussian significance levels and DoF.
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

_2ri= 1./mt.sqrt(2.0)

@nbu.rgi
def FastT_V1(z_sig, dof):
    az =abs(z_sig)
    if az <.01 :return z_sig
    if dof < 3.:
        # if az > 8.2590072: return 0. #erf will round to zero after this range with f64, though assumption is user will select a reasonable z-score range.
        err =mt.erf(z_sig *_2ri)
        if dof < 2.:
            return mt.tan((mt.pi / 2.0) * err)  # when dof is nearly 2. there will be significant error when z>5-ish, however its a fractional DoF and upper bounded so live with it.
        e = err * err
        den = 1.0 - e
        return mt.copysign(mt.sqrt(2. * e / den), err)

    # then poly
    p4, p3, p2, p1 = 0.004491152418700, 250.1508589461631, 0.243789262458383, 250.0952851341867
    if dof > p1:
        if dof > p1 + az * (p2 + az * (p3 + az * p4)): return z_sig

    return mt.copysign(tmv1_t1(az, dof), z_sig)


_p4o_t1=(22.72573914371964, -13.570479991819441, -0.338998884352377, 2.486243732743884, 0.207331149646325, 0.013561147304741, -1.628487578121468, 0.017894429239572, 0.000042785084111, 0.000035858284938, 1.000549459621945, 20.903260486632245, -11.339740516942792, -0.786224682733048, 2.085934258154098, 0.789626406029434, -0.002451118780528, -1.792855924513853, -0.263394466232209, 0.002740111828052, -0.00001488653939, 1.)

@nbu.rgi
def tmv1_t1(z: float, nu: float) -> float:
    #NOTE: This model is optimized for z .01 to 6, no guarantees outside of this range, make your own for a larger range but know that after
    #z>8 you'll run into floating point precision issues.
    p = z*z #same thing but it's a tuple
    r = 1.0/nu
    # Numerator
    C4 = _p4o_t1[0]
    C3 = _p4o_t1[1] + p * _p4o_t1[2]
    C2 = _p4o_t1[3] + p * (_p4o_t1[4] + p * _p4o_t1[5])
    C1 = _p4o_t1[6] + p * (_p4o_t1[7] + p * (_p4o_t1[8] + p * _p4o_t1[9]))
    C0 = _p4o_t1[10]
    num = (((C4 * r + C3) * r + C2) * r + C1) * r + C0
    D4 = _p4o_t1[11]
    D3 = _p4o_t1[12] + p * _p4o_t1[13]
    D2 = _p4o_t1[14] + p * (_p4o_t1[15] + p * _p4o_t1[16])
    D1 = _p4o_t1[17] + p * (_p4o_t1[18] + p * (_p4o_t1[19] + p * _p4o_t1[20]))
    D0 = _p4o_t1[21]
    den = (((D4 * r + D3) * r + D2) * r + D1) * r + D0
    return z * (num / den)
