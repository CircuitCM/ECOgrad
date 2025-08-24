
import numpy as np
import math as mt
import optional_numba as nbu
rng = np.random.default_rng()

def set_seed(s=None):
    global rng
    rng = np.random.default_rng(s)

def least_eps_delta(x,eps=2**(-52/2)):
    """The smallest scalar delta such that the largest floating point parameter can be represented to epsilon relative width.
    x is a 2d array with (# samples, problem parameter count).
    """
    return np.max(np.abs(x[0]))*eps

def make_prox(x,eps=2**(-52/2)):
    """Place the random vector samples as points proximal x[0] the sampling point."""
    d=least_eps_delta(x,eps)
    x[1:]*=d
    x[1:]+=x[0]
    return x,d

def sample_gauss(S):
    """Sample from the standard normal distribution."""
    rng.standard_normal(out=S)
    return S

def sample_sphere(S):
    """Samples point's on a sqrt(# dims) radius surface such that the scale of each parameter is invariant to the # of dimensions hence the sqrt(# dims) factor.

    As the # of dimensions increases, 'standard normal' sampling converges to an exact sqrt() surface radius in O(d^(-1/2)) time.
    """
    sample_gauss(S)
    nms=np.linalg.norm(S,axis=1)/mt.sqrt(S.shape[1])
    S[:]=(S.T/nms).T
    return S

def sample_orthog(x):
    """QR decomposition on standard normal produces orthogonal vectors that are also spread out uniformly on a sqrt(d) radius sphere's
     d-1 dimensional surface.

    While Sphere surface sampling produces row vectors with norm = sqrt(# dims). Orthogonal row vectors produce columns that have norm = sqrt(# dims),
     in addition to rows. As such the frobenius norm of a square Q = # dims exactly.

    Utilizing a full set of orthogonal vectors also amortizes the solution to a complete gradient interpolation.

    """
    sample_gauss(x) #this seems wrong double check later.
    Q, _ = np.linalg.qr(x[1:min(x.shape[0],x.shape[1])])
    x[1:]=Q
    return x

def minnormchg_grad(hat_a:np.ndarray, S:np.ndarray, c:np.ndarray,v=None):
    if v is None:
        G  = S @ S.T                # m×m Gram matrix
        y=np.linalg.pinv(G)@(c - S @ hat_a) #gram inv tends to have higher quality output, though fall back to system solve if instability.
        return hat_a + y @ S
    else:
        HS = v * S
        #HS/=np.linalg.norm(HS,axis=1)
        G  = S@np.diag(v) @ S.T
        y=np.linalg.pinv(G)@(c - S @ hat_a)
        return hat_a + y @ HS

def side1_sample(x:np.ndarray,S,f_eval,delta):
    """One-sided directional sample.
    If delta is a positive float we ignore the norm of S and just scale the samples by delta, our proximal sampling scalar. This will ignore distributional biases that exist in S such as L2 norms that do not = sqrt(d).
    If delta is None then normalization will be done by the L2 norm of the original direction vector. This projects all samples to the unit sphere and makes them equal weighted with respect to how the LMS or recursive/regression/interpolation seems them."""
    #Total cost # directional samples + 1 for the point.
    O=np.empty((S.shape[0]+1,S.shape[1]),dtype=x.dtype)
    O[:]=x
    O[1:]+=S*delta
    fs=f_eval(O)
    fs/=delta
    v=fs[1:]-fs[0]
    #These samples are scaled as directional derivatives always < ||true_gradient||, and NOT as stochastic gradient estimates. Which is why in monte carlo mean averaging they must be scaled by *d
    return v

def side2_sample(x,S,f_eval,delta):
    #Total cost 2*n evals
    #f1=value(x[1:])
    O=S*delta
    xh=x + O
    f1=f_eval(xh)
    del xh
    xl=x - O
    del O
    f2=f_eval(xl)
    nms=2*delta #np.linalg.norm(x1-x2,axis=1)
    #S=((2*S.T)/nms).T
    v=(f1/nms) - (f2/nms)
    return v

def gradcross_sample(S,grad):
    return S@grad


@nbu.jtc
def calc_info(g_est: np.ndarray, g: np.ndarray,gn2:float, infor: np.ndarray) -> None:
    """
    Compute metrics between true vector x and estimator y.
    Results are stored in outp as follows
      0: Cosine similarity between x and y
      1: Ratio of norms (||y||/||x||)
      2: Normalized MSE (MSE / ||x||^2)

    :param g_est: Gradient estimator vector.
    :param g: True gradient.
    :param gn2: Use the precalculated gradient norm.
    :param infor: 3 element info array of floats.
    """


    nx2=0.
    for v in g_est: nx2+= v * v
    ny2=gn2
    # ny2=0
    # for v in g: ny2+= v * v
    infor[0] = np.dot(g_est, g) / (nx2 * ny2) #Cosine sim
    infor[1] = mt.sqrt(ny2 / nx2) #norm
    rs=0.
    for i in range(g.shape[0]):
        r=g[i] - g_est[i]
        rs+= r*r
    infor[2] = rs/nx2 #MSE, can do rmse as well.
