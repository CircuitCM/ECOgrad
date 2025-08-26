import importlib.util
from typing import Callable

if importlib.util.find_spec("numba") is not None:
    import numba as nb
    from numba import types
    from numba.extending import register_jitable, overload
    _dft = dict(fastmath=True, error_model='numpy') #fastmath 4em instead of 1em, error_model numpy enables datatype specific optimizations.
    #can also add nogil = True, but only helps if you are launching numba dispatchers from separate python threads. If sequential execution is used
    #and you launch threads with a parallel decorator, there will be no performance improvement. I have experienced numpy buffer problems with nogil
    #even from a single-threaded python interpreter.
    jit_s = _dft
    jit_sc = jit_s | dict(cache=True)
    jit_si = jit_s | dict(inline='always')
    jit_sci = jit_si | dict(cache=True)

    jit_p = _dft | dict(parallel=True)
    jit_pc = jit_p | dict(cache=True)
    jit_pi = jit_p | dict(inline='always')
    jit_pci = jit_pi | dict(cache=True)

    #jt = nb.njit(**jit_s)
    jtc = nb.njit(**jit_sc)
    #jti = nb.njit(**jit_si)
    #jtic = nb.njit(**jit_sci)

    #jtp = nb.njit(**jit_p)
    jtpc = nb.njit(**jit_pc)
    #jtpi = nb.njit(**jit_pi)
    #jtpic = nb.njit(**jit_pci)

    _rg = register_jitable
    rg = _rg(**jit_s)
    rgc = _rg(**jit_sc)
    rgi = _rg(**jit_si)
    rgic = _rg(**jit_sci)

    #rgp = _rg(**jit_p)
    #rgpc = _rg(**jit_pc)
    #rgpi = _rg(**jit_pi)
    #rgpic = _rg(**jit_pci)

    # I'm pretty sure caching is redundant for overloads.
    #ovs = lambda impl: overload(impl, jit_options=jit_s)
    ovsi = lambda impl: overload(impl, jit_options=jit_s, inline='always')
    #ovsc = lambda impl: overload(impl, jit_options=jit_sc)
    #ovsic = lambda impl: overload(impl, jit_options=jit_sc, inline='always')
else:
    jtc=jtpc=rg=rgc=rgi=rgic=lambda i:i
    ovsi=lambda i:lambda s:s


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


@ovsi(op_call_args)
def _op_call_args(cal,args):
    ct=isinstance(cal,types.Callable) #otherwise tuple|list
    rt=isinstance(args,types.BaseTuple|types.LiteralList) #otherwise single element.
    #print('Here',ct,rt)
    if ct and rt:
        return lambda cal,args: cal(*args)
    if ct and (not rt):
        return lambda cal,args: cal(args)
    if (not ct) and rt:
        return lambda cal,args: cal[0](*args,*cal[1:])
    #if not ct and not rt:
    return lambda cal,args: cal[0](args,*cal[1:])