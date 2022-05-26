import numpy as np


def mos_survival(stype, T):
    """
    Function to calculate survival probability for an array of temperatures
    stype :: integer specifying the type of mosquito survival scheme
    T :: multidimensional array e.g. [x, y, month]
    p :: element-wise survival probability
    """
    if stype==0:
        # Origin Martens as used in default LMM settings - 
        # see Hoshen and Morse (2004)
        p = 0.45 + 0.054*T - 0.0016*np.square(T)
        p[p < 0.0] = 0.0
        p[T > 40.0] =0.0
    elif stype==2:
        # Craig/Martens from Craig et al [add ref]
        p = np.exp(-1.0/(-4.4+1.31*T-0.03*np.square(T)))
        p[T<4.0] = 0.0
        p[T>39.9] = 0.0
        p[p<0.0] = 0.0
    else:
        print("error - survival type not recognised")
        return None
    p[p>1.0]=1.0
    return p


def gono_length(T, Dg, Tg):
    """
    Function to calculate gonotrophic cycles
    T :: multidimensional array e.g. [x, y, month]
    Dg, Tg :: constants
    Gdays :: element-wise gonotrophic cycle length in days
    """
    Gdays = np.ones(T.shape)*1000.0
    idx = T>Tg
    Gdays[idx] = 1.0 + np.divide(Dg, (T[idx] - Tg))
    return Gdays


def sporo_length(T, Ds, Ts):
    """
    Function to calculate sporogonic cycles
    T :: multidimensional array e.g. [x, y, month]
    Ds, Ts :: constants
    Sdays :: element-wise sporogonic cycle length in days
    """
    Sdays = np.ones(T.shape)*1000.0
    idx = T>Ts
    Sdays[idx] = np.divide(Ds, (T[idx] - Ts))

    return Sdays


def calc_mosquito_pop(mos0, p, rain, steplen, rainmult, rainoffset, n=10):
    """
    Function to calculate mosquito population, 
    new mosquitoes are added as a linear function of 
    total rainfall over the previous n timesteps
    
    p, rain :: 3 dimensional arrays of the same shape [x, y, t] or [y, x, t], 
               last dimension is assumed to be time

    steplen, rainmult, rainoffset :: constants
    mos0 :: number of mosquitoes at the first timestep

    mosquitoes :: mosquito population size, of the same shape as input
    """
    if not (rain.shape==p.shape):
        print("error: supplied array shapes do not match")
        return None
    # grow mosquito population according to rainfall each timestep
    dims = p.shape
    if len(dims)!=3:
        print("error: arrays must have 3 dimensions")
        return None
    mosquitoes = np.empty(dims)
    mosquitoes[0] = mos0
    nt = dims[len(dims)-1]
    # loop over time to simulate population dynamics
    for i in range(1, nt):
        istart = max(0, i-1-n)
        #print(istart)
        rainacc = np.mean(rain[:,:,istart:i], axis=2)
        #print(rainacc)
        mosquitoes[:,:,i] = mosquitoes[:,:,i-1]*np.power(p[:,:,i-1], steplen)+\
                            rainmult*rainacc + rainoffset
    mosquitoes[mosquitoes<0.0] = 0.0
    return mosquitoes


def calculate_R0(mosquitoes, p, Gdays, Sdays, r, HIA, HBI, b, c):
    """
    Function to calculate transmission coefficient R0 from previously
    calculated components
    mosquitoes, p, GDays, SDays :: multidimensional arrays of the same shape
                                   e.g. [x, y, month]
    HBI, r, HIA :: constants
    
    R0 :: element-wise transmission coefficient R0
    """
    if not ((mosquitoes.shape==p.shape)&
            (p.shape==Gdays.shape)&
            (p.shape==Sdays.shape)):
        print("error: supplied array shapes do not match")
        return None
    dims = p.shape
    TP = np.empty(dims)
    R0 = np.empty(dims)
    a = np.empty(dims)
    mu = np.empty(dims)
    idx = p>0.0
    a[idx] = HBI/Gdays[idx]
    mu[idx] = -1.0*np.log(p[idx])
    TP[idx] = np.divide(np.square(a[idx])*b*c*np.exp(
                        -1.0*mu[idx]*Sdays[idx] - r*HIA), 
                        (mu[idx]*r))
    R0[idx] = np.multiply(mosquitoes[idx], TP[idx])
    return R0