"""
Copyright by Artem Vorontsov, AIST Company, 2021
email: aist.consulting@gmail.com
"""

#
# Infinite (complex) screen generation using CuPy
# CuPy version should compare with CUDA version installed
#

import numpy as np
import cupy as cp
import math
from scipy.interpolate import interp2d

np_DTYPE = np.float32
np_cDTYPE = np.complex64
cp_DTYPE = cp.float32
cp_cDTYPE = cp.complex64

class infinite_screen():

    def __init__(self, wl, a, l, n, in_scl, out_scl, t_type, sl, flt_dmns, seed):
# Wave length m^1 and wave number, m^(-1)
        self.wl = 0.6e-6
        self.kw = 2.0*math.pi/wl
# Beam radius, m
        self.a = a
# Diffraction distance, m
        self.z_diff = self.kw*a*a/2.0
# Modeling square, m/(beam radius)
        self.l = l
# Grid size, grid centers
        self.n = n
        self.nc = n//2
        self.mc = n//2
# Create modeling square
        self.x, self.y = self.set_domain()
# Create dual square
        self.la, self.mu = self.sp_razb2d()
# Inner scale, m/(beam radius)
        self.in_scl = in_scl
# Outer scale, m/(beam radius)
        self.out_scl = out_scl
# Turbulence type
        self.t_type = t_type
# Random number generator's seed
        self.seed = seed
# Section length
        self.sl = sl
# Filters -> maximum(log2(sl) + 1) intervals
        if len(flt_dmns) - 1 > np.log(sl)/np.log(2.0) + 1:
            print('Warning: incorrect filters length!')
            print('Filtration will not be used.')
            self.flt_dmns = cp.asarray([0, np.inf])
            self.n_fltrs = 1
        else:
            self.flt_dmns = (2.0*math.pi/self.l)*flt_dmns
            self.n_fltrs = len(flt_dmns) - 1
# Partition of spectrum function
        self.spectrs = self.spectr_gen()
# Sections of infinity screen
        self.s1 = np.zeros((n, sl*n), dtype=np_cDTYPE)
        self.s2 = np.zeros((n, sl*n), dtype=np_cDTYPE)
        self.current_sect = np.zeros((n, sl*n), dtype=np_cDTYPE)
# Counter for infinity screen moving
        self.sect_cnt = 0
# Smoothing function
        tmp1 = np.cos(math.pi*np.linspace(-0.5, 0.5, sl*n, dtype=np_DTYPE))
        tmp2 = np.asarray(range(n)).astype(np_DTYPE)
        self.phi, tmp3 = np.meshgrid(tmp1, tmp2)
        self.phi = self.phi.astype(dtype=np_cDTYPE)

    def spectr_gen(self):

        """
        Normalization of spectral density:

        Sp ~ kw^2*z*Cn2*(kappa)^(-11/3) = kw^2*(z/z_diff)*z_diff*(Cn2/a^(-2/3))*a^(-2/3)*(a*kappa)^(-11/3)*a^(11/3) =
           = kw^2*z_n*Cn2_n*(kappa_n)^(-11/3)*(a^3*z_diff),

        where z_n, Cn2_n, kappa_n are dimensionless values:
        z_n = z/z_diff,
        Cn2_n = Cn2*a^(2/3),
        kappa_n = kappa_n*a

        Then
        Sp d^2kappa ~ kw^2*z_n*Cn2_n*(kappa_n)^(-11/3)*a^3*z_diff (a^2*d^2kappa)a^(-2) =
                    = (kw^2*z_diff*a)*z_n*Cn2_n*(kappa_n)^(-11/3)*d^2kappa_n
        """

        n = self.n
        la, mu = self.la, self.mu
        in_scl, out_scl = self.in_scl, self.out_scl
        t_type = self.t_type
        n_fltrs, flt_dmns = self.n_fltrs, self.flt_dmns

        eleven_sixth = 11.0/6.0
        seven_sixth = 7.0/6.0
        coef1 = in_scl*in_scl/(5.92*5.92)
        coef2 = 4.0*math.pi*math.pi/(out_scl*out_scl)
        coef3 = in_scl/3.3
        mlt2 = self.kw**2*self.z_diff*self.a # *z_n*Cn2_n - will be accounted in gen_inf_screens procedure

    # Computation of (kappa_n)^(-11/3)*d^2kappa_n
        la_mesh, mu_mesh = cp.meshgrid(la, mu)
        ro2_gl = la_mesh*la_mesh + mu_mesh*mu_mesh
        spectrs = cp.zeros((n_fltrs, n, n), dtype=cp_DTYPE)

        for j in range(n_fltrs):

        # Scaling of spectral variables
            mlt1 = 2**(2*(j + 1 - n_fltrs))
            tmp = mlt1*ro2_gl
            inds = np.logical_and((tmp > flt_dmns[j]**2), (tmp <= flt_dmns[j + 1]**2))
            if t_type == 'KOLMOGOROV':
                spectrs[j, inds] = 0.033/tmp[inds]**eleven_sixth

            elif t_type == 'TATARSKY':
                spectrs[j, inds] = 0.033*cp.exp(-tmp[inds]*coef1)/tmp[inds]**eleven_sixth

            elif t_type == 'vonKARMAN':
                spectrs[j, :, :] = 0.033*cp.exp(-tmp*coef1)/(tmp + coef2)**eleven_sixth

            elif t_type == 'ANDREWS':
                tmp1 = cp.sqrt(tmp)
                spectrs[j, inds] = (0.033*cp.exp(-tmp*coef1)/(tmp + coef2)**eleven_sixth)*(1.0 + 1.802*tmp1*coef3 - 0.254*(tmp1*coef3)**seven_sixth)

        # Normalization of spectrum function
            spectrs[j, :, :] = mlt1*mlt2*math.pi*2.0*cp.sqrt(2.0)*spectrs[j, :, :]
            spectrs[j, :, :] = (2.0*math.pi/self.l)*cp.sqrt(spectrs[j, :, :])/cp.sqrt(2.0)

        # FFT shift
            spectrs[j, :, :] = cp.fft.fftshift(spectrs[j, :, :])

        return spectrs

    def turb_screen(self, spectr, seed):

        cp.random.seed(seed)
        tmp1 = (cp.random.randn(self.n, self.n) + 1j*cp.random.randn(self.n, self.n))/cp.sqrt(2.0)
        tmp2 = spectr*tmp1.astype(cp_cDTYPE)
        screen = cp.fft.fft2(tmp2)

        return screen

    def turb_section(self, spectrs, sl, seed):

        screen = cp.zeros((sl, self.n, self.n), dtype=cp_cDTYPE)
        for j in range(sl):
            sd = seed + j
            screen[j, :, :] = self.turb_screen(spectrs, sd)

        section = cp.zeros((self.n, sl*self.n), dtype=cp_cDTYPE)
        for k in range(sl):
            ind1 = self.n*k
            ind2 = self.n*(k + 1)
            tmp2, _ = cp.meshgrid(cp.asarray(range(ind1, ind2)), cp.asarray(range(self.n)))

            for j in range(sl):
                tmp1 = screen[j, :, :]
                tmp4 = cp.exp((1j*2.0*math.pi/self.n)*(j*tmp2/sl)).astype(cp_cDTYPE)
                section[:, ind1 : ind2] = section[:, ind1: ind2] + (1.0/cp.sqrt(sl))*tmp1*tmp4

        return section

    def turb_filt_section(self, seed):

        n = self.n
        sl = self.sl
        n_fltrs = self.n_fltrs
        l = self.l

# Section's grid
        x = np.asarray(range(sl*n))
        y = np.asarray(range(n))

# First filtered section
        tmp1 = self.spectrs[n_fltrs - 1, :, :]
        section = self.turb_section(tmp1, sl, seed).get()

# Other filtered sections
        for j in range(2, n_fltrs + 1):
            tmp1 = self.spectrs[n_fltrs - j, :, :]
            sd = seed + sl*j
            mlt = 2**(j - 1)
            tmp2 = self.turb_section(tmp1, int(sl/mlt), sd).get()

# Support for filtered section
            supp_x = np.asarray(np.arange(0, sl*n + mlt, mlt))
            supp_y = np.asarray(np.arange(0, n + mlt, mlt))

# External points for correct interpolation
            p = int(n/mlt)
            tmp3 = np.zeros((p + 1, sl*p + 1), dtype=np_cDTYPE)
            tmp3[:p, :sl*p] = tmp2[:p, :]
            tmp3[-1, :sl*p] = tmp2[p - 1, :sl*p]
            tmp3[:, -1] = tmp2[:p + 1, 0]
            tmp4 = interp2d(supp_x, supp_y, np.real(tmp3), 'linear')(x, y) + \
                   1j*interp2d(supp_x, supp_y, np.imag(tmp3), 'linear')(x, y)
            section = section + tmp4.astype(np_cDTYPE)

        return section

    def turb_infinity_screen(self, t):

        n = self.n
        sl = self.sl
        m = int(n*sl/2)

        if t == 0:
            self.s1 = self.turb_filt_section(self.seed)
            self.seed = self.seed + sl*self.n_fltrs
            self.s2 = self.turb_filt_section(self.seed)
            self.current_sect[:, m:] = self.phi[:, m:]*self.s1[:, m:] + self.phi[:, :m]*self.s2[:, :m]

        ind_loc = t + m*(1 - self.sect_cnt)

        if ind_loc + n < n*sl:
            screen = self.current_sect[:, ind_loc: ind_loc + n]
        else:
            self.s1 = self.s2
            self.seed = self.seed + sl*self.n_fltrs
            self.s2 = self.turb_filt_section(self.seed)
            self.current_sect[:, :m] = self.current_sect[:, m:]
            self.current_sect[:, m:] = self.phi[:, m:]*self.s1[:, m:] + self.phi[:, :m]*self.s2[:, :m]
            self.sect_cnt = self.sect_cnt + 1
            screen = self.current_sect[:, ind_loc - m: ind_loc - m + n]

        return screen

    def sp_razb2d(self):
        la = cp.linspace(-self.n*math.pi/self.l, self.n*math.pi/self.l, self.n, dtype=cp_DTYPE)
        mu = la
        la = la - la[self.nc]
        mu = mu - mu[self.mc]

        return la, mu

    def set_domain(self):
        x = cp.linspace(-self.l/2.0, self.l/2.0, self.n, dtype=cp_DTYPE)
        y = x
        x = x - x[self.nc]
        y = y - x[self.mc]

        return x, y

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gen_inf_screens(state, cn2, dist):

# 'cn2' and 'dist' are float arrays of the length 'n_scr'
# 'cn2' and 'dist' should be normalized by *a^(2/3) and *z_diff^(-1) correspondingly

    t = state['t']
    n_scr = state['n_scr']

    if t == 0:
        wl = state['wl']
        a = state['a']
        n = state['n']
        l = state['l']
        in_scl = state['in_scl']
        out_scl = state['out_scl']
        t_type = state['t_type']
        sl = state['sl']
        flt_dmns = state['flt_dmns']
        seed = state['seed']

        if n_scr%2:
            n_scr2 = (n_scr + 1)//2
        else:
            n_scr2 = n_scr//2

        state['nscr2'] = n_scr2
        state['inf_screens'] = []

        screens = []
        for j in range(n_scr2):
            inf_screen = infinite_screen(wl, a, l, n, in_scl, out_scl, t_type, sl, flt_dmns, seed[j])
            print(f'Complex infinite screen with the number {j} was created')
            cscreen = inf_screen.turb_infinity_screen(t)
            screens.append(np.real(np.sqrt(dist[2*j]*cn2[2*j])*cscreen))
            if 2*j + 1 < n_scr:
                screens.append(np.imag(np.sqrt(dist[2*j + 1]*cn2[2*j + 1])*cscreen))

            state['inf_screens'].append(inf_screen)

    else:
        inf_screens = state['inf_screens']
        screens = []
        for j in range(state['nscr2']):
            cscreen = inf_screens[j].turb_infinity_screen(t)
            screens.append(np.real(np.sqrt(dist[2*j]*cn2[2*j])*cscreen))
            if 2*j + 1 < n_scr:
                screens.append(np.imag(np.sqrt(dist[2*j + 1]*cn2[2*j + 1])*cscreen))

    state['t'] = t + state['dt']

    return screens, state
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gen_inf_screens_batch(state, cn2, dist, batch_size, phase_screens_in_batch='various'):

# Generation of consequent screens of the shape (batch_size, n_scr, n, n)
# Variable 'phase_screens_in_batch' defines different (='various') of identical (='identical') screens will be used in batch
# 'cn2' and 'dist' are a lists of the size 'batch_size' with float arrays of the length 'n_scr'
#
# 'cn2' and 'dist' should be normalized by *a^(2/3) and *z_diff^(-1) correspondingly

    if phase_screens_in_batch == 'various':
        screens_batch = []
        for j in range(batch_size):
           screens, state = gen_inf_screens(state, cn2[j, :], dist)
           screens_batch.append(np.stack(screens, axis=0))

        screens_batch = np.stack(screens_batch, axis=0)

    else:
        screens, state = gen_inf_screens(state, cn2[0, :], dist)
        screens_batch = np.stack(screens, axis=0)
        screens_batch = np.stack([screens_batch]*batch_size, axis=0)

    return screens_batch, state
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
# Simulation  ----------------------------------
    # Grid size, grid centers
    n = 512
    # Wave length m^1 and wave number, m^(-1)
    wl = 0.6e-6
    kw = 2.0*math.pi/wl
    # Beam radius, m
    a = 1.0
    # Modeling square, m/(beam radius)
    l = 1.0/a
    # Diffraction distance, m
    z_diff = kw*a*a/2.0
# Turbulence -----------------------------------
    # Inner scale, m/(beam radius)
    in_scl = 0.01/a
    # Outer scale, m/(beam radius)
    out_scl = 2.0/a
    # Turbulence type
    t_type = 'vonKARMAN'
# Infinite screens parameters ---------------
    # Section length, sl*n
    sl = 2**3
    # Filtration areas
    flt_dmns = np.asarray([0.0, 2.0, 5.0, 20.0, np.inf])
    # Random seed
    seed = 1
# Duration -----------------------------------
    T = 10000
    dt = 100
    T = np.asarray(np.arange(0, T, dt))
# Propagation paremeters ---------------------
    # Distance, m/z_diff vs time T
    dist = (1000.0/z_diff)*np.ones(T.shape)
    # Structural parameter Cn2, m^(-2/3)*(beam radius)^(2/3) vs time
    Cn2 = 5.0*1e-15*a**(2/3)*np.ones(T.shape)

# Infinity screen constructor
    my_screen = infinite_screen(wl, a, l, n, in_scl, out_scl, t_type, sl, flt_dmns, seed)

# Infinity screen moving
    import time
    import matplotlib.pyplot as plt

    ti = time.time()
    test = []
    for t, d, cn2 in zip(T, dist, Cn2):
        screen = np.sqrt(d*cn2)*my_screen.turb_infinity_screen(t)
        print(t)
        test.append(screen[256, 256])

    print(f'Generation of infinite phase screen using CuPy: done in {time.time() - ti} s.')

    plt.plot(np.real(test))
    plt.show()

if __name__ == "__main__":
    main()