"""
Copyright by Artem Vorontsov, AIST Company, 2021
email: aist.consulting@gmail.com
"""
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
from typing import Optional, Union

tf_dtype_r = tf.float32
tf_dtype_c = tf.complex64
np_dtype_r = np.float32

# Allow mixed presition
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# GPU/CPU usage for graph computations
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# GPU/CPU usage for graph computations
gpuflag = 1
if gpuflag == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def init_sim(L, N, Nc, Mc, lens_rad, lens_type: Union[str, list] = 'conv'):
# Create modeling square
    [x, y] = set_domain(L, N, Nc, Mc)
    [X, Y] = np.meshgrid(x, y)
    r2 = X**2 + Y**2

# Create spectral grid
    la, mu = spectral_domain(N, L, Nc, Mc)
    La, Mu = np.meshgrid(la, mu)
    ro2 = La**2 + Mu**2

# Create mask
    eps = L/2.0 - L/6.0
    # mask = np.ones((N, N), dtype=np_dtype_r)
    # mask = np.zeros((N, N), dtype=np_dtype_r)
    # mask[np.logical_and(np.abs(X) < eps, np.abs(Y) < eps)] = 1.0
    mask = np.exp(-(X*X/eps**2)**10 - (Y*Y/eps**2)**10)
    # mask = np.exp(-(r2/eps**2)**10)

# Create lens mask
    if lens_type == 'conv':
        lens_mask = np.zeros((N, N), dtype=np_dtype_r)
        lens_mask[r2 < lens_rad**2] = 1.0
        lens_phase = r2*lens_mask
    else:
        lens_mask = np.zeros((N, N), dtype=np_dtype_r)
        lens_mask[np.logical_and(X**2 < lens_rad**2, Y**2 < lens_rad**2)] = 1.0

        lrx, lry = lens_rad/lens_type[0], lens_rad/lens_type[1]
        xc = np.linspace(-lens_rad, lens_rad - 2*lrx, lens_type[0], dtype=np_dtype_r) + lrx
        yc = np.linspace(-lens_rad, lens_rad - 2*lry, lens_type[1], dtype=np_dtype_r) + lry
        lens_phase = np.zeros((N, N), dtype=np_dtype_r)
        for x in xc:
            for y in yc:
                loc_lens_mask = np.zeros((N, N), dtype=np_dtype_r)
                loc_lens_mask[np.logical_and((X - x)**2 < lrx**2, (Y - y)**2 < lry**2)] = 1.0
                lens_phase += ((X - x)**2 + (Y - y)**2)*loc_lens_mask

    return X, Y, r2, ro2, mask, lens_mask, lens_phase
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def turb_screen_sp(ro2, in_scl, out_scl, t_type, L, kw, z_diff, a):

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
    eleven_sixth = 11/6
    seven_sixth = 7/6
    coef1 = in_scl*in_scl/(5.92*5.92)
    coef2 = 4*math.pi*math.pi/(out_scl*out_scl)
    coef3 = in_scl/3.3
    mlt = kw**2*z_diff*a # *z_n*Cn2_n - will be accounted in atm_prop_graph_vx procedure

# Computation of (kappa_n)^(-11/3)*d^2kappa_n
    size = ro2.shape
    spectrum = np.zeros(size, dtype=np_dtype_r)

    if t_type == 'KOLMOGOROV':
        spectrum[ro2 > 0] = 0.033*(ro2[ro2 > 0])**(-eleven_sixth)

    elif t_type == 'TATARSKY':
        spectrum[ro2 > 0] = 0.033*np.exp(-ro2[ro2 > 0]*coef1)*(ro2[ro2 > 0])**(-eleven_sixth)

    elif t_type == 'vonKARMAN':
        spectrum = 0.033*np.exp(-ro2*coef1)/(ro2 + coef2)**eleven_sixth

    elif t_type == 'ANDREWS':
        ro = np.sqrt(ro2)
        Tmp = 0.033*np.exp(-ro2*coef1)*(ro2 + coef2)**(-eleven_sixth)
        spectrum = Tmp*(1.0 + 1.802*ro*coef3 - 0.254*(ro*coef3)**seven_sixth)

# Normalization
    spectrum = spectrum*mlt*math.pi*2.0*np.sqrt(2.0)
    spectrum = (2.0*math.pi/L)*np.sqrt(spectrum)/np.sqrt(2.0)

# FFT shift
    spectrum = np.fft.fftshift(spectrum)

    return spectrum
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_initial_amplitude(N, X, Y, beamType, d):
    ampl = np.zeros((N, N), dtype=np_dtype_r)
    r2 = X**2 + Y**2

    if beamType == 'ONEZERO':
        I = np.argwhere(r2 < d)
        ampl[I] = 1

    elif beamType == 'GAUSS':
        ampl = np.exp(-r2/d**2)

    elif beamType == 'SUPERGAUSS':
        ampl = np.exp(-(r2/d**2)**10)

    return ampl
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def set_domain(L, N, Nc, Mc):
    x = np.linspace(-L/2, L/2, N, dtype=np_dtype_r)
    y = np.linspace(-L/2, L/2, N, dtype=np_dtype_r)
    x = x - x[Nc]
    y = y - y[Mc]
    return x, y
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def spectral_domain(N, L, Nc, Mc):
    la = np.linspace(-N*math.pi/L, N*math.pi/L, N, dtype=np_dtype_r)
    mu = np.linspace(-N*math.pi/L, N*math.pi/L, N, dtype=np_dtype_r)
    la = la - la[Nc]
    mu = mu - mu[Mc]
    return la, mu
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def cn2_law(iters, type='const'):

    if type == 'sin':
        cn2 = 1.0*1e-16*(np.sin(np.linspace(0.0, 3.0*math.pi, iters, dtype=np_dtype_r)) + 1.0 + 1.0*1e-2)

    elif type == 'const':
        cn2 = 1.0*1e-15*np.ones((iters,), dtype=np_dtype_r)

    else:
        cn2 = 1.0*1e-15*np.ones((iters,), dtype=np_dtype_r)

    return cn2
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def save2hdf5(pp_inten, fp_inten, cn2_backet, filename):

# Normalization
    pp_inten = (255.0*(pp_inten - np.min(pp_inten))/(np.max(pp_inten) - np.min(pp_inten))).astype(int)
    fp_inten = (255.0*(fp_inten - np.min(fp_inten))/(np.max(fp_inten) - np.min(fp_inten))).astype(int)

    dic = {'Cn2': cn2_backet,
           'Pupil Plane': pp_inten,
           'Focal Plane': fp_inten}

    with h5py.File(filename, 'w') as hf:

        for key in dic:
            hf.create_dataset(key, data=dic[key])

    print(f'Data was successfully saved in {filename}')

    return []
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TF implementations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def atm_prop_graph_v3(N=512, L=1.0,
                      n_scr=1, batch_size=10,
                      use_2d_input_field=True,
                      initial_phase_randomization: Optional[str] = None,
                      phase_screens_in_batch='identical',
                      use_internal_screens_gen=True,
                      use_absorption_mask=True,
                      scope='atm_sim'):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

# Input propagation direction
        propagation_direction = tf.placeholder(shape=(), name='dpropagation_direction', dtype=tf.string)

# Input propagation distance
        dist = tf.placeholder(shape=(n_scr,), name='dist', dtype=tf_dtype_r)

# Input field
        if use_2d_input_field:
            field0 = tf.placeholder(shape=(N, N), name='field0', dtype=tf_dtype_c)
            field = tf.stack([field0]*batch_size)
        else:
            field0 = tf.placeholder(shape=(batch_size, N, N), name='field0', dtype=tf_dtype_c)
            field = field0

# Input Cn2 for screens
        cn2 = tf.placeholder(shape=(batch_size, n_scr), name='cn2', dtype=tf_dtype_r)

# Input random seed for screens
        scr_seed = tf.placeholder(shape=(4,), name='scr_seed', dtype=tf.int32)

# Input random seed for initial random phase
        phase_seed = tf.placeholder(shape=(2,), name='phase_seed', dtype=tf.int32)

# Initialize r2
        r2_ini = lambda: tf.placeholder(shape=(N, N), name='r2_ini', dtype=tf_dtype_r)
        r2 = tf.get_variable('r2', initializer=r2_ini, dtype=tf_dtype_r, trainable=False)

# Get fftshift(r2)
        r2s = tf.roll(r2, shift=[N//2, N//2], axis=[0, 1])

# Initialize statistics
        spectrum_ini = lambda: tf.placeholder(shape=(N, N), name='spectrum_ini', dtype=tf_dtype_r)
        spectrum = tf.get_variable(name='spectrum', initializer=spectrum_ini, dtype=tf_dtype_r, trainable=False)

# Initialize mask
        mask_ini = lambda: tf.placeholder(shape=(N, N), name='mask_ini', dtype=tf_dtype_r)
        mask = tf.get_variable(name='mask', initializer=mask_ini, dtype=tf_dtype_r, trainable=False)
        if not use_absorption_mask:
            mask = tf.ones_like(mask)

# Compute convolution kernel
        ker = tf_get_kernel(r2s, dist, L, N)

# Generate phase screens
        if use_internal_screens_gen:
            if phase_screens_in_batch == 'identical':
                # Identical screens in batch
                screens = tf_gen_screens(spectrum, scr_seed, n_scr)
                screens = tf.sqrt(tf.reshape(cn2[0, :]*dist, (n_scr, 1, 1)))*screens  # accounting of *z_n*Cn2_n
                screens = tf.complex(0.0, tf.stack([screens]*batch_size, axis=0))
            else:
                # Random screens in batch
                screens = tf_gen_screens(spectrum, scr_seed, batch_size*n_scr)
                screens = tf.reshape(screens, (batch_size, n_scr, N, N))
                screens = tf.sqrt(tf.reshape(cn2*dist, (batch_size, n_scr, 1, 1)))*screens # accounting of *z_n*Cn2_n
                screens = tf.complex(0.0, screens)
        else:
            screens = tf.placeholder(shape=(batch_size, n_scr, N, N), name='screens', dtype=tf_dtype_r)
            screens = tf.complex(0.0, screens)

        if propagation_direction == 'inverse':
            screens = tf.reverse(screens, axis=[1])
            dist = tf.reverse(dist, axis=[0])

# Generate random phases and prepare initial fields
        if initial_phase_randomization is not None:
            field = field*tf.exp(get_random_phase(initial_phase_randomization, phase_seed, batch_size, N))

# Propagate loop over phase screens
        j = tf.constant(0)
        cond = lambda j, var: tf.less(j, n_scr)

        def body(j, var):

    # Multiply field by the absorption mask
            var = var*tf.complex(tf.stack([mask]*batch_size), 0.0)

    # Propagate field
            loc_ker = ker[j, :, :]
            loc_ker = tf.stack([loc_ker]*batch_size)
            var = tf_propagate(var, loc_ker)

    # Pass phase screens
            loc_screen = screens[:, j, :, :]
            var = var*tf.exp(loc_screen)

            j = tf.add(j, 1)

            return [j, var]

        loop = tf.while_loop(cond, body, [j, field], parallel_iterations=10, swap_memory=False)
        field1 = loop[1]

        if use_internal_screens_gen:
            screens = tf.imag(tf.reshape(screens, (batch_size, n_scr, N, N)))

    out = dict(propagation_direction=propagation_direction,
               dist=dist, field0=field0,
               Cn2=cn2,
               scr_seed=scr_seed, phase_seed=phase_seed,
               screens=screens, field1=field1)

    return out
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def tf_gen_screens(spectrum, seed, n_scr):

    if n_scr%2:
        n_scr2 = (n_scr + 1)//2
    else:
        n_scr2 = n_scr//2

    spectrum = tf.stack([tf.complex(spectrum, 0.0)]*n_scr2)

    shape = spectrum.shape

    wn = tf.complex(tf.random.stateless_normal(shape, seed=seed[:2], dtype=tf_dtype_r),
                    tf.random.stateless_normal(shape, seed=seed[2:], dtype=tf_dtype_r))
    screens = tf.fft2d(spectrum*wn)
    screens = tf.concat([tf.real(screens), tf.imag(screens)], axis=0)
    screens = screens[:n_scr, :, :]

    return screens
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def tf_get_kernel(r2, dist, L, N):

    n = dist.shape[0]
    step = L/N

    z = math.pi*math.pi*dist/(L*L*step*step)
    z = tf.stack([tf.stack([z]*N)]*N)
    z = tf.transpose(z, perm=[2, 0, 1])

    r2 = tf.stack([r2]*n)

    ker = tf.exp(tf.complex(0.0, z*r2))

    return ker
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def tf_propagate(field_before, ker):
    tmp1 = tf.fft2d(field_before)
    tmp2 = tmp1*ker
    field_after = tf.ifft2d(tmp2)
    return field_after
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_random_phase(type, seed, batch_size, N):
    mlt = 2.0*math.pi
    # range = (1.0, 1e5*tf.random_uniform(shape=(), dtype=tf_dtype_r) + 1.0)
    range = (1.0, 1.0e3)

    if type == 'conv':
        phi = mlt*tf.complex(0.0, tf.random.stateless_normal(shape=(batch_size, N, N), seed=seed, dtype=tf_dtype_r))

    else:
        t = tf.cast(tf.linspace(range[0], range[1], batch_size), dtype=tf_dtype_r)
        t = tf.stack([tf.stack([t]*N)]*N)
        t = tf.transpose(t, perm=[2, 0, 1])
        tmp = mlt*tf.random.stateless_normal(shape=(N, N), seed=seed, dtype=tf_dtype_r)
        # tmp = mlt*tf.random_normal(shape=(N, N), dtype=tf_dtype_r)
        tmp = tf.stack([tmp]*batch_size)
        phi = tf.complex(0.0, t*tmp)

    return phi
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def lens_prop_graph(N=512, L=1.0,
                    batch_size=10,
                    scope='lens_sim'):

    with tf.variable_scope('atm_sim', reuse=tf.AUTO_REUSE):
# Initialize r2
        r2_ini = lambda: tf.placeholder(shape=(N, N), name='r2_ini', dtype=tf_dtype_r)
        r2 = tf.get_variable('r2', initializer=r2_ini, dtype=tf_dtype_r, trainable=False)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# Input focal distance
        f = tf.placeholder(shape=(1,), name='focus', dtype=tf_dtype_r)

# Input image distance
        dist = tf.placeholder(shape=(1,), name='dist', dtype=tf_dtype_r)

# Input field
        field0 = tf.placeholder(shape=(batch_size, N, N), name='field0', dtype=tf_dtype_c)

# Initialize lens mask
        lens_mask_ini = lambda: tf.placeholder(shape=(N, N), name='lens_mask_ini', dtype=tf_dtype_r)
        lens_mask = tf.get_variable(name='lens_mask', initializer=lens_mask_ini, dtype=tf_dtype_r, trainable=False)
        lens_mask = tf.complex(tf.stack([lens_mask]*batch_size), 0.0)

# Get fftshift(r2)
        r2s = tf.roll(r2, shift=[N//2, N//2], axis=[0, 1])

# Pass lens
        field = field0*lens_mask*tf.exp(tf.complex(0.0, tf.stack([r2/f]*batch_size)))

# Compute convolution kernel
        ker = tf_get_kernel(r2s, dist, L, N)
        ker = tf.tile(ker, [batch_size, 1, 1])

# Propagate
        field1 = tf_propagate(field, ker)

    out = dict(focus=f, dist=dist, field0=field0,
               field1=field1)

    return out
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def lens_prop_graph_v2(N=512, L=1.0,
                       batch_size=10,
                       scope='lens_sim'):

    with tf.variable_scope('atm_sim', reuse=tf.AUTO_REUSE):
# Initialize r2
        r2_ini = lambda: tf.placeholder(shape=(N, N), name='r2_ini', dtype=tf_dtype_r)
        r2 = tf.get_variable('r2', initializer=r2_ini, dtype=tf_dtype_r, trainable=False)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
# Input focal distance
        f = tf.placeholder(shape=(1,), name='focus', dtype=tf_dtype_r)

# Input image distance
        dist = tf.placeholder(shape=(1,), name='dist', dtype=tf_dtype_r)

# Input field
        field0 = tf.placeholder(shape=(batch_size, N, N), name='field0', dtype=tf_dtype_c)

# Initialize lens mask
        lens_mask_ini = lambda: tf.placeholder(shape=(N, N), name='lens_mask_ini', dtype=tf_dtype_r)
        lens_mask = tf.get_variable(name='lens_mask', initializer=lens_mask_ini, dtype=tf_dtype_r, trainable=False)
        lens_mask = tf.complex(tf.stack([lens_mask]*batch_size), 0.0)

# Initialize lens phase
        lens_phase_ini = lambda: tf.placeholder(shape=(N, N), name='lens_phase_ini', dtype=tf_dtype_r)
        lens_phase = tf.get_variable(name='lens_phase', initializer=lens_phase_ini, dtype=tf_dtype_r, trainable=False)
        lens_phase = tf.stack([lens_phase]*batch_size)

# Get fftshift(r2)
        r2s = tf.roll(r2, shift=[N//2, N//2], axis=[0, 1])

# Pass lens
        field = field0*lens_mask*tf.exp(tf.complex(0.0, lens_phase/f))

# Compute convolution kernel
        ker = tf_get_kernel(r2s, dist, L, N)
        ker = tf.tile(ker, [batch_size, 1, 1])

# Propagate
        field1 = tf_propagate(field, ker)

    out = dict(focus=f, dist=dist, field0=field0,
               field1=field1)

    return out
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def reset_graph():
    global sess
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
# SCENARIOS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def gen_screens(sess, atm_graph, Z, Cn2, N):
# atm_graph should be with:
# use_2d_input_field = True,
# phase_screens_in_batch = 'various'

# Dummy input for propagation
    phase_seed = np.asarray([1, 2]) #any positive ints of len=4
    field0 = np.zeros((N, N))

    t = time.time()

# Compute phase screens of the shape = (batch_size, N, N)
    screen_seed = (1e10*np.random.rand(4)).astype(np.int64)
    screen_seed = np.asarray(screen_seed)

    feed_dict = {atm_graph['propagation_direction']: 'direct',
                 atm_graph['dist']: Z, atm_graph['Cn2']: Cn2,
                 atm_graph['field0']: field0,
                 atm_graph['scr_seed']: screen_seed, atm_graph['phase_seed']: phase_seed}

    # Generate set of phase screens
    screens = sess.run(atm_graph['screens'], feed_dict)

    batch_size = screens.shape[0]
    n_scr = screens.shape[1]
    print('Generation of {0:d}x{1:d} phase screens of the size {2:d}x{3:d} using Tensorflow: done in {4:f} s.'
          .format(batch_size, n_scr, N, N, time.time() - t))

    return screens
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def imaging(sess,
            atm_graph, Z, field0, Cn2,
            lens_graph, f, dist,
            n_starts,
            screens: Optional[np.ndarray] = None):
# atm_graph should be with:
# use_2d_input_field = True,
# phase_randomization = 'conv' or 'experimental'
# phase_screens_in_batch = 'identical'
# use_mask = True

# Usage
# inten = imaging(sess,
#                 atm_graph, Z, field0, Cn2,
#                 lens_graph, f, dist,
#                 n_starts)

# plt.imshow(np.flip(inten, axis=0), cmap='gray')
# plt.show()

    scr_seed = np.asarray([111, 2222, 40101, 2020])
    field2, inten = [], []

    t = time.time()
    for j in range(n_starts):
        # Compute propagation through the set of phase screens
        seed = (1e10*np.random.rand(2)).astype(np.int64)
        phase_seed = np.asarray(seed)

        feed_dict = {atm_graph['propagation_direction']: 'direct',
                     atm_graph['dist']: Z, atm_graph['Cn2']: Cn2,
                     atm_graph['field0']: field0,
                     atm_graph['scr_seed']: scr_seed, atm_graph['phase_seed']: phase_seed}

        if screens is not None:
            feed_dict = {atm_graph['screens']: screens}

        field1 = sess.run(atm_graph['field1'], feed_dict)

        # Pass lens
        feed_dict = {lens_graph['focus']: [f], lens_graph['dist']: [dist], lens_graph['field0']: field1}
        field2 = sess.run(lens_graph['field1'], feed_dict)

        inten.append(np.mean(np.abs(field2)**2, axis=0))

    batch_size = field2.shape[0]
    n_scr = Cn2.shape[1]
    N = field2.shape[1]
    print('Propagation of {0:d} field realizations through {1:d} phase screens and lens of the size {2:d}x{3:d} using Tensorflow: done in {4:f} s.'
          .format(batch_size*n_starts, n_scr, N, N, time.time() - t))

    inten = np.stack(inten, axis=0)
    inten = np.mean(inten, axis=0)

    return inten
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def propagation_forward(sess,
                        atm_graph, Z, field0, Cn2,
                        screens: Optional[np.ndarray] = None):
# atm_graph should be with:
# use_2d_input_field = False,
# phase_randomization = None
# phase_screens_in_batch = 'various'
# use_mask = True

# field0 should has the shape = (batch_size, N, N)

    scr_seed = (1e10*np.random.rand(4)).astype(np.int64)
    scr_seed = np.asarray(scr_seed)
    phase_seed = np.asarray([1, 2])

    t = time.time()

# Compute propagation through the set of phase screens
    feed_dict = {atm_graph['propagation_direction']: 'direct',
                 atm_graph['dist']: Z, atm_graph['Cn2']: Cn2,
                 atm_graph['field0']: field0,
                 atm_graph['scr_seed']: scr_seed, atm_graph['phase_seed']: phase_seed}

    if screens is not None:
        feed_dict.update({atm_graph['screens']: screens})

    field = sess.run(atm_graph['field1'], feed_dict)

    batch_size = field.shape[0]
    n_scr = Cn2.shape[1]
    N = field.shape[1]
    print('Propagation of {0:d} field realizations through {1:d} phase screens of the size {2:d}x{3:d} using Tensorflow: done in {4:f} s.'
          .format(batch_size, n_scr, N, N, time.time() - t))

    return field, scr_seed
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def propagation_backward(sess,
                        atm_graph, Z, field0, Cn2,
                        scr_seed,
                        screens: Optional[np.ndarray] = None):
# atm_graph should be with:
# use_2d_input_field = False,
# phase_randomization = None
# phase_screens_in_batch = 'various'
# use_mask = True

# field0 should has the shape = (batch_size, N, N)
# Z and Cn2 should be in the direct order

    phase_seed = np.asarray([1, 2])

    t = time.time()

# Compute propagation through the set of phase screens
    feed_dict = {atm_graph['propagation_direction']: 'inverse',
                 atm_graph['dist']: Z, atm_graph['Cn2']: Cn2,
                 atm_graph['field0']: field0,
                 atm_graph['scr_seed']: scr_seed, atm_graph['phase_seed']: phase_seed}

    if screens is not None:
        feed_dict.update({atm_graph['screens']: screens})

    field = sess.run(atm_graph['field1'], feed_dict)

    batch_size = field.shape[0]
    n_scr = Cn2.shape[1]
    N = field.shape[1]
    print('Propagation of {0:d} field realizations through {1:d} phase screens of the size {2:d}x{3:d} using Tensorflow: done in {4:f} s.'
          .format(batch_size, n_scr, N, N, time.time() - t))

    return field
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def pass_lens(sess,
              lens_graph, f, dist, field1):
# fields should has the shape = (batch_size, N, N)

    t = time.time()
# Pass lens
    feed_dict = {lens_graph['focus']: [f], lens_graph['dist']: [dist], lens_graph['field0']: field1}
    field = sess.run(lens_graph['field1'], feed_dict)


    N = field.shape[1]
    print('Pass lens of the size {0:d}x{1:d} using Tensorflow: done in {2:f} s.'.format(N, N, time.time() - t))

    return field
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def back_scattering(sess,
                    atm_graph, Z, field0, Cn2,
                    screens: Optional[np.ndarray] = None):
# atm_graph should be with:
# use_2d_input_field = False,
# phase_randomization = None
# phase_screens_in_batch = 'various'
# use_mask = True

# field0 should has the shape = (N, N)
# Z and Cn2 should be in the direct order

    batch_size = Cn2.shape[0]
    N = field0.shape[0]

    field0 = np.stack([field0]*batch_size, axis=0)
    field1, scr_seed = propagation_forward(sess, atm_graph, Z, field0, Cn2, screens)
    field1 = field1*np.exp(1j*(2.0*math.pi*np.random.rand(1, N, N)))
    field2 = propagation_backward(sess, atm_graph, Z, field1, Cn2, scr_seed, screens)

    return field2
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def MAPR(sess,
         atm_graph, Z, field0, Cn2,
         lens_graph, f,
         screens: Optional[np.ndarray] = None):
# atm_graph should be with:
# use_2d_input_field = False,
# phase_randomization = None
# phase_screens_in_batch = 'various'
# use_mask = True

# field0 should has the shape = (N, N)
# Z and Cn2 should be in the direct order

    batch_size = Cn2.shape[0]

    field0 = np.stack([field0]*batch_size, axis=0)
    field1, scr_seed = propagation_forward(sess, atm_graph, Z, field0, Cn2, screens)
    field2 = pass_lens(sess, lens_graph, f, f, field1)

    return np.abs(field1)**2, np.abs(field2)**2

# # %%%%%%%%%%%%%%%%%%% SIMULATION EXAMPLE - IMAGING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # General -------------------------------------
# # Grid size, grid centers
# N = 512
# Nc, Mc = int(N/2), int(N/2)
# # Wave length m^1 and wave number, m^(-1)
# wl = 0.6e-6
# kw = 2.0*math.pi/wl
# # Beam radius, m
# a = 1.0
# # Modeling square, m/(beam radius)
# L = 1.0/a
#
# # Propagation ----------------------------------
# # Diffraction distance, m
# z_diff = kw*a*a/2.0
# # Propagation distance, m/(z_diff)
# z = 4000.0/z_diff
# # Focal distance, m/(z_diff)
# f = z/2.0
# dist = f*z/(z - f)
# # Lens radius, m/(beam radius)
# lens_rad = 0.125/a
# # Type of the lens - 'conv' for radial lens or shape of square lenslet [int, int]
# lens_type = 'conv'
#
# # Turbulence -----------------------------------
# # Inner scale, m /(beam radius)
# in_scl = 0.01/a
# # Outer scale, m / (beam radius)
# out_scl = 2.0/a
# # Turbulence type
# t_type = 'vonKARMAN' #'KOLMOGOROV'
# # Number of phase screens
# n_scr = 10
#
# # Parallelization -------------------------------
# # Parallelization of the computations
# batch_size = 20
# # Number of restarts
# n_starts = 10
#
# # Output ----------------------------------------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# results_dir = current_dir + '\\Results'
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)
#
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# def main():
#
# # Initialize modeling geometry
#     X, Y, r2, ro2, mask, lens_mask, lens_phase = init_sim(L, N, Nc, Mc, lens_rad, lens_type)
#
# # Create initial amplitude, phase and initial field distribution
#     import cv2
#     ampl0 = np.mean(cv2.imread('lena.jpeg'), axis=2)/255.0
#     ampl0 = ampl0.astype(np_dtype_r)
#
#     phi0 = np.zeros((N, N), dtype=np_dtype_r)
#     field0 = ampl0*np.exp(1j*phi0)
#
# # Statistics for turbulent phase screens
#     spectrum = turb_screen_sp(ro2, in_scl, out_scl, t_type, L, kw, z_diff, a)
#
# # Propagation distances
#     Z = np.diff(np.linspace(0.0, z, n_scr + 1))
#
# # Graph for simulation of wave propagation
# # Parameters:
# # use_2d_input_field = True or False,
# # phase_randomization = None or 'conv' or 'experimental'
# # phase_screens_in_batch = 'various' or 'identical'
# # use_absorption_mask = True or False
#     reset_graph()
#     atm_graph = atm_prop_graph_v3(n_scr=n_scr, batch_size=batch_size,
#                                   use_2d_input_field=True,
#                                   initial_phase_randomization='conv',
#                                   phase_screens_in_batch='identical',
#                                   use_internal_screens_gen=True,
#                                   use_absorption_mask=True)
#     lens_graph = lens_prop_graph_v2(batch_size=batch_size)
#
# # Initialize variables in the graph
#     with tf.Session() as sess:
#         init = tf.global_variables_initializer()
#         list = tf.global_variables()
#         sess.run(init, feed_dict={list[0].initial_value.name: r2,
#                                   list[1].initial_value.name: spectrum,
#                                   list[2].initial_value.name: mask,
#                                   list[3].initial_value.name: lens_mask,
#                                   list[4].initial_value.name: lens_phase,
#                                  })
#
#         # Set up Cn2 value
#         cn2 = 1.0*1e-15*a**(2/3)*np.ones((batch_size, n_scr), dtype=np_dtype_r)
#
#         # Image propagation
#         inten = imaging(sess,
#                         atm_graph, Z, field0, cn2,
#                         lens_graph, f, dist,
#                         n_starts)
#
#         plt.imshow(np.flip(inten, axis=0), cmap='gray')
#         plt.show()
#
# if __name__ == "__main__":
#     main()