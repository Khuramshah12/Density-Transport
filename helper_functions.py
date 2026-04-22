import os
from shutil import copy

import h5py
import numpy as np
import numpy.fft as fft
from scipy.interpolate import RegularGridInterpolator as rgi

from athena_read import athinput as athinput_dict
 
ATHENA_BIN_PATH = '/home/abbkh891/athena_expanding_zade/bin/athena_ics'
#'/nesi/nobackup/uoo02637/khurram/PhD_2024/from_array24/'
 
#--- REINTERPOLATION FUNCTIONS ---#

def pad_array(x):
    # Pad data arrays at edges in order to make 'periodic'
    # Helps ensure that the interpolation returns a valid result at box edges.
    return np.pad(x, (1, 1), 'wrap')

def pad_grid(Xg):
    # Extending the grid one grid point before 0
    # and one grid point after Ls
    # Helps with interpolation of a periodic box
    if Xg.size == 1:
        l_point = [Xg[0]-1]
        r_point = [Xg[0]+1]
    else:
        l_point = [2*Xg[0] - Xg[1]]
        r_point = [2*Xg[-1] - Xg[-2]]
    return np.concatenate((l_point, Xg, r_point))

def reinterp_generate_grid(Ns, Ls, return_mesh=0, return_edges=0, pad=1):
    Xs = []
    for i in range(3):
        # z = 0, y = 1, x = 2
        Xe = np.linspace(0, Ls[i], Ns[i]+1)
        if return_edges:
            if pad:
                Xs.append(pad_grid(Xe))
            else:
                Xs.append(Xe)
        else:
            # Get cell-centered coordinates and extend
            Xg = 0.5*(Xe[1:] + Xe[:-1])
            if pad:
                Xs.append(pad_grid(Xg))
            else:
                Xs.append(Xg)
    
    if return_mesh:
        return np.meshgrid(*Xs, indexing='ij')
    else:
        return Xs

def get_grid_info(data):
    Ns = data['RootGridSize'][::-1]
    Ls = []
    for i in range(1, 4):
        string = 'RootGridX' + str(i)
        L = data[string][1] - data[string][0]
        Ls.append(L)
    Ls = np.array(Ls[::-1])
    return Ns, Ls


# --- GENERATING ICS FUNCTIONS --- #

# edit_athinput - edits the corresponding athinput file with quantites input below
# ONLY WORKS FOR athinput.from_array layout
def edit_athinput(athinput, save_folder, n_X, X_min, X_max, meshblock, h5name,
                  time_lim, dt, iso_sound_speed, expand, exp_rate,
                  exp_init=1, start_time=0.0, n_hst=0, n_hdf5=0, gamma=5./3., use_isothermal=False):
    ath_path = save_folder + athinput.split('/')[-1] + '_' + h5name.split('/')[-1].split('.h5')[0]
    copy(athinput, ath_path)
    ath_dict = athinput_dict(athinput)

    # tlim and start time
    ath_dict['time']['tlim'] = time_lim
    ath_dict['time']['start_time'] = start_time

    # hst output
    ath_dict['output1']['file_number'] = n_hst

    # hdf5 output
    ath_dict['output2']['dt'] = dt
    ath_dict['output2']['file_number'] = n_hdf5

    # X1
    ath_dict['mesh']['nx1'] = n_X[0]
    ath_dict['mesh']['x1min'] = X_min[0]
    ath_dict['mesh']['x1max'] = X_max[0]

    # X2
    ath_dict['mesh']['nx2'] = n_X[1]
    ath_dict['mesh']['x2min'] = X_min[1]
    ath_dict['mesh']['x2max'] = X_max[1]

    # X3
    ath_dict['mesh']['nx3'] = n_X[2]
    ath_dict['mesh']['x3min'] = X_min[2]
    ath_dict['mesh']['x3max'] = X_max[2]

    # meshblocks
    ath_dict['meshblock']['nx1'] = meshblock[0]
    ath_dict['meshblock']['nx2'] = meshblock[1]
    ath_dict['meshblock']['nx3'] = meshblock[2]

    # sound speed
    #ath_dict['hydro']['iso_sound_speed'] = iso_sound_speed

    # sound speed (if running isothermal) New addition for adiabatic eos
    if use_isothermal:
        ath_dict['hydro']['iso_sound_speed'] = iso_sound_speed if iso_sound_speed is not None else ath_dict['hydro'].get('iso_sound_speed', 0.0)
        # ensure gamma not set for isothermal run (some templates may have both)
        if 'gamma' in ath_dict['hydro']:
            try:
                del ath_dict['hydro']['gamma']
            except KeyError:
                pass
    else:
        # adiabatic: write gamma (required)
        ath_dict['hydro']['gamma'] = gamma if gamma is not None else ath_dict['hydro'].get('gamma', 5.0/3.0)

    # hdf5 file name
    ath_dict['problem']['input_filename'] = h5name

    # expansion
    ath_dict['problem']['expanding'] = 'true' if expand else 'false'
    ath_dict['problem']['expand_rate'] = exp_rate if expand else 0.

    with open(ath_path, 'w') as f:
        for key in ath_dict.keys():
            f.write(f'<{key}' + '>\n')
            for value in ath_dict[key]:
                f.write(f'{value} = {str(ath_dict[key][value])}' + '\n')
                if key == 'mesh' and value.startswith('o'):
                    f.write('\n')
            f.write('\n')

    return ath_path

def read_athinput(athinput, reinterpolate=0):

    ath_dict = athinput_dict(athinput)
    n_X = np.array([ath_dict['mesh']['nx1'], ath_dict['mesh']['nx2'], ath_dict['mesh']['nx3']])
    X_min = np.array([ath_dict['mesh']['x1min'], ath_dict['mesh']['x2min'], ath_dict['mesh']['x3min']])
    X_max = np.array([ath_dict['mesh']['x1max'], ath_dict['mesh']['x2max'], ath_dict['mesh']['x3max']])
    meshblock = np.array([ath_dict['meshblock']['nx1'], ath_dict['meshblock']['nx2'], ath_dict['meshblock']['nx3']])
    
    if reinterpolate:
        dt_hst = ath_dict['output1']['dt']
        dt = ath_dict['output2']['dt']
        expand = ath_dict['problem']['expanding']
        exp_rate = ath_dict['problem']['expand_rate']
        iso_sound_speed = ath_dict['hydro']['iso_sound_speed']
        return n_X, X_min, X_max, meshblock, dt_hst, dt, expand, exp_rate, iso_sound_speed
    else:
        return n_X, X_min, X_max, meshblock 

def generate_grid(X_min, X_max, n_X):
    # cell-edge grid
    xe = np.linspace(X_min[0], X_max[0], n_X[0]+1)
    ye = np.linspace(X_min[1], X_max[1], n_X[1]+1)
    ze = np.linspace(X_min[2], X_max[2], n_X[2]+1)
    # cell-centered grid
    xg = 0.5*(xe[:-1] + xe[1:])
    yg = 0.5*(ye[:-1] + ye[1:])
    zg = 0.5*(ze[:-1] + ze[1:])
    
    # grid spacings
    dx = xg[1] - xg[0]
    dy = np.inf if n_X[1] == 1 else yg[1] - yg[0]
    dz = np.inf if n_X[2] == 1 else zg[1] - zg[0]
    Xgrid, dX = (xg, yg, zg), (dx, dy, dz)
    return Xgrid, dX

def generate_mesh_structure(folder, athinput):
    ath_path = ATHENA_BIN_PATH
    cdir = os.getcwd()
    os.chdir(folder)
    os.system(f'{ath_path} -i {athinput} -m 1')
    blocks = read_mesh_structure(f'{folder}mesh_structure.dat')
    os.remove('mesh_structure.dat')
    n_blocks = blocks.shape[1]
    os.chdir(cdir)
    return n_blocks, blocks

# read meshblock - gets structure from mesh_structure.dat
def read_mesh_structure(data_fname):
    blocks = []
    with open(data_fname) as f:
        s = f.readline()
        while len(s) > 0:
            s = f.readline()
            # Looking for 'location = (%d %d %d)' and obtaining numbers in brackets
            if 'location =' in s:
                loc = s.split('=')[1].split('(')[1].split(')')[0]
                temp = [int(c) for c in loc.split() if c.isdigit()]
                blocks.append(temp)
    return np.array(blocks).T

# meshblock check - checks that athinput file and meshblock input match
def check_mesh_structure(blocks, n_X, meshblock):
    n_blocks = n_X / meshblock
    if n_blocks.prod() != blocks.shape[1]:
        raise AssertionError('Number of meshblocks doesnt match: must have input wrong in athinput or script')
    if np.any(blocks.max(axis=1) + 1 != n_blocks):
        raise AssertionError('Meshblock structure doesnt match: must have input wrong in athinput or script')


def make_meshblocks(folder, athinput, n_X, meshblock, one_D):
    if one_D:
        n_blocks = n_X[0] / meshblock[0]
        blocks = np.array([np.arange(n_blocks), np.zeros(n_blocks), np.zeros(n_blocks)])
    else:
        n_blocks, blocks = generate_mesh_structure(folder, athinput)
    check_mesh_structure(blocks, n_X, meshblock)
    return n_blocks, blocks

# shift and extend A - moves A from cell-centre to cell-faces
# and makes it periodic allowing for numerical derivatives to be computed at
# the boundary
def shift_and_extend_A(Ax, Ay, Az):
    Ax = reshape_helper(Ax, 0)
    Ax = 0.5*(Ax[:, :-1, :] + Ax[:, 1:, :])
    Ax = 0.5*(Ax[:-1, :, :] + Ax[1:, :, :])

    Ay = reshape_helper(Ay, 1)
    Ay = 0.5*(Ay[:-1, :, :] + Ay[1:, :, :])
    Ay = 0.5*(Ay[:, :, :-1] + Ay[:, :, 1:])

    Az = reshape_helper(Az, 2)
    Az = 0.5*(Az[:, :, :-1] + Az[:, :, 1:])
    Az = 0.5*(Az[:, :-1, :] + Az[:, 1:, :])
    return Ax, Ay, Az

# reshape helper - concatenates A differently along different axes to allow
# for periodicity
# Based off of Jono's MATLAB script
def reshape_helper(A, component):
    # component = 0 ⟺ x, 1 ⟺ y, 2 ⟺ z
    # pad notation: first tuple pad on (axis 0, axis 1, axis 2)
    # second tuple: (x, y) add last x entries to front of axis 
    # and first y entries to end of array
    # e.g. x = [0, 1, 2], pad(x, (1, 1), 'wrap') = [2, 0, 1, 2, 0]
    if component != 0:
        pad = ((1, 1), (0, 1), (1, 1)) if component == 1 else ((0, 1), (1, 1), (1, 1))
    else:
        pad = ((1, 1), (1, 1), (0, 1))
    return np.pad(A, pad, 'wrap')

def remove_prev_h5file(h5name):
    if os.path.isfile(h5name):  # 'overwrite' old ICs
        os.remove(h5name)

def setup_hydro_grid(n_X, X_grid, N_HYDRO, Dnf, UXf, UYf, UZf, BXf, BYf, BZf):
    # Athena orders cooridinates (Z, Y, X) while n_X is in the form (X, Y, Z)
    # It's easier to start from this ordering instead of having to do array
    # manipulations at the end.
    Hy_grid = np.zeros(shape=(N_HYDRO, *n_X[::-1]))

    # --- GRID CREATION --- #

    xg, yg, zg = X_grid
    Zg, Yg, Xg = np.meshgrid(zg, yg, xg, indexing='ij')

    # Place quantites on grid
    Hy_grid[0] = Dnf(Xg, Yg, Zg)
    Hy_grid[1] = Hy_grid[0] * UXf(Xg, Yg, Zg)  # using momentum for conserved values
    Hy_grid[2] = Hy_grid[0] * UYf(Xg, Yg, Zg)
    Hy_grid[3] = Hy_grid[0] * UZf(Xg, Yg, Zg)
    BXcc = BXf(Xg, Yg, Zg)
    BYcc = BYf(Xg, Yg, Zg)
    BZcc = BZf(Xg, Yg, Zg)

    # ignoring NHYDRO > 4 for now
    # if NHYDRO == 5: etc for adiabatic or CGL

    Xg, Yg, Zg = None, None, None  
    return Hy_grid, BXcc, BYcc, BZcc

def save_hydro_grid(h5name, Hy_grid, N_HYDRO, n_blocks, blocks, meshblock, remove_h5=1):
    Hy_h5 = np.zeros(shape=(N_HYDRO, n_blocks, *meshblock[::-1]))
    for h in range(N_HYDRO):
        for m in range(n_blocks):  # save to each meshblock individually
            off = blocks[:, m]
            ind_s = (meshblock*off)[::-1]
            ind_e = (meshblock*off + meshblock)[::-1]
            Hy_h5[h, m, :, :, :] = Hy_grid[h, ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]

    Hy_grid = None
    with h5py.File(h5name, 'a') as f:
        f['cons'] = Hy_h5
    Hy_h5 = None

def calc_and_save_B(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks, dx, dy, dz):
    # Get mean of B-field (inverting and redoing curl takes this away)
    B_mean = np.array([BXcc.mean(), BYcc.mean(), BZcc.mean()])
    
    # Calculate A from B using Fourier space by inverting the curl
    K = {
        k: 2j * np.pi / (X_max[k] - X_min[k]) * np.roll(np.arange(-(n_X[k]//2), (n_X[k]+1)//2, 1), (n_X[k]+1)//2)
        if n_X[k] > 1
        else np.array(0j)
        for k in range(3)
    }

    K_z, K_y, K_x = np.meshgrid(K[2], K[1], K[0], indexing='ij')
    K_2 = abs(K_x)**2 + abs(K_y)**2 + abs(K_z)**2
    K_2[0, 0, 0] = 1
    K_x /= K_2
    K_y /= K_2
    K_z /= K_2

    ftBX = fft.fftn(BXcc)
    ftBY = fft.fftn(BYcc)
    ftBZ = fft.fftn(BZcc)

    BXcc, BYcc, BZcc = None, None, None

    A_x = np.real(fft.ifftn(K_y*ftBZ - K_z*ftBY))
    A_y = np.real(fft.ifftn(K_z*ftBX - K_x*ftBZ))
    A_z = np.real(fft.ifftn(K_x*ftBY - K_y*ftBX))


    # A is cell-centred; shift to edges and make periodic
    A_x, A_y, A_z = shift_and_extend_A(A_x, A_y, A_z) 

    # Calculate B from A using finite-difference curl (this is how Athena++ calculates B from A)
    # Copied from Jono's MATLAB script
    # Bx
    B_mesh = meshblock + [1, 0, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + meshblock)[::-1]
        A_z_slice = A_z[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]+1] / dy
        A_y_slice = A_y[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]+1] / dz
        curlA_x = np.diff(A_z_slice, axis=1) - np.diff(A_y_slice, axis=0)
        if np.all(abs(curlA_x) < 1e-2):
            curlA_x *= 0.0
        B_h5[m, :, :, :] = B_mean[0] + curlA_x
    with h5py.File(h5name, 'a') as f:
        f['bf1'] = B_h5

    # By
    B_mesh = meshblock + [0, 1, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + meshblock)[::-1]
        A_x_slice = A_x[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]] / dz
        A_z_slice = A_z[ind_s[0]:ind_e[0], ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]+1] / dx
        curlA_y = np.diff(A_x_slice, axis=0) - np.diff(A_z_slice, axis=2)
        if np.all(abs(curlA_y) < 1e-2):
            curlA_y *= 0.0
        B_h5[m, :, :, :] = B_mean[1] + curlA_y
    with h5py.File(h5name, 'a') as f:
        f['bf2'] = B_h5

    # Bz
    B_mesh = meshblock + [0, 0, 1]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + meshblock)[::-1]
        A_y_slice = A_y[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]+1] / dx
        A_x_slice = A_x[ind_s[0]:ind_e[0]+1, ind_s[1]:ind_e[1]+1, ind_s[2]:ind_e[2]] / dy
        curlA_z = np.diff(A_y_slice, axis=2) - np.diff(A_x_slice, axis=1)
        if np.all(abs(curlA_z) < 1e-2):
            curlA_z *= 0.0
        B_h5[m, :, :, :] = B_mean[2] + curlA_z
    with h5py.File(h5name, 'a') as f:
        f['bf3'] = B_h5


def constB2_faceinterp(BXcc, BYcc, BZcc, h5name, n_X, X_min, X_max, meshblock, n_blocks, blocks):
    Ns, Ls = n_X[::-1], (X_max - X_min)[::-1]
    ze, ye, xe = reinterp_generate_grid(Ns, Ls, return_edges=1, pad=0) 
    zg, yg, xg = reinterp_generate_grid(Ns, Ls, pad=0) 

    BX_interp = rgi((pad_grid(zg), pad_grid(yg), pad_grid(xg)), pad_array(BXcc))
    BY_interp = rgi((pad_grid(zg), pad_grid(yg), pad_grid(xg)), pad_array(BYcc))
    BZ_interp = rgi((pad_grid(zg), pad_grid(yg), pad_grid(xg)), pad_array(BZcc))
    interps = [BX_interp, BY_interp, BZ_interp]

    BX_grid = np.meshgrid(zg, yg, xe, indexing='ij')
    BY_grid = np.meshgrid(zg, ye, xg, indexing='ij')
    BZ_grid = np.meshgrid(ze, yg, xg, indexing='ij')
    B_grids = [BX_grid, BY_grid, BZ_grid]

    B_faces = []

    for idx, B_grid in enumerate(B_grids):
        faces_Ns = Ns + np.roll([0, 0, 1], -idx)
        B_grid_z, B_grid_y, B_grid_x = B_grid
        pts = np.array([B_grid_z.ravel(), B_grid_y.ravel(), B_grid_x.ravel()]).T
        B_faces.append(interps[idx](pts).reshape(*faces_Ns))
    
    # Bx
    B_mesh = meshblock + [1, 0, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + B_mesh)[::-1]
        B_h5[m, :, :, :] = B_faces[0][ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]
    with h5py.File(h5name, 'a') as f:
        f['bf1'] = B_h5

    # By
    B_mesh = meshblock + [0, 1, 0]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + B_mesh)[::-1]
        B_h5[m, :, :, :] = B_faces[1][ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]
    with h5py.File(h5name, 'a') as f:
        f['bf2'] = B_h5

    # Bz
    B_mesh = meshblock + [0, 0, 1]
    B_h5 = np.zeros(shape=(n_blocks, *B_mesh[::-1]))
    for m in range(n_blocks):
        off = blocks[:, m]
        ind_s = (meshblock*off)[::-1]
        ind_e = (meshblock*off + B_mesh)[::-1]
        B_h5[m, :, :, :] = B_faces[2][ind_s[0]:ind_e[0], ind_s[1]:ind_e[1], ind_s[2]:ind_e[2]]
    with h5py.File(h5name, 'a') as f:
        f['bf3'] = B_h5
