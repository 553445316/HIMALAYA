import numpy as np
from scipy import stats
from scipy import signal
from numpy import linalg as LA 
from astropy.convolution import convolve_fft
import scipy.optimize as opt
from scipy.optimize import curve_fit

def transfer_mean_and_std(ps2d, ps2d_true): # ps2d_true / ps2d
    ps2d = np.array(ps2d)
    
    # Ensure ps2d_true is also a numpy array
    ps2d_true = np.array(ps2d_true)
    
    # Calculate the ratio of ps2d_true to ps2d in one operation
    ps_transfer = ps2d_true / ps2d

    # Calculate the mean and standard deviation along the 0th axis (stacked matrices)
    mean = np.mean(ps_transfer, axis=0)
    std = np.std(ps_transfer, axis=0)

    return mean, std





def svd_fg_removal(y, svd_data=None,n_comp=None):

    nf, nx, ny = y.shape
    data_ = y.reshape([nf, -1]).astype('float64')
    
    if svd_data is None:
        C_data = np.cov(data_)
        u_data, s_data, v_data = np.linalg.svd(C_data, hermitian=True)
        w_ = np.ones(nf)
    
        if n_comp is not None:
            if isinstance(n_comp, int) and 0 <= n_comp < nf:  # Check if n_comp is a valid integer
                w_[:n_comp] = 0
            else:
                raise ValueError("n_comp should be an integer between 0 and nf-1")
        if n_comp == nf:
            print("Warning: no n_comp found")
    else:
        s_data, u_data, w_, v_data = svd_data  # Extract u_data, w, and v_data from svd_data

    w = np.diag(w_)
    rec_map = u_data @ w @ v_data @ data_
    rec_map = rec_map.reshape([nf, nx, ny])

    del data_   # Free up memory

    return rec_map, (s_data, u_data, w_, v_data)



def calculate_svd_map_in_kbin(image, res_per, one_bin_edge, n_comp=None, eps=None):
    nz, nx, ny = image.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=res_per) # 1/Mpc
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=res_per)
    kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')
    k2d = np.sqrt(kx2d ** 2 + ky2d ** 2)
    k2d_flat = k2d.flatten()
    y_uv = np.fft.fft2(image.astype('float64'),axes=(1,2))
    yuv_flat = np.reshape(y_uv.flatten(), [nz, -1])
    data_ = np.reshape(image.flatten(), [nz, -1])

    _, _, binnumber = stats.binned_statistic(k2d_flat, values=None, statistic='count', bins=one_bin_edge)

    uv_data = []
    for i in range(nx * ny):
        if binnumber[i] == 1:
            uv_data.append(yuv_flat[:, i])

    uv_data = np.array(uv_data)
    C_data = np.cov(uv_data.T).real
    u_data, s_data, v_data = np.linalg.svd(C_data, hermitian=True)
    w_ = np.ones(nz)
    s_data/=nx*ny
    if n_comp is not None:
        w_[:n_comp] = 0
        w = np.diag(w_)
    elif eps is not None:
        n_comp = nz - np.searchsorted(s_data[::-1], eps)
        w_[:n_comp] = 0
        w = np.diag(w_)

    if n_comp == nz:
        print("warning: no n_comp found")

    rec_map = u_data @ w @ v_data @ data_
    rec_map = rec_map.reshape([nz, nx, ny])
    del uv_data,y_uv,yuv_flat,kx2d,ky2d,k2d,data_  # Free memory occupied by uv_data
    return  rec_map,n_comp, s_data




def truncate_array(x, a):
    """
    Truncate an array using the tanh mapping x -> a * tanh(x / a).

    Parameters:
    x (numpy.ndarray): The input array to be truncated.
    a (float): The truncation parameter.

    Returns:
    numpy.ndarray: The truncated array.
    """
    return a * np.tanh(x/a)





def convolution_with_psf_scipy(image, psf):
    """
    Convolve an image with a point spread function (PSF) using scipy's convolution.

    Args:
        image (numpy.ndarray): Input image.
        psf (numpy.ndarray): Point spread function.

    Returns:
        numpy.ndarray: Convolved image.

    Raises:
        ValueError: If the dimensions of the image and PSF are unsupported.
                    If the number of PSF slices does not match the number of image slices (for 3D PSF).
    """
    if image.ndim == 2 and psf.ndim == 2:
        # 2D image and 2D PSF, perform convolution
        return signal.fftconvolve(image, psf, mode='same')
    elif image.ndim == 3 and psf.ndim == 2:
        # 3D image and 2D PSF, perform convolution for each 2D slice of the image
        result = np.empty_like(image)
        for i in range(image.shape[0]):
            result[i] = signal.fftconvolve(image[i], psf, mode='same')
        return result
    elif image.ndim == 3 and psf.ndim == 3:
        # 3D image and 3D PSF, perform convolution for each corresponding slice
        if image.shape[0] != psf.shape[0]:
            raise ValueError("The number of PSF slices must match the number of image slices.")
        result = np.empty_like(image)
        for i in range(image.shape[0]):
            result[i] = signal.fftconvolve(image[i], psf[i], mode='same')
        return result
    else:
        raise ValueError("Unsupported dimensions for image and PSF arrays.")




def convolution_with_psf_astropy(image, psf, boundary='fill'):
    """
    Convolve an image with a point spread function (PSF) using astropy's convolution.

    Args:
        image (numpy.ndarray): Input image.
        psf (numpy.ndarray): Point spread function.
        boundary (str): Boundary condition for astropy's convolution function ('fill' or 'warp', default is 'fill').

    Returns:
        numpy.ndarray: Convolved image.

    Raises:
        ValueError: If the dimensions of the image and PSF are unsupported.
                    If the number of PSF slices does not match the number of image slices (for 3D PSF).
    """
    if image.ndim == 2 and psf.ndim == 2:
        # 2D image and 2D PSF, perform convolution
        return convolve_fft(image, psf, boundary=boundary,normalize_kernel=False)
    elif image.ndim == 3 and psf.ndim == 2:
        # 3D image and 2D PSF, perform convolution for each 2D slice of the image
        result = np.empty_like(image)
        for i in range(image.shape[0]):
            result[i] = convolve_fft(image[i], psf, boundary=boundary,normalize_kernel=False)
        return result
    elif image.ndim == 3 and psf.ndim == 3:
        # 3D image and 3D PSF, perform convolution for each corresponding slice
        if image.shape[0] != psf.shape[0]:
            raise ValueError("The number of PSF slices must match the number of image slices.")
        result = np.empty_like(image)
        for i in range(image.shape[0]):
            result[i] = convolve_fft(image[i], psf[i], boundary=boundary,normalize_kernel=False)
        return result
    else:
        raise ValueError("Unsupported dimensions for image and PSF arrays.")





def taper_uv_edge(image, alpha=0.1): # act on pfs not visibility itself
    nx, ny = image.shape
    
    # Calculate the Tukey window for the necessary indices
    half_nx = nx // 2
    tukey_window = signal.tukey(half_nx + 1, alpha)
    
    # Calculate the indices for the Tukey window
    index = abs(np.fft.fftfreq(nx, d=1. / nx).astype(int))
    
    # Create the 2D Tukey window
    win2d = np.outer(tukey_window[index], tukey_window[index])
    win2d[0, 0] = 1.0  # No effect on DC mode
    
    # Apply the Tukey window to the FFT result in-place
    image_fft = np.fft.fft2(image)
    image_fft *= win2d
    
    
    # Delete intermediate variables to save memory
    del  tukey_window, win2d
    
    return (np.fft.ifft2(image_fft)).real

def number_pixel_gaussian(fwhm, res_pixel=[16,16]):
    # units should be same for fwhm, res_pixel, like arcsec
    area= np.pi / (4 * np.log(2)) * (fwhm[0] * fwhm[1])
    return  area/(res_pixel[0] * res_pixel[1])




def zero_padding(image, pad_size):
    # Get the original dimensions of the image
    original_height, original_width = image.shape

    # Calculate the dimensions of the padded image
    padded_height = original_height + 2 * pad_size
    padded_width = original_width + 2 * pad_size

    # Create a zero-padded image of the desired size
    padded_image = np.zeros((padded_height, padded_width), dtype=image.dtype)

    # Place the original image in the center of the zero-padded image
    pad_x = (padded_width - original_width) // 2
    pad_y = (padded_height - original_height) // 2
    padded_image[pad_y:pad_y + original_height, pad_x:pad_x + original_width] = image

    return padded_image


def generate_gaussian_beam(hpbw_major, hpbw_minor, pixel_resolution=16, image_size=2048):
    """
    Generate a 2D Gaussian beam based on pixel resolution, pixel sizes in
    the x and y directions, and half-power beam widths (HPBW) along the major
    and minor axes. All units are in arcseconds.

    Parameters:
    - pixel_resolution: Number of pixels in x and y directions (rows, columns).
    - image_size (float): Size of each pixel. 
    - hpbw_major (float): Half-power beam width along the major axis 
    - hpbw_minor (float): Half-power beam width along the minor axis 

    Returns:
    - ndarray: 2D Gaussian beam profile.
    """
    rows = image_size
    cols = image_size
    x = np.linspace(-pixel_resolution * (cols-1 ) / 2, pixel_resolution * (cols-1 ) / 2, cols)
    y = np.linspace(-pixel_resolution * (rows-1 ) / 2, pixel_resolution * (rows-1 ) / 2, rows)
    X, Y = np.meshgrid(x, y)

    # Calculate the Gaussian beam profile
    major_sigma = hpbw_major / (2 * np.sqrt(2 * np.log(2)))
    minor_sigma = hpbw_minor / (2 * np.sqrt(2 * np.log(2)))
    gaussian_beam = np.exp(-((X / major_sigma) ** 2 + (Y / minor_sigma) ** 2) / 2)

    return gaussian_beam.astype('float32')




def oversampling_pfs(image, pad_size):

    original_height, original_width = image.shape
    padded_height = original_height + 2 * pad_size
    padded_width = original_width + 2 * pad_size
    image_fft= np.fft.fftshift(np.fft.fft2(image))
    
   # Create a zero-padded image of the desired size
    padded_image = np.zeros((padded_height, padded_width), dtype=image_fft.dtype)
    
    # Place the original image in the center of the zero-padded image
    pad_x = (padded_width - original_width) // 2
    pad_y = (padded_height - original_height) // 2
    padded_image[pad_y:pad_y + original_height, pad_x:pad_x + original_width] = image_fft
    oversampled_image = np.fft.ifft2(np.fft.ifftshift(padded_image))
    return oversampled_image.real






def crop_data_cube(data_cube, crop_size):
    """
    Crop a 3D data cube along the second and third axes.

    Parameters:
    - data_cube: A 3D NumPy array representing the data cube.
    - crop_size: Tuple specifying the size of the crop in the second and third dimensions, e.g., (height, width).

    Returns:
    - A cropped 3D NumPy array.
    """
    
    depth, height, width = data_cube.shape
    

    # Unpack the crop size
    crop_height = crop_size
    crop_width = crop_size
    if crop_height < height and crop_width < width:
        # Calculate the starting and ending indices for the crop
        start_h = (height - crop_height) // 2
        end_h = start_h + crop_height
        start_w = (width - crop_width) // 2
        end_w = start_w + crop_width
        return data_cube[:, start_h:end_h, start_w:end_w]
    return data_cube

def cube_apply_psf_beam(image,psf,beam): # data= psf@ beam*cube
    dirty_image = np.copy(image)
    for j in range(image.shape[0]):
        dirty_image[j] = signal.fftconvolve(dirty_image[j]*beam[j], psf[j], mode='same')      
    return dirty_image




def convert_units_jy2k(theta_arcs,nu_GHz):
    """
    https://science.nrao.edu/facilities/vla/proposing/TBconv
    Converts units for the given expression T_K = 1.222  * I / (nu^2 * theta_maj * theta_min): 
    as theta is also proportional to the wavelength, so the factor becomes frequency-independent 

    Parameters:
        I (float): Current in Jy/beam.
        nu_ref_GHz (float): Frequency in GHz. 
        theta_maj (float): Major axis angle of Synthesized beam (1'', 1/3600 degree)
        theta_min (float): Minor axis angle of Synthesized beam (1'')
         
    Returns:
        T (float): The result of the unit conversion (K).
    """

    # Perform the unit conversion
#    T =  I*1e-3*(13.6* (300.0/nu)**2 /theta_maj /theta_min)
#    T_K= 1e3*1.222*10**3*I/nu_GHz**2/theta_maj_arcsec /theta_min_arcsec
    return 1e3*1.222*1e3/nu_GHz**2/theta_arcs[0] /theta_arcs[1] # prefactor 1e3 for Jy->mJy



def crop_image(image, crop_size):
    height, width = image.shape[:2]
    
    # Calculate the dimensions for the central crop based on the specified crop_size
    new_height = crop_size
    new_width = crop_size
    
    # Calculate the coordinates for the center crop
    start_y = (height - new_height) // 2
    end_y = start_y + new_height
    start_x = (width - new_width) // 2
    end_x = start_x + new_width
    
    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return cropped_image



def dc_zero_image(image): # astype('float64') psf-=psf.mean(axis=(1,2))[:, np.newaxis, np.newaxis]
    # Calculate the FFT of the input image
    fft_image = np.fft.fft2(image)
    
    # Subtract the value at (0, 0) of the FFT
    fft_image[0, 0] = 0
    
    return np.fft.ifft2(fft_image).real





'''
def svd_fg_removal(y, n_comp=None, eps=None):
    """
    (numpy.ndarray): Input 3D numpy array.
    n_comp (int, optional): Number of components to be removed during SVD (default is None).
    eps (float, optional): Threshold for determining n_comp (default is None).s[n]>=eps

    Returns:
    numpy.ndarray: Resulting 3D numpy array after foreground removal.
    """
    nf, nx, ny = y.shape

    data_ = np.reshape(y.flatten(), [nf, -1])
    C_data = np.cov(data_.astype('float64'))
#    C_data = np.corrcoef(data_)
    u_data, s_data, v_data = np.linalg.svd(C_data, hermitian=True)
    w_ = np.ones(nf)
    if n_comp is not None:
        w_[:n_comp] = 0
        w = np.diag(w_)
    elif eps is not None:
        n_comp = nf-np.searchsorted(s_data[::-1], eps)
        w_[:n_comp] = 0
        w = np.diag(w_)
    if(n_comp==nf): 
        print("warning: no n_comp found")

    rec_map = u_data @ w @ v_data @ data_
    rec_map = rec_map.reshape([nf, nx, ny])
    
    del data_, C_data, u_data, v_data, w  # Free up memory

    return rec_map,n_comp,s_data
'''


def bin_centers_to_edges(bin_centers): # nbin->nbin+1
    # Calculate the bin width assuming uniform binning
    bin_width = bin_centers[1] - bin_centers[0]
    
    # Calculate the edges using the bin width
    bin_edges = [bin_centers[0] - bin_width / 2.0]
    for center in bin_centers:
        edge = center + bin_width / 2.0
        bin_edges.append(edge)
    
    return np.array(bin_edges)

def generate_cube_from_ps2d(ps2d, shape, pixel_length, bin_edges_klos, bin_edges_kper, seed=None):

    """
    Calculate the 2D cylindrical mean power spectrum from a 3D image spectral cube.

    Parameters
    ----------
    shape:  dimensions of cube (nz,nx,ny) [nlos, nper,nper]
    ps2d : numpy.ndarray
        2D cylindrical mean power spectrum. units = Mpc^3 [klos, kperp]
    pixel_length[0]: length/pixel * nz = size_LOS_cMpc : float
    length of the cube in comoving megaparsecs along the line of sight.    
   pixel_length[1]: length/pixel *nx = size_perp_cMpc : float
        Size of the cube in comoving megaparsecs perpendicular to the line of sight.
    bin_edges_klos : numpy.ndarray or None, optional
        Bin edges for line-of-sight wavenumbers. If None, they will be computed.
    bin_edges_kper : numpy.ndarray or None, optional
        Bin edges for perpendicular wavenumbers. If None, they will be computed.
    seed: Seed for random number generation (optional).

    Returns
    -------
    cube : numpy.ndarray
        3D image spectral cube.
    """
    nz,nx,ny= shape
    size_LOS_cMpc  = nz*pixel_length[0]
    size_perp_cMpc = nx*pixel_length[1]



    volume = size_perp_cMpc ** 2 * size_LOS_cMpc
    voxsize = volume / (nx * ny * nz)
    image = np.random.normal(loc=0., scale=1./np.sqrt(nz*nx*ny), size=shape)
    # Perform 3D FFT, make std(\delta k)= 1
    image_fft = np.fft.fftn(image.astype('float64'))
    scale =  np.sqrt(volume*ps2d) 
    
    # Calculate cylindrical wave numbers
    kz = np.abs(2 * np.pi * np.fft.fftfreq(nz, d=pixel_length[0]))
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=pixel_length[1])
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=pixel_length[1])
    kz3d, kx3d, ky3d = np.meshgrid(kz,kx, ky, indexing='ij') # not change the order as cube being same order
    kper3d = np.sqrt(kx3d ** 2 + ky3d ** 2)

# Flatten the arrays
    kper_flat = kper3d.flatten()
    klos_flat = kz3d.flatten()
    image_fft_flat = image_fft.flatten()
    
    klos_bin_i = np.digitize(klos_flat, bin_edges_klos)
    kper_bin_i = np.digitize(kper_flat, bin_edges_kper)
    nlos = len(bin_edges_klos)
    nper = len(bin_edges_kper)
    
    # fill pk at nearest bins if k is out of bound
    klos_bin_i = np.clip(klos_bin_i,1,nlos-1)
    kper_bin_i = np.clip(kper_bin_i,1,nper-1)

    image_fft_flat*= scale[klos_bin_i-1,kper_bin_i-1]
    del kper3d, kz3d, kx3d, ky3d, kper_flat, klos_flat, scale

    
    image = np.fft.ifftn(image_fft_flat.reshape(nz,nx,ny))/voxsize
    del image_fft, image_fft_flat
        
    return image.real



def calculate_2d_power_spectrum(cube, pixel_length, nbin=10, key='mean', bin_edges_klos=None, bin_edges_kper=None, apply_window=False):


    """
    Calculate the 2D cylindrical mean power spectrum from a 3D image spectral cube.

    Parameters
    ----------
    cube : numpy.ndarray
        3D image spectral cube.
    pixel_length[1]:  size_perp_cMpc/pixel : float
        Size of the cube in comoving megaparsecs perpendicular to the line of sight.
    length[0]: size_LOS_cMpc/pixel : float
        Size of the cube in comoving megaparsecs along the line of sight.
    nbin : int
        Number of perpendicular bins or line-of-sight bins. Must be the same for statistic_2d.
    apply_window : bool, optional
        Whether to apply a window function along the frequency axis (default is False).
    bin_edges_klos : numpy.ndarray or None, optional
        Bin edges for line-of-sight wavenumbers. If None, they will be computed.
    bin_edges_kper : numpy.ndarray or None, optional
        Bin edges for perpendicular wavenumbers. If None, they will be computed.

    Returns
    -------
    ps2d : numpy.ndarray
        2D cylindrical mean power spectrum. units = Mpc^3 [klos, kperp]
    bin_central_kper : numpy.ndarray
        Central bin values for perpendicular wavenumbers.
    bin_central_klos : numpy.ndarray
        Central bin values for line-of-sight wavenumbers.
    """

    # Create a copy of the cube for windowing if required
    if apply_window:
        cube_windowed = cube.copy()
        window = signal.windows.nuttall(cube_windowed.shape[0])
        window /= window.sum()
        cube_windowed = cube_windowed * window[:, np.newaxis, np.newaxis]
    else:
        cube_windowed = cube

    # Perform 3D FFT
    nz, nx, ny = cube_windowed.shape
    
    size_LOS_cMpc  = nz*pixel_length[0] 
    size_perp_cMpc = nx*pixel_length[1]

    
    volume = size_perp_cMpc ** 2 * size_LOS_cMpc
    voxsize = volume / (nx * ny * nz)

    cube_fft = np.fft.fftn(cube_windowed.astype('float64'))

    # Calculate power spectrum
    ps3d = np.abs(cube_fft) ** 2  # dimensionless
    ps3d *= voxsize ** 2 / volume  # Mpc^3

    # Calculate cylindrical wave numbers
    kz = np.abs(2 * np.pi * np.fft.fftfreq(nz, d= pixel_length[0]))
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=pixel_length[1])
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=pixel_length[1])
    
    
    
    kz3d, kx3d, ky3d = np.meshgrid(kz,kx, ky, indexing='ij') # not change the order as cube being same order
    kper3d = np.sqrt(kx3d ** 2 + ky3d ** 2)

    
    # Flatten the arrays
    kper_flat = kper3d.flatten()
    klos_flat = kz3d.flatten()
    ps3d_flat = ps3d.flatten()

    # Calculate the bin edges for binning statistics if not provided
    if bin_edges_klos is None:
        bin_edges_klos = np.linspace(1e-4, klos_flat.max(), nbin + 1)
    if bin_edges_kper is None:
        bin_edges_kper = np.linspace(1e-4, kper_flat.max(), nbin + 1)

    # Calculate binned statistics of the power spectrum
    ps2d,los_edge,per_edge,_ = stats.binned_statistic_2d(klos_flat,kper_flat, values=ps3d_flat, statistic=key,
                                                     bins=[bin_edges_klos,bin_edges_kper])
    
    del cube_windowed, cube_fft, ps3d, kz3d, kx3d, ky3d, kper3d, kper_flat, klos_flat, ps3d_flat


    return ps2d,los_edge,per_edge




def calculate_1d_power_spectrum(cube, pixel_length, nbin=10, key='mean', bin_edges=None, apply_window=False): # for 3d map


    # Create a copy of the cube for windowing if required
    if apply_window:
        cube_windowed = cube.copy()
        window = signal.windows.nuttall(cube_windowed.shape[0])
        window /= window.sum()
        cube_windowed = cube_windowed * window[:, np.newaxis, np.newaxis]
    else:
        cube_windowed = cube

    # Perform 3D FFT
    nz, nx, ny = cube_windowed.shape
    
    size_LOS_cMpc  = nz*pixel_length[0]
    size_perp_cMpc = nx*pixel_length[1]

    
    volume = size_perp_cMpc ** 2 * size_LOS_cMpc
    voxsize = volume / (nx * ny * nz)

    cube_fft = np.fft.fftn(cube_windowed.astype('float64'))

    # Calculate power spectrum
    ps3d = np.abs(cube_fft) ** 2  # dimensionless
    ps3d *= voxsize ** 2 / volume  # Mpc^3

    # Calculate cylindrical wave numbers
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=pixel_length[0])
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=pixel_length[1])
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=pixel_length[1])

    
    kz3d, kx3d, ky3d = np.meshgrid(kz,kx, ky, indexing='ij') # not change the order as cube being same order
    k3d = np.sqrt(kz3d**2 + kx3d ** 2 + ky3d ** 2)
    
    # Flatten the arrays
    k_flat = k3d.flatten()
    ps3d_flat = ps3d.flatten()

    # Calculate the bin edges for binning statistics if not provided
    if bin_edges is None:
        bin_edges = np.linspace(1e-4, k_flat.max(), nbin + 1)

    # Calculate binned statistics of the power spectrum
    ps1d,k_edge,_ = stats.binned_statistic(k_flat, values=ps3d_flat, statistic=key,
                                                   bins=bin_edges)

    del cube_windowed,  cube_fft, ps3d, kz3d, kx3d, ky3d, k3d, k_flat, ps3d_flat

    return ps1d,k_edge







def calculate_image_1dpower_spectrum(image, pixel_length, nbin=10, key='mean', bin_edges=None): # for 2d map 
    # Perform 2D FFT
    fft_image = np.fft.fft2(image.astype('float64'))

    # Calculate image dimensions and area
    nx, ny = image.shape
    size_x_cMpc = nx * pixel_length
    size_y_cMpc = ny * pixel_length
    area = size_y_cMpc * size_x_cMpc
    pixelsize = pixel_length**2

   # Calculate power spectrum
    ps2d = np.abs(fft_image) ** 2  # dimensionless
    ps2d *= pixelsize ** 2 / area  # Mpc^2


    # Calculate wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=pixel_length)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=pixel_length)

    kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')
    k2d = np.sqrt(kx2d ** 2 + ky2d ** 2)

    # Flatten the arrays
    k_flat = k2d.flatten()
    ps2d_flat = ps2d.flatten()

    # Calculate the bin edges for binning statistics if not provided
    if bin_edges is None:
        bin_edges = np.linspace(1e-4, k_flat.max(), nbin + 1)

    # Calculate binned statistics of the power spectrum
    ps1d, k_edge, _ = stats.binned_statistic(k_flat, values=ps2d_flat, statistic=key, bins=bin_edges)
    
    del fft_image, ps2d, kx2d, ky2d, k2d, k_flat, ps2d_flat

    return ps1d, k_edge


def fit_2d_gaussian(img):
    def generalGauss2d(xy, x0, y0, sigma_x, sigma_y, theta=0, amp=1, offset=0):
        norm = amp
        x, y = xy

        rtheta = np.deg2rad(theta)

        cos2 = np.cos(rtheta) ** 2
        sin2 = np.sin(rtheta) ** 2
        cos_sin = np.sin(2 * rtheta)

        a = (np.cos(rtheta)**2.)/(2.*(sigma_x**2.)) + (np.sin(rtheta)**2.)/(2.*(sigma_y**2.))
        b = -1.*(np.sin(2.*rtheta))/(4.*(sigma_x**2.)) + (np.sin(2.*rtheta))/(4.*(sigma_y**2.))
        c = (np.sin(rtheta)**2.)/(2.*(sigma_x**2.)) + (np.cos(rtheta)**2.)/(2.*(sigma_y**2.))

        g = offset + norm * np.exp(-1.0 * (a * ((x - x0) ** 2) - 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))

        return g.ravel()

    def get_BMAJwidth_BPA(img):
        nx, ny = img.shape
        x = np.arange(nx)
        y = np.arange(ny)
        x, y = np.meshgrid(x, y)

        initial_guess = (nx / 2, ny / 2, 10, 10, 0)
        bounds = ([nx * 0.2, ny * 0.2, 1, 1, -180], [nx * 0.8, ny * 0.95, nx / 2, ny / 2, 180])

        popt, _ = curve_fit(generalGauss2d, (x, y), img.ravel(), p0=initial_guess, bounds=bounds)

        x0, y0, sigma_x, sigma_y, BPA  = popt
        BMAJ = sigma_x * 2
        BMIN = sigma_y * 2

        return BMAJ, BMIN, BPA

    return get_BMAJwidth_BPA(img)

def generate_2dimage_from_ps1d(ps1d, shape, pixel_length, bin_edges_kper, seed=None):

    nx=shape[0]
    ny=shape[1]
    size_x_cMpc = nx * pixel_length
    size_y_cMpc = ny * pixel_length
    area = size_y_cMpc * size_x_cMpc
    pixelsize = pixel_length**2

    image = np.random.normal(loc=0., scale=1./np.sqrt(nx*ny), size=shape)
    # Perform 3D FFT, make std(\delta k)= 1
    image_fft = np.fft.fft2(image.astype('float64'))
    scale =  np.sqrt(area*ps1d) 

     # Calculate wave numbers 
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=pixel_length)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=pixel_length)
        
    kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')
    kper2d = np.sqrt(kx2d ** 2 + ky2d ** 2)

    # Flatten the arrays
    kper_flat = kper2d.flatten()
    image_fft_flat = image_fft.flatten()
    kper_bin_i = np.digitize(kper_flat, bin_edges_kper)
    nper = len(bin_edges_kper)
    
    # fill pk at nearest bins if k is out of bound
    kper_bin_i = np.clip(kper_bin_i,1,nper-1)

    image_fft_flat*= scale[kper_bin_i-1]
    
    image = np.fft.ifft2(image_fft_flat.reshape(nx,ny))/pixelsize
    del image_fft, image_fft_flat,kper_flat, kper2d, kx2d, ky2d, kper_bin_i
        
    return image.real
