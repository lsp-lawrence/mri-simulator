import numpy as np

def reconstruct_from_2DFT(S,kx,ky):
    """Reconstructs an MR image from a 2DFT pulse sequence
    Params:
        S: (2D numpy array of floats) S_{ij} = M[kx[i],ky[j]], where M[kx,ky] = Fourier transform of magnetization m[x,y]
        kx: (1D numpy array of floats) vector of samples in kx direction
        ky: (1D numpy array of floats) vector of samples in ky direction
    Returns:
        img: (2D numpy array of floats) MR image
    """
    I = np.fft.fft2(S)
    return I
    
def shift_2DFT(A):
    """Shifts a 2DFT matrix ordered with negative frequency convention to a 2DFT matrix ordered with IFFT convention
    Params:
        A: (2D numpy array) 2DFT matrix with negative frequency ordering
    Returns:
        Ap: (2D numpy array) 2DFT matrix with IFFT frequency ordering
    Note:
        A must have an add number of rows and columns
    """
    # Check params
    if not (A.ndim == 2):
        raise TypeError("A must be a 2D numpy array")
    num_rows = A.shape[0]
    num_cols = A.shape[1]
    if not (np.mod(num_rows,2) != 0 and np.mod(num_cols,2) != 0):
        raise ValueError("A must have an odd number of rows and columns")
    # Reorder
    Ap = np.empty([num_rows,num_cols])
    ry = int(num_rows/2)
    rx = int(num_cols/2)
    Ap[0:ry,0:(rx+1)] = A[(ry+1):,rx:]
    Ap[ry:,0:(rx+1)] = A[0:(ry+1),rx:]
    Ap[0:ry,(rx+1):] = A[(ry+1):,0:rx]
    Ap[ry:,(rx+1):] = A[0:(ry+1),0:rx]
    return Ap
    
