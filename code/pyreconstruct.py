import numpy as np

def reconstruct_from_2DFT(S,kx_max,ky_max):
    """Reconstructs an MR image from a 2DFT pulse sequence
    Params:
        S: (2D numpy array of floats) S_{ij} = M[kx[i],ky[j]], where M[kx,ky] = Fourier transform of magnetization m[x,y]
    Returns:
        img: (2D numpy array of floats) MR image
    """
    # Check params
    if not(S.ndim == 2 and S.dtype == np.float64):
        raise TypeError("S must be a 2D numpy array of floats")
    # Do the reconstruction
    S_dft = shift_2DFT(S)
    img_dft = np.abs(np.fft.ifft2(S_dft))
    img = ishift_2DFT(img_dft)
    num_rows = S.shape[0]
    num_cols = S.shape[1]
    x = np.arange(num_cols)-int(num_cols/2)
    x = x/(2*kx_max)
    y = np.arange(num_rows)-int(num_rows/2)
    y = y/(2*ky_max)
    return img,x,y
    
def shift_2DFT(A):
    """Shifts a 2DFT matrix ordered with negative frequency convention to a 2DFT matrix ordered with IFFT convention
    Params:
        A: (2D numpy array) 2DFT matrix with negative frequency ordering
    Returns:
        Ap: (2D numpy array) 2DFT matrix with IFFT frequency ordering
    Note:
        A must have an odd number of rows and columns
    """
    # Check params
    if not (A.ndim == 2):
        raise TypeError("A must be a 2D numpy array")
    num_rows = A.shape[0]
    num_cols = A.shape[1]
    if not (np.mod(num_rows,2) != 0 and np.mod(num_cols,2) != 0):
        raise ValueError("A must have an odd number of rows and columns")
    # Reorder
    Ap = np.empty([num_rows,num_cols],dtype=A.dtype)
    ry = int(num_rows/2)
    rx = int(num_cols/2)
    Ap[0:ry,0:(rx+1)] = A[(ry+1):,rx:]
    Ap[ry:,0:(rx+1)] = A[0:(ry+1),rx:]
    Ap[0:ry,(rx+1):] = A[(ry+1):,0:rx]
    Ap[ry:,(rx+1):] = A[0:(ry+1),0:rx]
    Ap = np.flipud(Ap)
    return Ap

def ishift_2DFT(Ap):
    """Inverse of shift_2DFT
    Params:
        Ap: (2D numpy array) 2DFT matrix with IFFT ordering
    Returns:
        A: (2D numpy array) 2DFT matrix with negative frequency ordering
    Note:
        Ap must have an odd number of rows and columns
    """
    # Check params
    if not (Ap.ndim == 2):
        raise TypeError("Ap must be a 2D numpy array")
    num_rows = Ap.shape[0]
    num_cols = Ap.shape[1]
    if not (np.mod(num_rows,2) != 0 and np.mod(num_cols,2) != 0):
        raise ValueError("Ap must have an odd number of rows and columns")
    # Reorder
    A = np.empty([num_rows,num_cols],dtype=Ap.dtype)
    ry = int(num_rows/2)
    rx = int(num_cols/2)
    Ap = np.flipud(Ap)
    A[(ry+1):,rx:] = Ap[0:ry,0:(rx+1)]
    A[0:(ry+1),rx:] = Ap[ry:,0:(rx+1)]
    A[(ry+1):,0:rx] = Ap[0:ry,(rx+1):]
    A[0:(ry+1),0:rx] = Ap[ry:,(rx+1):]
    return A

def adjust_phase(S):
    """Adjusts the phase of each row of an MR signal matrix acquired with a 2DFT sequence under the assumption of a real image to account for arbitrary phase associated with each line
    Params:
        S: (2D numpy array of complex): MR signal matrix from 2DFT acquisition
    Returns:
        S_phase_adjusted: (2D numpy array of complex): phase-adjusted MR signal matrix
    Note:
        Assumes S has an odd number of rows and columns
    """
    # Check params
    if not (S.ndim == 2):
        raise TypeError("S must be a 2D numpy array")
    num_rows = S.shape[0]
    num_cols = S.shape[1]
    if not (np.mod(num_rows,2) != 0 and np.mod(num_cols,2) != 0):
        raise ValueError("S must have an odd number of rows and columns")
    # Adjust phase from theory
    num_rows = S.shape[0]
    S_phase_adjusted = np.empty(S.shape,dtype=complex)
    for row_no in range(int(num_rows/2)):
        S_uv = S[row_no,0]
        S_minus_uv = S[-(row_no+1),-1]
        phi = np.angle(S_minus_uv)
        phi_theory = -np.angle(S_uv)
        S_phase_adjusted[row_no,:] = S[row_no,:]
        S_phase_adjusted[-(row_no+1),:] = S[-(row_no+1),:]*np.exp(1j*(phi_theory-phi))
    row_no = int(num_rows/2)
    for col_no in range(int(num_cols/2)):
        S_uv = S[row_no,col_no]
        S_minus_uv = S[row_no,-(col_no+1)]
        phi = np.angle(S_minus_uv)
        phi_theory = -np.angle(S_uv)
        S_phase_adjusted[row_no,col_no] = S[row_no,col_no]
        S_phase_adjusted[row_no,-(col_no+1)] = S[row_no,-(col_no+1)]*np.exp(1j*(phi_theory-phi))
    col_no = int(num_cols/2)
    S_phase_adjusted[row_no,col_no] = S[row_no,col_no]*np.exp(-1j*np.angle(S[row_no,col_no]))
    return S_phase_adjusted
        
