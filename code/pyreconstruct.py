def reconstruct(S,kx,ky):
    """Reconstructs an MR image
    Params:
        S: (2D numpy array of floats) S_{ij} = M[kx[i],ky[j]], where M[kx,ky] = Fourier transform of magnetization m[x,y]
        kx: (1D numpy array of floats) vector of samples in kx direction
        ky: (1D numpy array of floats) vector of samples in ky direction
    Returns:
        img: (2D numpy array of floats) MR image
    """
    
