import numpy as np
import matplotlib.pyplot as plt
from pyreconstruct import shift_2DFT

"""This script tests the matrix reordering function shift_2DFT.
- We assume k-space is sampled with ky = [ky_max:-delta_ky:-ky_max] and kx = [-kx_max:delta_kx:kx_max] to populate the matrix S_acquired.
- S_acquired is reordered using shift_2DFT to S_dft.
- One may call IFFT directly on S_dft to recover the image.
"""

save_fig = True
fig_name = 'shift_2DFT_test.pdf'

N = 11
S_acquired = np.zeros([N,N])
S_acquired[5,5] = N
S_acquired[5,4] = N
S_acquired[5,6] = N
S_acquired[4,5] = N
S_acquired[6,5] = N
S_dft = shift_2DFT(S_acquired)
img = np.real(np.fft.ifft2(S_dft))
img_theory = np.zeros([N,N])
for m in range(N):
    for n in range(N):
        img_theory[m,n] = 1 + np.cos(2*np.pi/N*m) + np.cos(2*np.pi/N*n)

plt.subplot(221)
plt.imshow(S_acquired)
plt.title('S_acquired')
plt.xlabel('kx')
plt.ylabel('ky')
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.subplot(222)
plt.imshow(S_dft)
plt.title('S_dft')
plt.subplot(223)
plt.imshow(img)
plt.title('image from S_dft')
plt.subplot(224)
plt.imshow(img_theory)
plt.title('image from theory')
plt.tight_layout()
if save_fig:
    plt.savefig(fig_name)
else:
    plt.show()
