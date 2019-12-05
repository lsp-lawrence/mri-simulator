from pysim import Sim
from pyem import Em
import matplotlib.pyplot as plt
import numpy as np

save_fig = True
fig_name = 'density.pdf'

# Spread ems
max_ems_grid_point = 100
grid_dim = 10
em_positions = np.empty([0,3])
xlim = 5e-2
ylim = xlim
zlim = xlim
sigma = xlim/1.5
x_vals = np.linspace(-xlim,xlim,grid_dim)
y_vals = np.linspace(-ylim,ylim,grid_dim)
z_vals = np.linspace(-zlim,zlim,grid_dim)
rho = np.empty([grid_dim,grid_dim])
for i,x in zip(range(grid_dim),x_vals):
    for j,y in zip(range(grid_dim),y_vals):
        num_ems_here = int(np.exp(-(x**2+y**2)/(2*sigma**2))*max_ems_grid_point)
        positions = np.tile(np.array([[x,y,0.0]]),(num_ems_here,1))
        em_positions = np.concatenate((em_positions,positions))
        rho[i,j] = num_ems_here

plt.imshow(rho,extent=[x_vals[0]*1e2,x_vals[-1]*1e2,y_vals[0]*1e2,y_vals[-1]*1e2])
plt.colorbar(label='number of ems')
plt.xlabel('X position (cm)')
plt.ylabel('Y position (cm)')
if save_fig:
    plt.savefig(fig_name)
else:
    plt.show()
