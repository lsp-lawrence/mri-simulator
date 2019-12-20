import numpy as np
import matplotlib.pyplot as plt

fig_name = 'mr_b_free_water.pdf'

# Extract data
run_no = 0
dw_data = np.loadtxt('dw_data_' + str(run_no) + '.csv',delimiter=',')
b_vals = dw_data[:,0]
num_b_vals = len(b_vals)
num_runs = 7
S = np.empty([num_runs,num_b_vals])

# Compute mean
for run_no in range(num_runs):
    dw_data = np.loadtxt('dw_data_' + str(run_no) + '.csv',delimiter=',')
    S[run_no,:] = dw_data[:,1]/dw_data[0,1]
S_mean = np.mean(S,axis=0)
S_std = np.std(S,axis=0)

plt.plot(b_vals*1e-6,S_mean)
plt.show()

### Do adc fit
##p = np.polyfit(b_vals,np.log(S_mean),deg=1)
##S_fit = np.exp(b_vals*p[1])
##
### Plot
##plt.figure()
##plt.errorbar(b_vals*1e-6,S_mean,S_std,label='data')
##plt.plot(b_vals*1e-6,S_fit,label='fit')
##plt.title('mr signal amplitude versus b-value')
##plt.xlabel('b value (mm^2/s)')
##plt.ylabel('amplitude (a.u.)')
##plt.savefig(fig_name)
##plt.show()
