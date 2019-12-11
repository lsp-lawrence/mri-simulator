import numpy as np
import matplotlib.pyplot as plt

### Memory usage as a function of number of ems with TR = 100 ms
##num_ems = np.array([100,1000,5000,10000]) # number of ems
##mem = np.array([0.031,0.906,5.184,10.586]) # memory usage in MiB
##p = np.polyfit(num_ems,mem,1)
##print("mem per em: " + str(p[0]*(2**20)/(1e6)*1e3) + ' kB')
##plt.figure()
##plt.plot(num_ems,mem,'o',label='data')
##plt.plot(num_ems,p[0]*num_ems+p[1],'--',label='fit')
##plt.legend()
##plt.title('memory usage')
##plt.xlabel('number of ems')
##plt.ylabel('memory (MiB)')
##plt.savefig('mem_num_ems.pdf')
##plt.show()

# Memory usage as a function of pulse sequence length with num_ems = 10
ps_len = np.array([1,2,3,4]) # mil
ps_mem = np.array([0.957,1.91,2.86,3.82]) # memory to instantiate Pulse
sim_mem = np.array([0.105,0.0976,0.0937,0.129]) # memory to instantiate Sim
p = np.polyfit(ps_len,ps_mem,1)
plt.figure()
plt.plot(ps_len,ps_mem,'o',label='data')
plt.plot(ps_len,p[0]*ps_mem+p[1],'--',label='fit')
plt.legend()
plt.title('memory usage')
plt.xlabel('pulse length')
plt.ylabel('memory (MiB)')
plt.savefig('mem_ps.pdf')
plt.show()
