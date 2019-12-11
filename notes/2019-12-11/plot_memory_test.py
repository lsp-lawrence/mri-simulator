import numpy as np
import matplotlib.pyplot as plt

# Memory usage as a function of number of ems with TR = 100 ms, Macbook
num_ems = np.array([100,1000,3000,5000,8000,10000]) # number of ems
mem = np.array([0.031,0.906,3.07,5.184,8.48,10.586]) # memory usage in MiB
mem_ss = np.array([0.039,0.644,3.27,5.77,9.38,11.81]) # memory usage in MiB
p = np.polyfit(num_ems,mem,1)
p_ss = np.polyfit(num_ems,mem_ss,1)
print("mem per em: " + str(p[0]*(2**20)/(1e6)*1e3) + ' kB')
print("mem_ss per em: " + str(p_ss[0]*(2**20)/(1e6)*1e3) + ' kB')
plt.figure()
plt.plot(num_ems,mem,'or',label='data macbook')
plt.plot(num_ems,p[0]*num_ems+p[1],'--r',label='fit macbook')
plt.plot(num_ems,mem_ss,'ob',label='data supershop')
plt.plot(num_ems,p_ss[0]*num_ems+p[1],'--b',label='fit supershop')
plt.legend()
plt.title('memory usage')
plt.xlabel('number of ems')
plt.ylabel('memory (MiB)')
plt.savefig('mem_num_ems.pdf')
plt.show()


