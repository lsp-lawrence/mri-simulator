import matplotlib.pyplot as plt
import numpy as np
num_ems = np.array([1e2,1e3,1e4,1e5],dtype=float)
usage = np.array([41238528,41828352,47497216,108830720])
usage_nm = np.array([41238528,41775104,47521792,108847104])

p = np.polyfit(num_ems,usage,1)
usage_fit = p[0]*num_ems+p[1]
p_nm = np.polyfit(num_ems,usage_nm,1)
usage_nm_fit = p_nm[0]*num_ems+p_nm[1]

plt.plot(num_ems,usage*1e-6,'bo',label='with methods')
plt.plot(num_ems,usage_fit*1e-6,'b--')
plt.plot(num_ems,usage_nm*1e-6,'ro',label='without methods')
plt.plot(num_ems,usage_nm*1e-6,'r--')
plt.xlabel('Number of Em objects')
plt.ylabel('Memory Usage (MB)')
plt.legend()
print("Usage per em with methods: " + str(p[0]*1e-3) + " kB")
print("Usage per em without methods: " + str(p_nm[0]*1e-3) + "kB")
plt.savefig("em_usage.pdf")
