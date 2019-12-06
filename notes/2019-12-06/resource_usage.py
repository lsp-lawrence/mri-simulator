import numpy as np
import matplotlib.pyplot as plt

num_ems = np.array([1000,2000,5000,10000]) # number ems
max_memory_usage = np.array([93216,95324,99832,109632]) # kilobytes


# Plot
q = np.polyfit(num_ems,max_memory_usage,deg=1)
max_memory_usage_fit = q[0]*num_ems+q[1]
plt.plot(num_ems,max_memory_usage*1e-3,'o')
plt.plot(num_ems,max_memory_usage_fit*1e-3,'--')
plt.xlabel('Number of ems')
plt.ylabel('Memory usage (MB)')
plt.tight_layout()
print(str(q[0]) + ' kB per em')
plt.savefig('supershop_resource_usage.pdf')
