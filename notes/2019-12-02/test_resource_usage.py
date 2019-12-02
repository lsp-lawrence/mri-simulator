import resource

N = 1000000
mylist = [i for i in range(N)]
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
