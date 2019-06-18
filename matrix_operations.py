import numpy as np

def maxmin(m):
  a = np.zeros([len(m),2])
  b = [np.zeros(2),1]
  for i in range(len(m)):
    a[i,0] = max(m[i,])
    a[i,1] = min(m[i,])
  b[0] = max(a[:,0])
  b[1] = min(a[:,1])
  return b
