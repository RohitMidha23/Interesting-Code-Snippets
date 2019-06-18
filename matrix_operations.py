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

def getMatrixMinor(m,r,c):
  a = np.zeros([len(m)-1,len(m)-1])
  p=0;q=0;
  for i in range(len(m)):
    for j in range(len(m)):
      if r!=i and c!=j :
        a[p,q] = m[i,j]
        if j!=c :
          q = q+1
        if len(m)-1 == q :
          q=0
    if i!=r: p=p+1
    if len(m)-1 == p:
      p=p-1

  return a
