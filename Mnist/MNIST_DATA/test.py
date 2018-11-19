import numpy as np

a = np.array([1,0,2,4,52,4,51])
b = np.array([True,False,True,True])
c = np.array([1,0,0,1,0,1,1,0,0,1])
d = np.array([1,0,0,1,0,1,1,0,0,1])
e = np.equal(np.argmax(c),np.argmax(d))
print(np.argmax(c))

print(np.argmax(d))

print(np.argmax(e))


