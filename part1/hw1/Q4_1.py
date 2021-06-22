import numpy as np
from matplotlib import pyplot as plt

x=np.array([[1,0.5],[1,1.5],[1,3],[1,4.0],[1,5.0]])
d=np.array([8.0,6.0,5,2,0.5])
w=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.transpose(x)),d)
y=np.dot(x,w)
e=d-y
error = np.mean(np.abs(e))

x2 = np.linspace(0,5,100)
y2 = w[0]+w[1]*x2

fig2=plt.figure(figsize=(10,10))
plt.xlabel('x')
plt.ylabel("y")

plt.scatter(x[:,1], d, marker = 'x',color = 'red', s = 40)
plt.plot(x2,y2)
plt.title('LLS')
plt.show()
print('yes')