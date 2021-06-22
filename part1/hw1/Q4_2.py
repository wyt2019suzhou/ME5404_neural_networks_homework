import numpy as np
from matplotlib import pyplot as plt

x=np.array([[1,0.5],[1,1.5],[1,3],[1,4.0],[1,5.0]])
d=np.array([8.0,6.0,5,2,0.5])
w=np.random.random((2))
rate=0.02
rollout=[w]
errors=[]

for i in range(100):
    e=d-np.dot(x,w)
    error=np.mean(np.abs(e))
    errors.append(error)
    w=w+rate*np.dot(e,x)
    rollout.append(w)


rollout=np.asarray(rollout)
fig=plt.figure()
plt.plot(list(range(len(rollout))),rollout[:,0],label='b',color='red')
plt.plot(list(range(len(rollout))),rollout[:,1],label='w',color='green')
plt.xlabel('iteration time')
plt.ylabel('weight')
plt.title('trajectory of weight' )
plt.legend(loc = 'best')
plt.show()

x2 = np.linspace(0,5,100)
y2 = w[0]+w[1]*x2
fig2=plt.figure(figsize=(10,10))
plt.xlabel('x')
plt.ylabel("y")
plt.scatter(x[:,1], d, marker = 'x',color = 'red', s = 40)
plt.plot(x2,y2)
plt.show()

fig3=plt.figure()
plt.plot(list(range(len(errors))),errors)
plt.title("Error change graph ")
plt.xlabel('Number of iterations')
plt.ylabel("Error")
plt.show()

print('yes')