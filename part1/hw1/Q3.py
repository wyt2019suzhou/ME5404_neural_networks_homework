import numpy as np
from matplotlib import pyplot as plt

# w=np.random.random((3))
w=np.random.random((2))
# x=np.array([[1,1,1,1],[0,1,0,1],[0,0,1,1]])
x=np.array([[1,1],[0,1]])
v=np.dot(w,x)
y=np.array([0,0])
for index,item in enumerate(v):
    if item>0:
        y[index]=1
    else:
        y[index]=0
d=np.array([1,0])
e=d-y
rate=0.01
errors=[]
rollout=[w]
while any(y!=d):
    e=d-y
    error = np.mean(np.abs(e))
    errors.append(error)
    w=w+rate*np.dot(x,e)
    rollout.append(w)
    v = np.dot(w, x)
    for index, item in enumerate(v):
        if item > 0:
            y[index] = 1
        else:
            y[index] = 0

errors.append(0)

fig1=plt.figure(figsize=(10,10))
plt.plot(list(range(len(errors))),errors)
plt.title("Error change graph ")
plt.xlabel('Number of iterations')
plt.ylabel("Error")
plt.show()


x2 = np.linspace(0,1,10)
y2 = w[0]/-w[2]+(w[1]/-w[2])*x2
fig2=plt.figure(figsize=(10,10))
plt.xlabel('x1')
plt.ylabel("x2")
plt.scatter(x[1,-1], x[2,-1], marker = 'x',color = 'red', s = 40 ,label = 'class 0')
plt.scatter(x[1,:-1], x[2,:-1], marker = 'o',color = 'green', s = 40 ,label = 'class 1')
plt.plot(x2,y2,label='decision boundary')
plt.legend(loc = 'best')
plt.show()

fig3=plt.figure(figsize=(10,10))
rollout=np.asarray(rollout)
plt.plot(list(range(len(rollout))),rollout[:,0],label='w0',color = 'red')
plt.plot(list(range(len(rollout))),rollout[:,1],label='w1',color = 'green')
# plt.plot(list(range(len(rollout))),rollout[:,2],label='w2',color = 'yellow')
plt.xlabel('iteration time')
plt.ylabel('weight')
plt.title('trajectory of weight' )
plt.legend(loc = 'best')
plt.show()

print('yes')
