import numpy as np 


def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

np.random.seed(0)

train_input = np.array([ [0,0,0,0,1],
                         [0,0,0,1,0],
                         [0,0,0,1,1],
                         [0,0,1,0,0],
                         [0,0,1,0,1],
                         [0,0,1,1,0],
                         [0,0,1,1,1],
                         [0,1,0,0,0],
                         [0,1,0,0,1],
                         [0,1,0,1,0],
                         [0,1,0,1,1],
                         [1,0,0,0,0],
                         [1,0,0,0,1],
                         [1,0,0,1,0],
                         [1,0,0,1,1],
                         [1,0,1,0,0],
                         [1,0,1,0,1],
                         [1,0,1,1,0],
                         [1,1,0,0,0],
                         [1,1,0,0,1],
                         [1,1,0,1,0],
                         [1,1,0,1,1],
                         [1,1,1,0,0],
                         [1,1,1,0,1],
                         [1,1,1,1,0],
                         [1,1,1,1,1]])

train_output = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
                          0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 
                          1, 0, 1, 0, 1, 0]]).T

input_data = np.array([[0, 1, 0, 1, 1]])



weights0 = 2*np.random.random((5,1)) - 1

for i in range(50):
  l0 = train_input
  l1 = nonlin(np.dot(l0,weights0))

  l1_error = train_output - l1
  l1_delta = l1_error * nonlin(l1,True)

  weights0 += np.dot(l0.T,l1_delta)


print(nonlin(np.dot(input_data,weights0)))
