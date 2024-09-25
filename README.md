# Custom-Activation-Function

This code is an ANN (Artificial Neural Network) made with 2 hidden layers and a custom Activation Function (AF).

Ada-Act Function:< br / >
g(x) = k0 + k1*x

The ANN is made from scratch using NumPy and Pandas libraries only which is tested with the MNIST dataset of more than 45,000 images and gaining an accuracy of 86% & F1-score of 0.857.

Forward Propagation Equations:
z1 = a0 x w1 + b1
a1 = g(z1)
z2 = a1 x w2 + b2
a2 = g(z2)
z3 = a2 × w3 + b3 
a3 = Softmax(z3)

Back Propagation Equations:
dz3 = z3 - y
dw3 = (1/t) a2^T x dz3
db3 = avgcol(dz3)
da2 = dz3 × wT 
dz2 = g′(z2) ∗ da2
dw2 = (1/t) a1^T x dz2
db2 = avgcol(dz2)
dK2 = avge(da2 ∗ z2)
da1 = dz2 x w^T
dz1 = g′(z1) ∗ da1
dw1 = (1/t) a0^T x dz1
db1 = avgcol(dz1)
dK1 = avge(da1 ∗ z1)
dK = dK2 + dK1

Parameter Updation:
w1 = w1 − α.dw1 
b1 = b1 − α.db1 
w2 = w2 − α.dw2 
b2 = b2 − α.db2 
w3 = w3 − α.dw3 
b3 = b3 − α.db3 
K = K − α.dK
