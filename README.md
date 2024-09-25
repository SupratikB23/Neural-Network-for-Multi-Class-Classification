# Custom-Activation-Function

This code is an ANN (Artificial Neural Network) made with 2 hidden layers and a custom Activation Function (AF).

Ada-Act Function:<br/>
g(x) = k0 + k1*x

The ANN is made from scratch using NumPy and Pandas libraries only which is tested with the MNIST dataset of more than 45,000 images and gaining an accuracy of 86% & F1-score of 0.857.

Forward Propagation Equations:<br/>
z1 = a0 x w1 + b1<br/>
a1 = g(z1)<br/>
z2 = a1 x w2 + b2<br/>
a2 = g(z2)<br/>
z3 = a2 × w3 + b3 <br/>
a3 = Softmax(z3)

Back Propagation Equations:<br/>
dz3 = z3 - y<br/>
dw3 = (1/t) a2^T x dz3<br/>
db3 = avgcol(dz3)<br/>
da2 = dz3 × wT <br/>
dz2 = g′(z2) ∗ da2<br/>
dw2 = (1/t) a1^T x dz2<br/>
db2 = avgcol(dz2)<br/>
dK2 = avge(da2 ∗ z2)<br/>
da1 = dz2 x w^T<br/>
dz1 = g′(z1) ∗ da1<br/>
dw1 = (1/t) a0^T x dz1<br/>
db1 = avgcol(dz1)<br/>
dK1 = avge(da1 ∗ z1)<br/>
dK = dK2 + dK1

Parameter Updation:<br/>
w1 = w1 − α.dw1 <br/>
b1 = b1 − α.db1 <br/>
w2 = w2 − α.dw2 <br/>
b2 = b2 − α.db2 <br/>
w3 = w3 − α.dw3 <br/>
b3 = b3 − α.db3 <br/>
K = K − α.dK
