
import keras
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
X_train.shape
to_p = X_train[:9]

f, axarr = plt.subplots(2,4)
axarr[0,0].imshow(to_p[0], cmap='gray')
axarr[0,1].imshow(to_p[1], cmap='gray')
axarr[0,2].imshow(to_p[2], cmap='gray')
axarr[0,3].imshow(to_p[3], cmap='gray')
axarr[1,0].imshow(to_p[4], cmap='gray')
axarr[1,1].imshow(to_p[5], cmap='gray')
axarr[1,2].imshow(to_p[6], cmap='gray')
axarr[1,3].imshow(to_p[7], cmap='gray')
plt.show()
