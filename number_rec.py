import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()


n_samples = len(digits.images)
print(f"Number of samples in the data set is: {str(n_samples)}")

X = digits.images.reshape((n_samples,-1))
print(f"Shape of input matrix x is: {str(x.shape)}")

y = digits.target
print(f"Shape of target vector y is: {str(y.shape)}")


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)



images_and_labels = list(zip(digits.images, digits.target))

plt.figure(figsize=(5,5))
for index, (image,label) in enumerate(images_and_labels[:15]):
    plt.subplot(3,5,index+1)
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"{label}")

plt.show()
# pl.gray()
# pl.matshow(digits.images[0])
# pl.show()


print(digits.images[0])