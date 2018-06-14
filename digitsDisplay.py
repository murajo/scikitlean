from sklearn import datasets
from matplotlib import pyplot as plt

digits = datasets.load_digits()

# number 0
plt.subplot(141), plt.imshow(digits.images[0], cmap = 'gray')
plt.title('number 0'), plt.xticks([]), plt.yticks([])

# number 1
plt.subplot(142), plt.imshow(digits.images[1], cmap = 'gray')
plt.title('number 1'), plt.xticks([]), plt.yticks([])

# number 2
plt.subplot(143), plt.imshow(digits.images[2], cmap = 'gray')
plt.title('number 2'), plt.xticks([]), plt.yticks([])

# number 9
plt.subplot(144), plt.imshow(digits.images[-2], cmap = 'gray')
plt.title('number 9'), plt.xticks([]), plt.yticks([])

plt.show()
