import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,  img_to_array
from skimage import io
import os
from skimage.feature import canny



img = io.imread('data/potato_src/1473498368.jpg', as_grey=True)  # this is a Numpy array with shape (3, 150, 150)
# Compute the Canny filter for two values of sigma
edges = canny(img, sigma=3)

# display results
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

ax1.imshow(img, cmap=plt.cm.jet)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges2, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()