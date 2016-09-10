from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import io
import os

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


def augument_dir(src_path, dist_path):
    imgs_paths = os.listdir(src_path)
    imgs_paths = [os.path.join(src_path, y) for y in imgs_paths]
    for img_path in imgs_paths:
        img = io.imread(img_path, as_grey=False)  # this is a Numpy array with shape (3, 150, 150)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=dist_path, save_prefix='potato',
                                  save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

augument_dir('data/potato_src', 'data/potato')
augument_dir('data/not_potato_src', 'data/not_potato')

