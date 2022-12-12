import idx2numpy
import scipy.ndimage

def load_mnist_data(mnist_data_path):
    '''
    given a path to the location of the decompressed MNIST data with the default naming conventions,
    return a tuple of the following numpy arrays:
    (train_img_arr, train_label_arr, test_img_arr, test_label_arr)
    '''
    train_img_file = f'{mnist_data_path}/train-images.idx3-ubyte'
    train_label_file = f'{mnist_data_path}/train-labels.idx1-ubyte'
    test_img_file = f'{mnist_data_path}/t10k-images.idx3-ubyte'
    test_label_file = f'{mnist_data_path}/t10k-labels.idx1-ubyte'

    train_img_arr = idx2numpy.convert_from_file(train_img_file) # shape = (60000, 28, 28)
    train_label_arr = idx2numpy.convert_from_file(train_label_file) # shape = (60000,)
    test_img_arr = idx2numpy.convert_from_file(test_img_file) # shape = (10000, 28, 28)
    test_label_arr = idx2numpy.convert_from_file(test_label_file) # shape = (10000,)

    return (train_img_arr, train_label_arr, test_img_arr, test_label_arr)


def blur_images(imgs, img_d1, img_d2, radius):
    blurred_imgs = scipy.ndimage.uniform_filter(imgs.reshape(-1, img_d1, img_d2), size=(0, radius, radius))
    return blurred_imgs.reshape(-1, img_d1*img_d2)
