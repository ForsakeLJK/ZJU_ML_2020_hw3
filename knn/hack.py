import numpy as np

import knn
import show_image
from extract_image import extract_image

def hack(img_name):
    '''
    hack Recognize a CAPTCHA image
      Inputs:
          img_name: filename of image
      Outputs:
          digits: 4 digits in the input CAPTCHA image, shape(4, ).
    '''
    # hack_data.npz contains 100 images with labels, 
    # i.e., 400 digits with labels
    data = np.load('hack_data.npz')

    # YOUR CODE HERE (you can delete the following code as you wish)
    x_train = data['x_train']
    y_train = data['y_train']

    # begin answer
    N = x_train.shape[0]
    # square of N, square(400) in this case
    k = 20
    # test matrix, 4-by-144
    x_test = extract_image(img_name)

    digits = knn.knn(x_test, x_train, y_train, k)
    # end answer

    return digits
