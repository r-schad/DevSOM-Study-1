import numpy as np
import import_ipynb
from load_MNIST_data import load_mnist_data
import math, time
import matplotlib.pyplot as plt

def calc_distances(neuron_rows, neuron_cols, winner):
    '''
    Takes in a (d1 x d2) array of the row index of each neuron,
    a (d1 x d2) array of the column index of each neuron,
    and a winning neuron.
    Returns the (d1 x d2) array of distances of each neuron from the winner,
    using the distance formula.
    '''
    return np.sqrt(np.square(np.subtract(neuron_rows, winner[0])) + np.square(np.subtract(neuron_cols, winner[1])))


def displayImage(img, show=False, save=False, filename=''):
    '''
    Takes in a 2d array of img pixel values [0,1],
    an (optional) boolean to plt.show() the image,
    an (optional) boolean to save the image, and
    an (optional) string to use as the filename of the image (if save==True). 
    '''
    img_values = np.mat(img)
    fig, ax = plt.subplots()
    ax.imshow(img_values, cmap='gray')
    if show: 
        plt.show()
        
    plt.close()

    if save:
        fig.savefig(filename)
            
    return img_values

class SOFM():
    def __init__(self, d1, d2, num_features, sigma_o, tau_N):
        '''
        A class to initialize a vanilla, rectangular, Kohonen Self-Organizing Feature Map (SOFM). 
        Takes in two dimensions (d1 and d2) of the map, a number of features in the input vector, 
        a (fixed) hyperparameter sigma_o for the initial neighborhood size, 
        and a (fixed) hyperparameter tau_N for the neighborhood shrinkage rate.
        
        Weights are initialized to small, random, values [0, 0.2).        
        
        '''
        self.d1 = d1
        self.d2 = d2
        self.neurons = np.array([[(i,j) for j in range(self.d1)] for i in range(self.d2)], dtype='i,i')
        self.neuron_rows = np.array([[i for _ in range(self.d1)] for i in range(self.d2)])
        self.neuron_cols = np.array([[j for j in range(self.d1)] for _ in range(self.d2)])
        self.dist_arrays = self.get_distances_for_all_winners()
        self.num_features = num_features
        self.weights = np.random.rand(self.d1 * self.d2, self.num_features) * 0.2 #CHANGEME - weight initialization
        self.sigma_o = sigma_o
        self.tau_N = tau_N


    def get_distances_for_all_winners(self):
        '''
        Initializes a ((d1*d2) x d1 x d2) array of the Euclidian norms of each neuron for every possible winner.
        (This avoids having to compute these values for each input example;
        instead, we just do it for all possible neurons once at the start.)
        '''
        dist_arrs = np.ndarray((self.d1*self.d2, self.d1, self.d2))
        for r in range(self.d1):
            for c in range(self.d2):
                i = self.convert_to_index((r,c))
                dist_arrs[i] = calc_distances(self.neuron_rows, self.neuron_cols, (r,c))

        return dist_arrs


    def convert_to_coord(self, i):
        '''
        Takes in an integer index i, and returns its tuple coordinate based on the dimensions of the SOFM
        '''
        assert type(i) == int, 'Index must be type int' # convert from index to coordinates
        return (i // self.d2, i % self.d2)


    def convert_to_index(self, coords):
        '''
        Takes in a tuple coordinate, and returns its integer index based on the dimensions of the SOFM
        '''
        assert type(coords) == tuple, 'Coordinates must be type tuple' # convert from coordinates to index
        return (coords[0] * self.d2) + coords[1]
            

    def forward(self, input_vec):
        '''
        Takes in a single input vector and a desired return type (tuple or int), 
        and returns the winning neuron in the form of the desired return type
        '''
        norms = np.linalg.norm(self.weights - input_vec, axis=1)
        winner_index = int(np.argmin(norms))

        return self.convert_to_coord(winner_index)


    def sigma(self, epoch):
        '''
        Takes in the current epoch and uses the model's fixed hyperparameters to return the 
        range of the neighborhood.
        '''
        # sigma = sigma_o * exp { -epoch / tau_N }
        return self.sigma_o * math.e ** (-1 * epoch / self.tau_N)


    def neighborhood(self, winner, neighborhood_size):
        '''
        Takes in a winning neuron and current epoch and returns a 2d array (n x n)
        of the Gaussian neighborhood scaling factor for each neuron centered around the winner.
        '''
        # neighborhood =  exp  {    ( -norm(neuron_i - winner) ) ^ 2      }
        #                      { ---------------------------------------  }
        #                      {          2 * sigma(epoch) ^ 2            }
        winner_i = self.convert_to_index(winner)
        dists = self.dist_arrays[winner_i] # get the dist_array for the winner neuron
        top = np.negative(np.square(dists)) 
        bottom = 2 * neighborhood_size ** 2
        return np.exp(np.divide(top, bottom))


    def update_weights(self, input_vec, winner, sigma, lr):
        '''
        Takes in a single input vector, winning neuron, current epoch, and learning rate,
        and updates the model's weights in-place.
        '''
        weight_changes = lr * self.neighborhood(winner, sigma).reshape(self.d1*self.d2,1) * np.subtract(input_vec, self.weights)
        self.weights += weight_changes


    def train(self, img_arr, num_epochs, lr):
        '''
        Takes in a (n x m) array of images, where n = number of inputs and m = number of features;
        a number of epochs to train for, and a learning rate.
        '''
        for epoch in range(num_epochs):
            start_epoch = time.time()
            # get random shuffle of training set each epoch
            img_arr_shuffled = np.random.permutation(img_arr)
            print(f'\n\n--------------Epoch: {epoch}--------------\n\n')

            for q in range(len(img_arr_shuffled)):
                # get winning neuron
                winner = self.forward(img_arr_shuffled[q])
                print(f'Winner of input {q}: {winner}')
                # update weights
                neighborhood_size = self.sigma(epoch)
                self.update_weights(img_arr_shuffled[q], winner, neighborhood_size, lr)
            
            print(f'\n-----Time: {time.time() - start_epoch}-----\n\n')

