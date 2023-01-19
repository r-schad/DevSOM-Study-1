import numpy as np
import pandas as pd
import math, time
import matplotlib.pyplot as plt
import seaborn as sns
import os

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


def save_model_params(params, filename):
    df = pd.DataFrame(params)
    df.to_excel(filename, index=False, header=False)

    
def load_model_params(filename):
    return pd.read_excel(filename, header=None).to_numpy()


class SOFM():
    def __init__(self, d1, d2, image_dims, init_nhood, init_lr):
        '''
        A class to initialize a vanilla, rectangular, Kohonen Self-Organizing Feature Map (SOFM). 
        Takes in two dimensions (d1 and d2) of the map, a tuple of the input dimensions, 
        an initial neighborhood range, 
        and an initial learning rate.
        
        Weights are initialized to small, random, values [0, 0.2).        
        
        '''
        self.d1 = d1
        self.d2 = d2
        self.neurons = np.array([[(i,j) for j in range(self.d1)] for i in range(self.d2)], dtype='i,i')
        self.neuron_rows = np.array([[i for _ in range(self.d1)] for i in range(self.d2)])
        self.neuron_cols = np.array([[j for j in range(self.d1)] for _ in range(self.d2)])
        self.dist_arrays = self.get_distances_for_all_neurons()
        self.image_dims = image_dims
        self.num_features = self.image_dims[0] * self.image_dims[1]
        self.weights = np.random.rand(self.d1 * self.d2, self.num_features) * 0.2
        self.init_nhood = init_nhood
        self.init_lr = init_lr

    def get_distances_for_all_neurons(self):
        '''
        Initializes a ((d1*d2) x d1 x d2) array of the Euclidian norms of each neuron for every possible winner.
        (This avoids having to compute these values for each input example;
        instead, we just do it for all possible neurons once at the start of the current training stage.)
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
        and returns the winning neuron as a coordinate
        '''
        norms = np.linalg.norm(self.weights - input_vec, axis=1)
        winner_index = int(np.argmin(norms))

        return self.convert_to_coord(winner_index)


    def hyperbolic_decay(self, init_value, epoch, total_iterations):
        '''
        Takes in an initial value for the hyperparameter undergoing hyperbolic decay,
        the current epoch, and the total number of iterations, where
        total_iterations = number of epochs to be trained * size of training set.

        Computes the value of the hyperparameter of the current epoch as:
        a(t) = a_0 / (1 + (2t / T))

        Where a is the hyperparameter to be computed, t is the current epoch, 
        and T is the total number of iterations of the current training stage.
        
        Returns the value of the hyperparameter for the current epoch.
        '''
        return init_value / (1 + (2 * epoch / total_iterations))


    def learning_rate(self, epoch, total_iterations):
        '''
        Takes in the current epoch and the total number of iterations for the current training stage, 
        and returns the learning rate at the current epoch.
        '''
        return self.hyperbolic_decay(self.init_lr, epoch, total_iterations)

    
    def neighborhood_range(self, epoch, total_iterations):
        '''
        Takes in the current epoch and the total number of iterations for the current training stage, 
        and returns the size of the neighborhood at the current epoch.
        '''
        return self.hyperbolic_decay(self.init_nhood, epoch, total_iterations)


    def neighborhood(self, winner, neighborhood_size):
        '''
        Takes in a winning neuron and current epoch and returns a 2d array (n x n)
        of the Gaussian neighborhood scaling factor for each neuron centered around the winner.
        '''
        winner_i = self.convert_to_index(winner)
        dists = self.dist_arrays[winner_i] # get the dist_array for the winner neuron
        top = np.negative(np.square(dists)) 
        bottom = 2 * neighborhood_size ** 2
        return np.exp(np.divide(top, bottom))


    def update_weights(self, input_vec, winner, neighborhood_size, lr):
        '''
        Takes in a single input vector, winning neuron, neighborhood range, and learning rate,
        and updates the model's weights in-place.
        '''
        weight_changes = lr * self.neighborhood(winner, neighborhood_size).reshape(self.d1*self.d2,1) * np.subtract(input_vec, self.weights)
        self.weights += weight_changes

    
    def train(self, img_arr, current_stage, num_epochs, lr, readout_interval=0, readout_examples=[], readout_path=''):
        '''
        Takes in a (n x m) array of images, where n = number of inputs and m = number of features;
        a number of epochs to train for, and a learning rate.
        '''
        total_iterations = num_epochs * img_arr.size[0]
        for epoch in range(num_epochs):
            # save readouts
            if readout_interval != 0:
                if (epoch % readout_interval == 0):
                    self.plot_readouts(readout_examples, current_stage, epoch, alpha=10, gamma=2, theta=0., filepath=readout_path)
                    self.plot_readout_process(readout_examples, current_stage, epoch, alpha=10, gamma=2, theta=0., filepath=readout_path)


            start_epoch = time.time()
            # get random shuffle of training set each epoch
            img_arr_shuffled = np.random.permutation(img_arr)
            print(f'\n--------------Stage {current_stage} | Epoch: {epoch}--------------')

            for q in range(len(img_arr_shuffled)):
                # get winning neuron
                winner = self.forward(img_arr_shuffled[q])
                if q % 1000 == 0:
                    print(f'{round(q / len(img_arr_shuffled) * 100, 1)}%')
                # update weights
                neighborhood_size = self.neighborhood_range(epoch, total_iterations)
                lr = self.learning_rate(epoch, total_iterations)
                self.update_weights(img_arr_shuffled[q], winner, neighborhood_size, lr)
            
            print(f'------------Time: {time.time() - start_epoch}------------')


    def complexify(self, new_d1,  new_d2, weight_noise, new_init_nhood, new_init_lr):
        sofm_scale_factor = (new_d1 / self.d1) * (new_d2 / self.d2)
        self.weights = np.repeat(self.weights, repeats=sofm_scale_factor, axis=0) + np.random.normal(0, weight_noise, (new_d1*new_d2, 784))
        self.d1 = new_d1
        self.d2 = new_d2
        self.neurons = np.array([[(i,j) for j in range(self.d1)] for i in range(self.d2)], dtype='i,i')
        self.neuron_rows = np.array([[i for _ in range(self.d1)] for i in range(self.d2)])
        self.neuron_cols = np.array([[j for j in range(self.d1)] for _ in range(self.d2)])
        self.dist_arrays = self.get_distances_for_all_neurons()
        self.init_nhood = new_init_nhood
        self.init_lr = new_init_lr


    def get_readout_weights(self):
        readout_weights = np.zeros((self.num_features, int(self.d1 * self.d2)))
        for j in range(self.weights.shape[0]):
            for i in range(self.weights.shape[1]):
                readout_weights[i][j] = self.weights[j][i]

        return readout_weights


    def sofm_activation(self, input_vec, alpha):
        '''
        Given an input vector and a Gaussian scaling parameter, alpha,
        calculates the activation of the SOFM neurons given the current input
        using a Gaussian Radial Basis Function with peak 1 at the weight vector
        of the SOFM neuron. Returns the vector of SOFM activations.
        '''
        return np.exp(-1 * np.square(np.linalg.norm(self.weights - input_vec, axis=1)) / alpha)


    def readout_activation(self, readout_net_inputs, gamma, theta):
        return 1 / (1 + np.exp(gamma * (readout_net_inputs - theta)))


    def readout(self, input_vec, alpha, gamma, theta):
        activations = self.sofm_activation(input_vec, alpha)
        readout_weights = self.get_readout_weights()
        readout_net_inputs = np.sum(np.multiply(readout_weights, activations), axis=1) # FIXME: change to np.dot??
        readout_outputs = 1 - self.readout_activation(readout_net_inputs, gamma, theta) 
        return readout_outputs


    def plot_readouts(self, input_vecs, current_stage, current_epoch, alpha, gamma, theta, filepath):
        f, axs = plt.subplots(2, 10)
        for i in range(10):
            axs[0][i].imshow(input_vecs[i].reshape(self.image_dims[0], self.image_dims[1]), cmap='gray', vmin=0, vmax=1)
            axs[0][i].get_xaxis().set_visible(False)
            axs[0][i].get_yaxis().set_visible(False)
            readout = self.readout(input_vecs[i], alpha=alpha, gamma=gamma, theta=theta)
            axs[1][i].imshow(readout.reshape(self.image_dims[0], self.image_dims[1]), cmap='gray', vmin=0, vmax=1)
            axs[1][i].get_xaxis().set_visible(False)
            axs[1][i].get_yaxis().set_visible(False)
        
        f.suptitle(f'Readouts of Stage {current_stage} at Epoch {current_epoch}\nAlpha: {alpha} | Gamma: {gamma} | Theta: {theta}')
        plt.tight_layout()
        figname = filepath + f"/readouts_epoch{current_epoch}_alpha{alpha}_gamma{gamma}_theta{theta}".replace('.','p')
        f.savefig(figname)
        plt.close()


    def grid_search_readouts(self, readout_examples, current_stage, current_epoch, alphas, gammas, thetas, filepath):
        gs_dir = filepath + '/grid_search'
        if not os.path.isdir(gs_dir):
            os.mkdir(gs_dir)
        
        for alpha in alphas:
            for gamma in gammas:
                for theta in thetas:
                    self.plot_readouts(readout_examples, current_stage, current_epoch, alpha=alpha, gamma=gamma, theta=theta, filepath=gs_dir)


    def plot_readout_process(self, input_images, current_stage, current_epoch, alpha, gamma, theta, filepath):
        f, axs = plt.subplots(10, 3)
        f.suptitle(f'Stage: {current_stage} at Epoch {current_epoch} | Alpha: {alpha} | Gamma: {gamma} | Theta: {theta}')
        for i in range(10):
            activations = self.sofm_activation(input_images[i], alpha=alpha)
            readout = self.readout(input_images[i], alpha=alpha, gamma=gamma, theta=theta)

            axs[i][0].imshow(input_images[i].reshape(self.image_dims[0], self.image_dims[1]), cmap='gray', vmin=0, vmax=1)
            axs[i][1].imshow(activations.reshape(self.d1,self.d2), cmap='gray', vmin=0, vmax=1)
            axs[i][2].imshow(readout.reshape(self.image_dims[0], self.image_dims[1]), cmap='gray', vmin=0, vmax=1)
            axs[i][0].set_xticks([])
            axs[i][0].set_yticks([])
            axs[i][1].set_xticks([])
            axs[i][1].set_yticks([])
            axs[i][2].set_xticks([])
            axs[i][2].set_yticks([])
        
        axs[0][0].set_title('Input Image', fontsize=10)       
        axs[0][1].set_title('Activations', fontsize=10)       
        axs[0][2].set_title('Readout', fontsize=10)

        f.savefig(filepath + f'/readout_process_epoch{current_epoch}_alpha{alpha}_gamma{gamma}_theta{theta}'.replace('.', 'p'))
        plt.close()
        
        
    def visualize_weights(self, filename):
        '''
        Given a filename, plots the weights of all neurons in the SOFM in a grid of shape (self.d1, self.d2).

        NOTE: Due to the sheer amount of data to plot, 
        this function takes a very significant amount of time to complete (up to an hour). 
        '''
        print('Creating neuron visualization plot...')
        plt.close()
        try: 
            fig, axs = plt.subplots(self.d1, self.d2, figsize=(self.d1,self.d2), sharex=True, sharey=True)
        except NotImplementedError:
            fig, axs = plt.subplots(self.d1, self.d2, figsize=(self.d1,self.d2), sharex=True, sharey=True)

        for r in range(self.d1):
            print(f'{100 * r / self.d1}%')
            for c in range(self.d2):
                i = self.convert_to_index((r,c))
                ax = axs[r][c]

                # plot image on subplot
                ax.imshow(self.weights[i].reshape(self.image_dims[0], self.image_dims[1]), cmap='gray', vmin=0, vmax=1)
                
                ax.set_xbound([0,self.image_dims[1]])

        plt.tight_layout()
        fig.savefig(filename)
        return fig


    def plot_win_percentages(self, win_percentages, filename):
        '''
        Given the win percentages of all neurons for each class, plots a heatmap of win percentages for each class
        and saves the figure of all combined heatmaps to filename.
        '''
        # plot win percentages on a heatmap for each class
        print('Plotting win percentage distribution heatmaps...')
        plt.close()
        fig, axs = plt.subplots(5, 2, figsize=(10, 25))
        for row in range(5):
            for col in range(2):
                class_i = (row * 2) + col
                axs[row][col].set_title(f'Class {class_i}')
                axs[row][col].set_xticks([], labels='')
                axs[row][col].set_yticks([], labels='')

                sns.heatmap(win_percentages[class_i], linewidths=.5, fmt= '.2f', ax=axs[row][col], cbar=False)

        plt.tight_layout()
        fig.savefig(filename)
        return fig
        

    def write_stats(self, experiment_name, current_stage, train_start, train_end, num_epochs, train_set_size, ncl_score=None, classification_score=None, filename='stats.txt'):
        '''
        Given a start and end time of training, a number of epochs, the size of the training set, the learning rate,
        and (optionally) an NCL metric score and a Classification metric score,
        writes all stats (and hyperparameters) to a text file, with the provided filename.        
        '''
        # write out stats file
        print('Writing out stats file...')
        if os.path.exists(filename):
            mode = 'a'
        else:
            mode = 'w+'

        with open(filename, mode) as f:
            f.write(f'{experiment_name}: Stage {current_stage}\n')
            f.write('\nTraining Start Time: ' + str(train_start))
            f.write('\nTraining End Time: ' + str(train_end))
            f.write('\nTraining Duration: ' + str(train_end - train_start))
            f.write('\nSOFM Shape: ' + str(self.d1) + ' x ' + str(self.d2))
            f.write('\nTotal Number of Epochs: ' + str(num_epochs))
            f.write('\nSize of Training Set: ' + str(train_set_size))
            f.write('\nInitial Learning Rate: ' + str(self.init_lr))
            f.write('\nInitial Neighborhood Size: ' + str(self.init_nhood))
            if ncl_score:
                f.write('\nNormalized Category Localization (NCL) Score: ' + str(ncl_score))
            if classification_score:
                f.write('\nClassification Score: ' + str(classification_score))
            f.write('\n\n\n')
        
        return filename

    def create_entropy_plot(self, entropy, filename):
        '''
        Given the entropy of all neurons, plots the entropy heatmap and saves the figure to filename.
        '''
        # create entropy plot
        plt.close()
        print('Creating entropy plot...')
        fig = plt.figure()
        try:
            plt.imshow(entropy, cmap='hot_r', vmin=0, vmax=2)
        except NotImplementedError:
            plt.imshow(entropy, cmap='hot_r', vmin=0, vmax=2)
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(filename)
        return fig

    def visualize_neuron_classes(self, neuron_labels, filename):
        '''
        Given the class labels that each neuron is most tuned to, creates a colored grid of all 
        neuron labels, and saves the figure to filename. 
        '''
        plt.close()
        fig = plt.figure()
        try:
            plt.matshow(neuron_labels, cmap='tab10', vmin = np.min(neuron_labels) - 0.5, vmax = np.max(neuron_labels) + 0.5)
        except NotImplementedError:
            plt.matshow(neuron_labels, cmap='tab10', vmin = np.min(neuron_labels) - 0.5, vmax = np.max(neuron_labels) + 0.5)

        plt.colorbar(ticks=range(int(np.min(neuron_labels)), int(np.max(neuron_labels)) + 1, 1))
        plt.savefig(filename)
        return fig

    def calc_win_percentages(self, dataset, labels):
        '''
        Given a dataset and its corresponding labels, calculates the win counts and win percentages 
        of each neuron in the SOFM for each class.

        Returns a tuple of the (win_counts, win_percentages).
        '''
        # get win percentage of each neuron by class
        winners = {}
        win_percentages = np.ndarray((10, self.d1, self.d2))
        win_counts = np.ndarray((10, self.d1, self.d2))
        for i in range(10):
            print(f"\n------Computing win percentages for class {i}------")
            # get subset of dataset with only examples of class i 
            dataset_i_only = dataset[labels==i]
            winners[i] = np.empty((len(dataset_i_only)), dtype='object')

            # get list of all winning neurons for each example of class i
            for q in range(len(dataset_i_only)):
                if q % 1000 == 0:
                    print(f'{round(q / len(dataset_i_only) * 100, 1)}%')
                winner_coords = self.forward(dataset_i_only[q].reshape(784))
                winners[i][q] = winner_coords

            # initialize win percentages and win counts of class i for each neuron 
            win_percentages[i] = np.zeros((self.d1,self.d2), dtype=np.float32)
            win_counts[i] = np.zeros((self.d1,self.d2), dtype=np.int32)
            
            # get all unique winning neurons and their corresponding win counts for class i
            unique_winners, counts = np.unique(winners[i], return_counts=True)
            for winner_coords, count in zip(unique_winners, counts):
                row = winner_coords[0]
                col = winner_coords[1]

                win_counts[i][row][col] = count
                # calculate percentage of all inputs of class i that are won by neuron [row, col]
                percentage = count / len(winners[i])
                win_percentages[i][row][col] = percentage

        return win_counts, win_percentages

    def calc_classification_metric(self, dataset, data_labels, neuron_labels):
        '''
        Given a dataset, its corresponding labels, and the neuron_labels that each
        neuron is tuned to, calculates the proportion of correctly classified input examples (hit_rate).

        Returns the hit rate and the confusion matrix.
        
        '''
        # get win percentage of each neuron by class (using testing data)
        conf_matrix = np.zeros((10, 10), dtype=np.int32)
        
        num_correct = 0
        for q in range(len(dataset)):
            if q % 1000 == 0:
                print(f'{round(q / len(dataset) * 100, 1)}%')

            winner_coords = self.forward(dataset[q])
            
            row = winner_coords[0]
            col = winner_coords[1]

            winner_label = neuron_labels[row][col]
            true_label = data_labels[q]
            conf_matrix[winner_label][true_label] += 1
            if winner_label == true_label:
                num_correct += 1

        hit_rate = num_correct / len(dataset)

        return hit_rate, conf_matrix
            

    def plot_conf_matrix(self, conf_matrix, filename):
        print('Plotting confusion matrix...')
        plt.close()
        fig = plt.figure()
        sns.heatmap(conf_matrix, cmap='hot', annot=True, fmt='g')
        fig.savefig(filename)
        return fig


    def calc_entropy(self, win_counts):
        ''' 
        Given the win_counts of all neurons for each class, calculates the entropy across all neurons.

        Returns the entropy of all neurons.
        '''
        # calculate entropy of each neuron
        total_wins = np.sum(win_counts, axis=0)
        fractions = np.nan_to_num(np.divide(win_counts, total_wins))
        log_fractions = np.log10(fractions, where=fractions > 0.0)
        entropy = -1 * (np.sum(np.multiply(fractions, log_fractions), axis=0))
        
        return entropy

    def calc_ncl_metric(self, win_counts, win_percentages):
        '''
        Given the win_counts and win_percentages of all neurons for each class, calculates the normalized class localization metric.

        Returns the network's mean NCL metric score. 
        '''
        total_wins = np.sum(win_counts, axis=0)
        fractions = np.nan_to_num(np.divide(win_counts, total_wins))
        ncl_per_class = np.sum(win_percentages * fractions, axis=(1,2))
        mean_ncl = (1 / 10) * np.sum(ncl_per_class, axis=0)

        return mean_ncl

