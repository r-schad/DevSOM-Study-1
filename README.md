# DevSOM-Study-1
The first study will be to train a series of growing SOFM networks on MNIST images of increasing resolution. To do this, we take the 28x28 MNIST images and blur them with a 3x3 blurring filter to get the blurred set MNIST-b3, and with a 5x5 blurring filter to get the blurred set MNIST-b5. For consistency, we call the full MNIST set MNIST-b1 (blurred by a 1x1 filter, i.e., not blurred).
## Experiment 1: 
Train a 24x24 SOFM on MNIST-b1 for $3m$ epochs.
## Experiment 2: 
Train a 6x6 SOFM on MNIST-b5 for $m$ epochs, complexify to 12x12 SOFM and train on MNIST-b3 for $m$ epochs, then complexify to 24x24 and train on MNIST-b1 for $m$ epochs. Compare performance on time and classification metrics with Experiment 1.
## Experiment 3: 
Repeat Experiment 2 with systematically smaller number of epochs for training the final 24x24 network. Compare performances of all trials with Experiments 1 and 2.
## Experiment 4: 
Repeat Experiment 2 keeping the total epochs set to $m$ but systematically allocating a larger fraction of that to smaller networks (use a well-defined protocol). Compare the results with Experiments 1 and 2.
## Readout System: 
To see what the SOFM “thinks” it is seeing, add a 28x28 readout layer to the system. Each neuron of the layer corresponds to a pixel in the readout image. Each readout neuron receives inputs from all the activations of the SOFM neurons. Since the normal SOFM algorithm does not compute an output of the SOFM neurons, define the SOFM activation function as a Gaussian Radial Basis Function with a peak of 1 at the weight vector of the SOFM neuron. Then, compute the activation, $y$, in response to the current input image, $\vec{x}$, of SOFM neuron $j$ by:

$$y_j = e^{\frac{-1}{\alpha}{||\vec{w_j}-\vec{x}||^2}}$$

where $\alpha$ is a scaling parameter of the Gaussian function.

The weight $v_{ij}$ from SOFM unit $j$ to readout unit $i$ is set to the same value as the (learned) weight $w_{ji}$ from input pixel $i$ to SOFM neuron $j$ (i.e. $v_{ij} = w_{ji}$). The net input of readout neuron $i$ is given by:

$$s_i = \sum_{j} v_{ij}y_j$$

Then, the output of readout neuron $i$ is a sigmoid function of its net input:

$$z_i = \frac{1}{1 + e^{-\gamma({s_i}-\theta)}}$$

where $\gamma$ and $\theta$ are parameters chosen by the user. With the right parameter setting, the readout image should show what the network’s internal view of the input is.
By having and updating the readout layer during training, we can see how the network is self-organizing from initially random readouts to organized ones that accurately reflect the input. Basically, we have trained a self-organized auto-encoder.
