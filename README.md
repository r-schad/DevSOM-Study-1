# DevSOM-Study-1
The first experiment will be to train a series of growing SOFM networks on MNIST images of increasing resolution. To do this, we take the 28x28 MNIST images and blur them with a 3x3 blurring filter to get the blurred set MNIST-b3, and with a 5x5 blurring filter to get the blurred set MNIST-b5. For consistency, we call the full MNIST set MNIST-b1 (blurred by a 1x1 filter, i.e., not blurred).
## Experiment 1: 
Train a 24x24 SOFM on MNIST-b1 for 3m epochs.
## Experiment 2: 
Train a 6x6 SOFM on MNIST-b5 for m epochs, complexify to 12x12 SOGM and train on MNIST-b3 for m epochs, then complexify to 24x24 and train on MNIST-b1 for m epochs. Compare performance on time and classification metrics with Experiment 1.
## Experiment 3: 
Repeat Experiment 2 with systematically smaller number of epochs for training the final 24x24 network. Compare performances of all trials with Experiments 1 and 2.
## Experiment 4: 
Repeat Experiment 2 keeping the total epochs set to m but systematically allocating a larger fraction of that to smaller networks (use a well-defined protocol). Compare the results with Experiments 1 and 2.
## Readout System: 
To see what the SOGM “thinks” it is seeing, add a 28x28 readout layer to the system. Each neuron of the layer corresponds to a pixel in the readout image. Each readout neuron receives inputs from all the SOFM neurons. The weight vij from SOFM unit j to readout unit i is set to the same value as the (learned) weight wji from input pixel i to SOFM neuron j. The net input of readout neuron i is given by:
s_i=∑_j▒v_ij  y_j
where yj is the activation of the SOFM neuron j in response to the current input image. Since the SOGM neuron activation is based on the Euclidean distance between its weight vector and the current image, the activation should be radial basis function with a peak of 1 at the weight vector of the SOFM neuron. The output of readout neuron i is a sigmoid function of its net input:
z_i=1/(1+exp⁡(〖γs〗_i- θ))
where  and  parameters chosen by the user. With the right parameter setting, the redout image should show what the network’s internal view of the input is.
By having and updating the readout layer during training, we can see how the network is self-organizing from initially random readouts to orgamized ones that accurately reflect the input. Basically, we have trained a self-organized auto-encoder.
