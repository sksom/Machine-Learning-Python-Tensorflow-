# Machine-Learning-Python-Tensorflow-
Experimental analysis of old style neural network and CNN(Convoluted Neural Network)

Experiments with a deep neural network using TensorFlow

Background:

In this project the objective was to experiment with methods of combating overfitting and the ADAM optimization algorithm. In all cases the network should be constructed so that its accuracy on test data is as high as possible. The result were four python scripts. Please notice that in order to run the script you should have the file our_mnist.py in the same folder as the script you are running. (You are not allowed to make any changes to our_mnist.py). It was created by reducing the number of training data of the mnist dataset. With a small number of training data it is expected that overfitting will be significant.

Script 1. In this script I used the tf.train.GradientDescentOptimizer.

Script 2. The limitations here are exactly the same as in Script1, except that I used tf.train.AdamOptimizer instead of tf.train.GradientDescentOptimizer.

Script 3. The limitations here are similar to Script1. I used the tf.train.GradientDescentOptimizer, and all layers are fully connected. I used other available options from the TensorFlow API to enhance the result and create a hybrid network.

Script 4. The limitations here are similar to Script2. I used the tf.train.AdamOptimizer, and all layers are fully connected. But here also I used other available options from the TensorFlow API to enhance the result and create a hybrid network.

Evaluation
The challenge is to get as high accuracy as possible, with the limitation that each script uses at most 3000 stochastic batches. Observe that because of the random initialization different runs of the programs may produce (slightly) different accuracy values.
