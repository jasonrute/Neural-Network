## Neural Network ##

This is an implementation of a neural network in Python3/NumPy.  It has a customizable number of input/output nodes and hidden layers.  I put work into making sure it runs relatively quickly.

This is mostly a hobbiest project, but I have used it for programming competitions on CodinGame.com where one doesn't have access to professional implementations such as TensorFlow (or it takes too long to load/run TensorFlow in the constraints given).

I provided an example where the neural network tries to predict the next frame of a cellular automata.  (Note, it takes as inputs whether or not a cell survives based on it's neighbors.  It does not read the whole grid as an image.)