# KerasLSTM
Using Keras LSTM in Java. THe ipython notebook exports the weights of an LSTM
and also exports some test for a random input sequence and the resulting output.
The Java implementation reads the exported weights and then applies them. All
matrix operations are implemented using jBlas.

+ The numeric helper class implements matrix loading and slicing.
+ Tanh and the sigmoid functions are also in the helper class.
+ The LSTM state is a seperate class holding the cell state and the hidden activation.
+ The LSTM holds all the weights and applies them to sequences.
