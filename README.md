# Custom Multi-Layer Perceptron (MLP)



## Original project

This project is an improvement of [this GitHub repo](https://github.com/OmarAflak/Medium-Python-Neural-Network), made by Omar Aflak. I'm *really grateful* to have randomly stumbled upon [his post](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65) where he describes how to build basic Neural Networks from scratch in Python. It really pushed me to better understand the raw details of their implementation ! He also made [a YouTube video](https://www.youtube.com/watch?v=pauPCy_s0Ok) explaining his code, which was very informative and extremely well animated !



## Description

The main purpose of this repo is to experiment with an [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) that is made **from scratch**, (almost) only using the NumPy module.

Dataset used : [MNIST](https://en.wikipedia.org/wiki/MNIST_database)

Language used : **Python**



## Improvements of the original repo

- Automated the creation of the (formatted) training, validation and testing sets by only specifying their respective number of samples at the beginning of the main script (`main.py`). Those 3 sets have the *same class proportions* as the initial raw (MNIST) data
- Added a **validation step** at each epoch
- Added **batch processing** for the training, validation *and* testing phases ! Note that, for the validation and testing phases, the batch size will *not* affect the resulting losses and accuracies. Therefore, for those 2 phases, you might want to put the maximum batch size that your CPU can handle, in order to speed up the computations (`val_batch_size` and `test_batch_size` are set to 32 by default). The batch size also doesn't have to perfectly divide the number of samples of the data that is going to be split into batches !
- The input data can be normalized such that each (input) batch sample has a mean of 0 and a standard deviation of 1. This feature is enabled when you instantiate the `Network` class with the `normalize_input_data` kwarg set to `True` (which is done by default)
- The weights and biases of the Dense layers are now initialized using the [He initialization](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=The%20he%20initialization%20method%20is,of%20inputs%20to%20the%20node.)
- Added the **Categorical Cross-Entropy** (CCE) loss function
- Added the **ReLU**, **leaky ReLU**, **softmax** and **sigmoid** activation functions
- Added the **Input**, **BatchNorm** and **Dropout** layers
- Building the network's architecture is now a bit more user-friendly : you no longer need to keep track of the output size of the previous Dense layer to add a new one ! Also, for the Activation layers, you can now simply input the name of the loss function (which is a string) instead of inputting the tuple of functions `(activation, activation_prime)` !
- The network's detailed summary can now be visualized (after the network is built)
- The detailed history of the network's training phase is printed *dynamically*. After the training is complete, you can even plot the network's history and/or save the plot if requested (it will be saved in the `saved_plots` folder by default)
- The final results are more detailed (they include the global accuracy score and the confusion matrix)
- For testing purposes, you can also plot some predictions of the network (after it's trained) !
- In order to be able to reproduce some results, you can set the seeds related to *all* the random processes directly inside the main script
- Globally speaking, the main script is written such that you can tweak a maximum amount of parameters related to the built MLP !



## Requirements

Run : `pip install -r requirements.txt`



## Run the MLP

Run the main Python script : `python main.py`

Or, equivalently, you can run the main Jupyter notebook (`main_notebook.ipynb`)

**Enjoy !**

