<center><h1>Custom-MLP</h1></center>
<center><h3>Implementing a Multi-Layer Perceptron (MLP) from scratch</h3></center>
<center><h4><i>NumPy is all you need</i></h4></center>

<br/>

## Original project

This project is an improvement of [this GitHub repo](https://github.com/OmarAflak/Medium-Python-Neural-Network), made by Omar Aflak. I'm *really grateful* to have randomly stumbled upon [his post](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65) where he describes how to build basic neural networks from scratch in Python. It really pushed me to better understand the raw details of their implementation ! He also made [a YouTube video](https://www.youtube.com/watch?v=pauPCy_s0Ok) explaining his code, which was very informative and extremely well animated !

<br/>

## Description

The main purpose of this repo is to experiment with a [Multi-Layer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) made **from scratch**, almost only using the **NumPy** module.

Main task of the MLP network : **Data classification**

Default dataset used : [MNIST](https://en.wikipedia.org/wiki/MNIST_database) (it can easily be replaced by another dataset ; see one of the following sections)

Language used : **Python**

<br/>

## Improvements of the original repo

- At the very beginning of the main script (`main.py`), you can **select the type of the data** that will flow through the network. For now, the only available datatypes are `float32` (default) and `float64`. Typically, compared to `float64` data, `float32` data will naturally make the computations a little less accurate, but they will be done faster and use less RAM and/or CPU !
- At the very beginning of the main script, you can also, if you want, **select a specific subset of class indices to work with**. For instance, if you only want to work with the class indices `2`, `4` and `7` (and not with all the class indices ranging from `0` to `9` for example), then all you need to do is set the variable `selected_classes` to the list `[2, 4, 7]` (in the main script) !
- **[<ins>Specific to the MNIST dataset</ins>]** Automated the creation of the (formatted) training, validation and testing sets by only specifying their respective number of samples at the beginning of the main script. Those 3 sets will have a **uniform class distribution**. In particular, if **all** the classes are selected, then those 3 sets will have (roughly) the **same class distribution** as the initial raw MNIST data (since the latter also has a uniform class distribution) !
- Added a **validation step** at each epoch. Even though this step is **optional**, it's highly recommended !
- Added **batch processing** for the training, validation *and* testing phases ! Note that, for the validation and testing phases, the batch size will *not* affect the resulting losses and accuracies. Therefore, for those 2 phases, you might want to put the maximum batch size your CPU can handle, in order to speed up the computations (`val_batch_size` and `test_batch_size` are set to `32` by default in the main script). The batch size also doesn't have to perfectly divide the number of samples of the data that is going to be split into batches ! In addition, in order to save memory during training, the (training) batches are created using a **generator function** (this is also done during testing)
- The input data can be normalized such that each (input) sample has a mean of `0` and a standard deviation of `1` (i.e. the data can be **standardized**). This feature is enabled when you instantiate the `Network` class with the `standardize_input_data` kwarg set to `True` (which is done by default)
- The weights and biases of the Dense layers are now initialized using the [He initialization](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=The%20he%20initialization%20method%20is,of%20inputs%20to%20the%20node.)
- Added the **Categorical Cross-Entropy** (CCE) loss function (in addition to the **Mean Squared Error**, or "MSE")
- Added the **ReLU**, **leaky ReLU**, **softmax** and **sigmoid** activation functions (in addition to the hyperbolic tangent, or **tanh**)
- Added the **Input**, **BatchNorm** and **Dropout** layers (in addition to the **Dense** and **Activation** layers)
- Added the **Adam** and **RMSprop** optimizers (in addition to the **Stochastic Gradient Descent**, or "SGD")
- Building the network's architecture is now a bit more user-friendly : you no longer need to keep track of the output size of the previous layer to add a new Dense layer ! Also, for the Activation layers, you can now simply input the name of the loss function (which is a string) instead of inputting the functions `activation` and `activation_prime` !
- Added an **"early stopping" callback**
- The detailed summary of the network's architecture can now be visualized right after the network is built
- The detailed history of the network's training phase is printed *dynamically*, at the end of each epoch. After the training is complete, you can even plot the network's history and/or save the plot if requested (the plot will be saved in the `saved_plots` folder by default)
- The network can also be **saved to the disk**, even if it's not trained (the network will be saved in the `saved_networks` folder by default)
- The final results are more detailed. They include : the **global accuracy score**, the **global "top-N accuracy score"** (where `N` can be defined), the **testing loss**, the **mean confidence levels** (of the correct and false predictions), the **raw confusion matrix** and the **normalized confusion matrices** (related to the **precision** and the **recall**) of the network
- For testing purposes, you can also plot some predictions of the network (after it's trained) !
- In order to be able to **reproduce some results**, you can set the seeds related to *all* the random processes directly from the main script
- Globally speaking, the main script is written such that you can tweak a <ins>**maximum**</ins> amount of parameters related to the MLP you want to build !

<br/>

## How to test this project with another dataset

- First, all the samples in your data must be numeric, and have the <ins>same dimensions</ins>
- The formatted data arrays (i.e. `X_train`, `X_test` and/or `X_val`) have to be 2D, where each row of the data is the <ins>flattened</ins> version of each (corresponding) sample
- The formatted label arrays (i.e. `y_train`, `y_test` and/or `y_val`) have to either be <ins>one-hot encoded</ins> or <ins>1D arrays of (positive) integers</ins>. The classes they represent also need to be the same : for instance, if `y_train` contains the classes `0`, `1` and `2`, then `y_test` and/or `y_val` also need to contain the classes `0`, `1` and `2`. Also, quite naturally, their first dimension needs to match the first dimension of their corresponding data subset !

**Assuming all the previous conditions are met**, you can simply replace the `"Loading and formatting the data"` section of the main script with *yours* ! Basically, all you need to do is define <ins>valid</ins> versions of `X_train`, `y_train`, `X_test` and `y_test` (and/or `X_val` and `y_val`) by the end of that replaced section (the generated data and label arrays will be checked right after that section anyway)

<br/>

## Requirements

<ins>Python version</ins> : has to be greater than or equal to **`3.6`**

Run : `pip install -r requirements.txt`

<br/>

## Run the MLP

Run the main Python script : `python main.py`

Or, <ins>equivalently</ins>, you can run the main Jupyter notebook (`main_notebook.ipynb`)

**Enjoy !**
