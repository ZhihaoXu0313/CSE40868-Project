# CSE 40868 Course Project:
## Comparing Graph Denoising Performance of Autoencoder and CNN Model

In this project, I have implemented and compared two different neural network architectures, an Autoencoder (AE) and a Convolutional Neural Network (CNN), for graph denoising. The chosen architectures are designed as follows:

### Autoencoder:
**Encoder**: The encoder consists of a series of graph convolutional layers followed by non-linear activation functions (ReLUs are used here). These layers are responsible for encoding the input graph's features into a lower-dimensional latent space representation. Max-pooling layers are employed between convolutional layers.

**Decoder**: The decoder consists of another series of graph convolutional layers followed by non-linear activation functions (also ReLUs). These layers are responsible for reconstructing the denoised graph from the lower-dimensional latent space representation.

**Loss Function**: We have used the mean squared error (MSE) loss function, which measures the difference between the original graph and the denoised graph produced by the autoencoder.

**Optimization Algorithm**: We have used the Adam optimization algorithm with a learning rate of 0.001.

### CNN:
**Convolutional Layers**: We have used three graph convolutional layers with 32, 64, and 128 filters, respectively. Each layer uses a ReLU activation function. These layers are responsible for extracting features from the input graph.

**Max-Pooling Layers**: After each convolutional layer, we have added max-pooling layers to reduce the spatial dimensions of the feature maps, resulting in a more compact and efficient representation.

**Dropout Layers**: We have used dropout layers with a dropout rate of 0.25 after each max-pooling layer. These layers help to prevent overfitting by randomly dropping a fraction of the neurons during training.

**Flatten Layer**: This layer is used to convert the output from the last max-pooling layer into a 1D array, which is required for the fully connected layers.

**Fully Connected Layers**: We have two fully connected layers, the first with 128 neurons and the second with the number of output classes. The first layer uses a ReLU activation function, while the second layer uses a softmax activation function to produce probability distributions over the output classes.

**Loss Function**: We have used the mean squared error (MSE) loss function to measure the difference between the original graph and the denoised graph produced by the CNN.

**Optimization Algorithm**: We have used the Adam optimization algorithm with a learning rate of 0.001.
