# CSE 40868 Course Project:
## Comparing Graph Denoising Performance of Autoencoder and CNN Model

In this project, I have implemented and compared two different neural network architectures, an Autoencoder (AE) and a Convolutional Neural Network (CNN), for graph denoising. The chosen architectures are designed as follows:

### Autoencoder:
**Encoder**: The encoder consists of a series of graph convolutional layers followed by non-linear activation functions (e.g., ReLU). These layers are responsible for encoding the input graph's features into a lower-dimensional latent space representation.

Decoder: The decoder consists of another series of graph convolutional layers followed by non-linear activation functions. These layers are responsible for reconstructing the denoised graph from the lower-dimensional latent space representation.

Loss Function: We have used the mean squared error (MSE) loss function, which measures the difference between the original graph and the denoised graph produced by the autoencoder.

Optimization Algorithm: We have used the Adam optimization algorithm with a learning rate of 0.001.
