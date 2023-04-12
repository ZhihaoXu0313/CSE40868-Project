# CSE 40868 Course Project:
## Comparing Graph Denoising Performance of Autoencoder and CNN Model

In this project, I have implemented and compared two different neural network architectures, an Autoencoder (AE) and a Convolutional Neural Network (CNN), for graph denoising. The chosen architectures are designed as follows:

### Architectures of NNs:
#### Autoencoder:
**Encoder**: The encoder consists of a series of graph convolutional layers followed by non-linear activation functions (ReLUs are used here). These layers are responsible for encoding the input graph's features into a lower-dimensional latent space representation. Max-pooling layers are employed between convolutional layers.

**Decoder**: The decoder consists of another series of graph convolutional layers followed by non-linear activation functions (also ReLUs). These layers are responsible for reconstructing the denoised graph from the lower-dimensional latent space representation.

**Loss Function**: We have used the mean squared error (MSE) loss function, which measures the difference between the original graph and the denoised graph produced by the autoencoder.

**Optimization Algorithm**: We have used the Adam optimization algorithm with a learning rate of 0.001.

#### CNN:
**Convolutional Layers**: We have used 7 graph convolutional layers. Each layer uses a ReLU activation function. These layers are responsible for extracting features from the input graph.

**Max-Pooling Layers**: After each convolutional layer, we have added max-pooling layers to reduce the spatial dimensions of the feature maps, resulting in a more compact and efficient representation.

**Dropout Layers**: We have used dropout layers with a dropout rate of 0.2 after each max-pooling layer. These layers help to prevent overfitting by randomly dropping a fraction of the neurons during training.

**Loss Function**: We have used the mean squared error (MSE) loss function to measure the difference between the original graph and the denoised graph produced by the CNN.

**Optimization Algorithm**: We have used the Adam optimization algorithm with a learning rate of 0.001.

### Database:
Labeled Faces in the Wild (LFW) is a database of face photographs designed for studying the problem of unconstrained face recognition. This database was created and maintained by researchers at the University of Massachusetts, Amherst. The size of the images in the database is proper for our network scale and it is interesting enough for investigating the performance of our denoising techniques on this database. Gaussian noise is added to each of the image. AE and GNN are used for denoise and the results are compared.
