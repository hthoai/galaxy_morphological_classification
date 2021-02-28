<div align="center">

# Galaxy Morphological Classification

</div>

Project Management: [Galaxy](https://www.notion.so/hoaiht/SCAN-Galaxy-5b42307525324c82876dae00713375e3)

Our model is implemented using Python 3 and PyTorch framework. The model is a convolutional neural network, and it is trained on Nvidia GPU. We train the model with 70 epochs. Each epoch takes 12 minutes on average time to train. The last output layer in our model incorporates probability constraints. Among 37 classes, there are classes that come from answering the same questions which means their probabilities sum up to 1. Thus, by grouping these output units together, we can use softmax activation function to obtain the final output.

**Results:**

| Dataset                      | RMSE    |
|------------------------------|---------|
| Val set                      | 0.1673  | 
| Test set (Kaggle submission) | 0.2123  |
