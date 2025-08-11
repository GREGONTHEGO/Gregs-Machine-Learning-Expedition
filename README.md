# Greg's Machine Learning Expedition
This repository is a curated collection of my machine learning projects using various datasets and model architectures. Each project explores a different type of problem or modeling approach, helping me understand how the right architecture for a given dataset.

# Iris Dataset
-  **Naive Bayes Classifier** - Uses conditional probabilities to classify Iris flowers based on all features.
- **Support Vector Machine (SVM)** - Applies linear SVM to find the optimal hyperplane with the largest margin.
-  **Decision Tree Regressor** - Recursively splits features to predict continuous outcomes. A depth of 4 effectively partitions the dataset.
- **Custom Neural Networks** - Manual implementation of multi-layer networks with forward/backward propagation and weight updates using gradients.
- **Modular Neural Network Architecture** - Built for flexibility, allowing variable layer sizes and depths, updating weights via gradient descent.
- ## Might need to look into adding perceptron here

# XY Dataset
- **Linear Regression** - Fit a line to noisy XY data, handling NaNs during preprocessing.
- **Ridge Regression + k-Fold Cross Validation** - Compared to linear regression, added L2 regularization to improve generalization. k here being 10 folds that were used to validate performance across k data splits.

# CIFAR Dataset
Experimented with a two-layer CNN, varying:
- Number of filters: 16, 32, 64, 128
- Kernel sizes: 3, 5, 7, 9
- Padding: 0 - 3 (based on kernel size)
Findings:
- Larger filter counts improved performance at higher compute cost
- Kernel size 5 performed best
- Padding of 1-3 yielded comparable results; padding 0 underperformed significantly

# Hospital Dataset

1. Feature Engineering for Mortality Prediction - Cleaned 10 selected columns and trained a neural net to predict patient mortality (~50% accuracy)
2. Full Feature Pipeline - Converted all 47 columns (text to numeric), applied standard scaling, and built a neural network with enhanced feature representation. Performed with (~92% accuracy)

The main difference between these two neural networks is that one uses a subset of the columns of the dataset and the other has access all of the columns.

# IMDB Dataset
Processed reviews using the 10,000 most frequent words and one-hot encoding.
Modeled with a sequential DNN using:
- L1/L2 regularization
- Dropout layers
- Final sigmoid output node
Limitations: Order, punctuation, and word count were not considered.

# MNIST Dataset
Grouped MNIST digits (e.g. [0&1], [2&3], etc.) into five classes.
Model: Sequential neural network with softmax output.
Purpose: Visualize and test generalization on merged class labels.

# Cats and Dogs Dataset
Used **ResNet50** as a frozen base model with custom head layers.
Later experiments unfroze some ResNet layers and fine-tuned with a smaller learning rate to improve classification accuracy.

# Pets Dataset
Visualized random images from the Pets dataset alongside their corresponding trimaps (one color for background, one for forground and one for not-classified). This setup supports later segmentation tasks and mask predictions.

# Entropy and Perceptron
1. Entropy Calculator - Wrote code to compute entropy over distributions.
2. Perceptron Logic Gates - Created perceptrons for AND, OR, and XOR logic gates. Demonstrated that XOR is non-linearly separable.
3. Training Sensitivity - Illustrated how different initial weights and learning rates affect convergence and risk of local minima.
