# Greg's Machine Learning Expedition
This is a compilation of all machine learning projects I have completed. The projects are organized by the dataset and further organized by the model used to sort them. The goal of this repo is to help me understand when looking at a dataset which architecture to used and also to learn more about what the various architectures are good at.

# Iris Dataset
The first project that I completed with the Iris dataset was to create and use a perceptron. A perceptron classifies inputs into discrete classes using a step function. In the Iris data case this was to split the Setosa and Virginica flowers by their sepal length and width. This is supervised.

The second project that I completed with this dataset was creating a Naive Bayes classification. This is a supervised learning approach that predicts the class based on its features by calculating the probability of each class given the features. This used and split the data for all three flowers and for all of the features.

The third project on this dataset was to create a linear support vector machine. This is used on data that can be split linearly and finds the best plane that has the largest margin. The reason for the largest margin to be important is so that it is more likely that the line holds true with tests.

The fourth project is to create a tree regressor. This will essentially split the data based on the best split at the moment. For example, for the first split it looks at all the data for all three flowers and can split one of them completely. Then it uses data from the left and right split to make more splits. From the Iris data, it seems that a depth of 4 is enough to perfectly split the data.

The Fifth project on the Iris dataset was to create our own hidden layers and code the math that accurs behind the scenes and compute the accuracy and loss. This entails both the forward and backward propagation as well as writing the code that will update the weight and bias of all of the connections. This neural network will be trained and tested on several combinations of the different flowers and the features that they have.

The last project that I have completed on the Iris dataset was creating neural network that can easily update for having more layers and neurons. This also updates weights and biases based on gradients rather than separately calculating for the individual layer.

# XY Dataset
The project completed on a bunch of xy datapoints was to find the line of best fit. This project introduced cleaning the dataset by getting rid of any values that had NaN. Linear Regression is a model that estimates the relationship between scalar response and one or more explanatory variables. This uses MSE as the loss function.

The next project was on the same dataset but it compares linear regression to ridge regression. Ridge regression is linear regression with an added regularization term which helps prevent overfitting and improves the model stability. I also run the ridge regression through a kFold cross validation. kFold uses the data such that the validation segment changes. In my case we had 10 folds which means that we have 10 different training and validation splits that the ridge regression is run on.

# CIFAR Dataset
The project was to create a two layer CNN and test different number of filters, size of kernels, and different amounts of padding. The number of filters that I tested was 16, 32, 64, 128, for which they consistently did better but they also took longer to run when there were more filters. The sizes of kernels tested was 3,5,7,9, for which a kernel size of 5 did the best and 9 did the worst with 3 and 7 looking similar. Now, I do understand that with bigger kernel size we should use a bigger amount of padding however, the test above was to just see what the difference does. The last set of tests for padding 0,1,2,3, shows that 0 does much worse than the others and 1,2, and 3 are all very similar.

# Hospital Dataset

The project here is cleaning and making all of the features numeric so that my NN can learn how likely a pacient is going to pass away. This only cleans 10 of the columns that I thought was important. However, I learned that giving features that do not seemingly have any relation to death can make the neural network much more accurate.

In another project using the same dataset I switched from using 10 columns to using all 47 columns. This means I switched all of the words in the dataset into numbers. I also used a standard scaler as there are certain columns where I do not want the answers to be worth more or less than each other. I will be working on making this better over the summer by looking into other feature working techniques.

# IMDB Dataset

This project takes the imdb dataset and loads the 10000 most frequent words decodes and one hot encodes them to their position in the 10000 words. The review will be one hot encoded so that if there is "great" in the review the index of "great" in the 10000 will be set to one. The problems with going over reviews like this is that it does not care about the order of words, number of words, or punctuation. This is how we made the words able to be processed by a DNN. The model for this preprocessed data is a sequential neural network where there are 3 layers that have l1, l2 and dropout for regularrization. And a final sigmoid node that decides whether the review is a positive or negative review.

# MNIST Dataset

This dataset is a compilation of 60000 hand drawn numbers from 0 to 9. What I did with this project was to combine some of the numbers together. For example, 0 and 1, 2 and 3, so on. The model that I used for this test was with a sequential neural network that has two layers an input of a 1D array of the picture and an output of the softmax of the five possible mixtures.

# Cats and Dogs Dataset

This project loads the cats and dogs dataset from the tensorflow datasets. The model is a convolutional neural network that takes the 150 x 150 rgb images and does data augmentation to allow the dataset to seem larger than it is. It also uses l2 and dropout for regularization.
