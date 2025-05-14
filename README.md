# Greg's Machine Learning Expedition
This is a compilation of all machine learning projects I have completed. The projects are organized by the dataset and further organized by the model used to sort them. The goal of this repo is to help me understand when looking at a dataset which architecture to used and also to learn more about what the various architectures are good at.

# Iris Dataset
The first project that I completed with the Iris dataset was to create and use a perceptron. A perceptron classifies inputs into discrete classes using a step function. In the Iris data case this was to split the Setosa and Virginica flowers by their sepal length and width. This is supervised.

The second project that I completed with this dataset was creating a Naive Bayes classification. This is a supervised learning approach that predicts the class based on its features by calculating the probability of each class given the features. This used and split the data for all three flowers and for all of the features.

The third project on this dataset was to create a linear support vector machine. This is used on data that can be split linearly and finds the best plane that has the largest margin. The reason for the largest margin to be important is so that it is more likely that the line holds true with tests.



# XY Dataset
The project completed on a bunch of xy datapoints was to find the line of best fit. This project introduced cleaning the dataset by getting rid of any values that had NaN. Linear Regression is a model that estimates the relationship between scalar response and one or more explanatory variables. This uses MSE as the loss function.

The next project was on the same dataset but it compares linear regression to ridge regression. Ridge regression is linear regression with an added regularization term which helps prevent overfitting and improves the model stability. I also run the ridge regression through a kFold cross validation. kFold uses the data such that the validation segment changes. In my case we had 10 folds which means that we have 10 different training and validation splits that the ridge regression is run on.

#CIFAR Dataset
The project was to create a two layer CNN and test different number of filters, size of kernels, and different amounts of padding. The number of filters that I tested was 16, 32, 64, 128, for which they consistently did better but they also took longer to run when there were more filters. The sizes of kernels tested was 3,5,7,9, for which a kernel size of 5 did the best and 9 did the worst with 3 and 7 looking similar. Now, I do understand that with bigger kernel size we should use a bigger amount of padding however, the test above was to just see what the difference does. The last set of tests for padding 0,1,2,3, shows that 0 does much worse than the others and 1,2, and 3 are all very similar.
