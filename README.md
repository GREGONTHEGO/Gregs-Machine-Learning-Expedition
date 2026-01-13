# Machine Learning using ML Libraries
This repository is a curated collection of my machine learning projects using various datasets and model architectures. Each project explores a different type of problem or modeling approach, helping me understand the right architecture for a given dataset.

## Computer Vision
* **Cats vs Dogs (ResNet50):** Used **ResNet50** as a frozen base model with custom head layers. Later experiments unfroze some ResNet layers and fine-tuned with a smaller learning rate to improve classification accuracy.
* **CIFAR-10 (CNN Tuning):** Experimented with a two-layer CNN, varying filter counts (16-128), kernel sizes (3-9), and padding. Found that larger filter counts and kernel size 5 performed best, while padding of 0 underperformed significantly.
* **Pets Dataset (Segmentation):** Visualized random images from the Pets dataset alongside their corresponding trimaps (one color for background, one for foreground, and one for not-classified). This setup supports later segmentation tasks and mask predictions.

## Natural Language Processing (NLP)
* **IMDB (Sentiment Analysis):**
    * **Task:** Classify movie reviews as positive or negative based on text content.
    * **Data Processing:** Vectorized the text using the **10,000 most frequent words** (Bag-of-Words approach).
    * **Architecture:** Built a deep, highly regularized network with **3 hidden layers** (64 neurons each).
    * **Regularization:** Applied both **L1/L2 regularization** and aggressive **Dropout (40%)** to every layer. This was crucial to prevent the model from overfitting on the sparse, high-dimensional input data.

## Experimental & Optimization
* **MNIST (Hybrid Digit Classification):**
    * **Concept:** A custom experiment to test model robustness on overlapping features.
    * **Data Engineering:** I manually created a new dataset by **averaging the pixel values** of digit pairs (merging 0s with 1s, 2s with 3s, etc.) to form 5 new "hybrid" classes.
    * **Goal:** To see if a standard neural network could disentangle and classify these superimposed images where features from two digits exist simultaneously.
* **Linear Programming Optimization (Project 1):**
    * **Method:** An exploration of mathematical optimization using `scipy.optimize`.
    * **Implementation:** Wrote a script to generate random generator matrices ($G$) and used **Linear Programming** to calculate specific metrics ($h_m$) across thousands of configurations ($n, k, m$).
    * **Technical Detail:** Involved setting up complex constraint matrices (upper bounds, equality constraints) to iteratively solve for optimal vectors in a geometric space.

## Real-World & Time Series
* **Temperature Forecasting (Climate Data):**
    * **Dataset:** Jena Climate dataset (2009-2016).
    * **Model:** Uses Recurrent Neural Networks (RNN/LSTM) to predict future temperatures based on historical time-series data.
    * **Results:** Validated by comparing predicted vs. actual temperatures (e.g., *Predicted: 12.10 °C, Actual: 7.38 °C*) to test temporal trend capture.
* **Hospital Dataset (Mortality Prediction):**
    * **Feature Engineering:** Cleaned 10 selected columns and trained a neural net (~50% accuracy).
    * **Full Pipeline:** Converted all 47 columns (text to numeric), applied standard scaling, and built a neural network with enhanced feature representation (~92% accuracy).
