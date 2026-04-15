**Binary Classification — Breast Cancer Dataset**

*A Learning Journal | PyTorch Logistic Regression*

# **Project Overview**

This notebook implements a binary classification model using PyTorch to predict whether a breast cancer tumour is malignant (0) or benign (1). The dataset used is sklearn's built-in Breast Cancer Wisconsin dataset — 569 samples, 30 numerical features, no missing values.

This is not just a finished project. It is a documented learning journey showing every mistake made, every fix applied, and every concept understood along the way.

# **Tech Stack**

* PyTorch — model definition, training loop, inference

* scikit-learn — dataset, train/test split, StandardScaler, confusion matrix

* torchmetrics — accuracy evaluation

* matplotlib — data visualisation

# **Model Architecture**

A single-layer logistic regression model built with nn.Linear (30 input features → 1 output) followed by torch.sigmoid() in the forward pass. Output is a probability between 0 and 1\. Threshold of 0.5 converts probability to class label.

* Loss function: nn.BCELoss()

* Optimizer: torch.optim.SGD (lr=0.01)

* Epochs: 1000

# **Final Results**

* Test Accuracy: 98.25%

* True Malignant (correct): 42 / 43

* True Benign (correct): 70 / 71

* False Positive: 1  |  False Negative: 1

Out of 114 test samples, only 2 were misclassified.

# **Mistakes Made & How I Fixed Them**

## **1\. Wrong shape index in nn.Linear**

Initial code used X\_train.shape\[30\] to get the number of input features. This threw an IndexError because shape only has 2 dimensions — (rows, cols). Index 30 does not exist.

* **Changed to X\_train.shape\[1\] — index 1 gives the number of columns (features).**Fix: 

## **2\. Missing sigmoid in forward()**

The forward() method initially returned raw linear output (e.g. \-3.2 or 7.8). BCELoss expects values between 0 and 1\. Without sigmoid, the loss function breaks.

* **Added torch.sigmoid() wrapping the linear layer output in forward().**Fix: 

* **sigmoid converts any real number to a probability. The threshold \> 0.5 then converts that probability to a class label. These are two separate steps.**Key insight: 

## **3\. Model not learning — loss frozen at 40.87 for all 1000 epochs**

The training loop ran correctly but loss never decreased. Every epoch printed the exact same loss value. This was the most confusing bug.

* **Feature scaling was missing. The breast cancer dataset has features at wildly different scales — some columns are in the 0.001 range, others exceed 1000\. SGD cannot navigate this landscape and gets stuck immediately.**Root cause: 

* **Applied StandardScaler from sklearn — fit on training data only, transform both train and test. This is also where I learned about data leakage: fitting the scaler on test data would be cheating.**Fix: 

## **4\. Shape mismatch error in BCELoss**

BCELoss threw a shape mismatch error during the training loop. The model output y\_pred had shape (N, 1\) but y\_train had shape (N,) — a 1D tensor.

* **Used .unsqueeze(1) on y\_train and y\_test to reshape from (N,) to (N, 1). Also added a guard (ndim \== 1 check) to prevent (N, 1\) becoming (N, 1, 1\) on re-runs.**Fix: 

## **5\. Sequential train/test split**

The original split used index slicing (X\[:train\_split\]) which takes the first 80% of rows as training data. This is wrong for most real datasets because the data may be ordered in some way, introducing bias.

* **Replaced with sklearn's train\_test\_split(X, y, test\_size=0.2, random\_state=42) for a proper random split.**Fix: 

## **6\. Misleading plot\_predictions function**

The original plot function was copied from a regression project. For a 30-feature classification dataset, plotting feature\[:, 0\] vs labels produces a meaningless scatter — you are only showing 1 out of 30 features with no real signal.

* **Removed the misleading plot. Replaced with a proper confusion matrix using sklearn's ConfusionMatrixDisplay, and added a 2-feature scatter plot (feature 0 vs feature 1, coloured by class) for the pre-training data distribution.**Fix: 

## **7\. No accuracy metric**

The original training loop only tracked loss. Loss alone does not tell you how many predictions are actually correct — a model can have low loss but still misclassify many samples.

* **Added torchmetrics Accuracy(task='binary') after generating y\_preds\_labels. Final accuracy: 98.25%.**Fix: 

# **Key Concepts Understood**

* Why BCELoss needs sigmoid: raw numbers are not probabilities. BCELoss assumes inputs are in \[0,1\].

* sigmoid vs threshold: sigmoid gives a probability, threshold converts that to a class label. Two separate operations.

* Feature scaling: SGD is sensitive to feature scale. StandardScaler is essential for gradient-based optimisers on unscaled data.

* Data leakage: scaler must be fit only on training data. Fitting on test data leaks future information into training.

* unsqueeze(1): broadcasting rules require y\_train and y\_pred to have matching shapes for BCELoss.

* Random vs sequential split: sequential slicing can introduce ordering bias. Always use random split for real datasets.

* Confusion matrix over scatter plot: for classification tasks, a confusion matrix is far more informative than a scatter plot of one feature.

# **What I Would Do Differently Next Time**

* Apply StandardScaler before converting to tensors as the very first preprocessing step.

* Always check feature scales before training — a quick df.describe() would have caught the scaling issue immediately.

* Include accuracy metric from the start, not as an afterthought.

* Use sklearn train\_test\_split by default for all datasets.

*