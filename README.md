# CS452-Project-State-Farm-Distracted-Driver-Detection-
This GitHub repository contains the material related to our project  for the course of Deep learning for visual perception (CS452) at FAST (NUCES) university, Karachi.

## Project Members:

- Abdul Mannan *(K163620)* 
- Murtaza Multanwala *(K163618)*
- Radheem Razi *(K163645)*
- Shahyar *(K163750)*
- Saira *(K163665)*

**Project Report:**

[Report Link in repo ](DLP_project_StateFarm_Distracted_Driver_Detection.pdf)

## Overview
The Center for Disease Control and Prevention
(CDC) found that nearly one out of five car
accidents is caused by a distracted driver.
Unfortunately, this means around 425,000 individuals get injured and 3,000 deaths are noted
because of distracted driving each year. State Farm is a large group of insurance companies throughout the United States with corporate headquarters in Bloomington, Illinois.
Their initiative is to improve these disturbing
statistics, and better ensure their customers, by
examining whether dashboard cameras can automatically detect drivers participating in distracting practices. Provided a dataset of dashboard camera pictures, State Farm is challenging Kagglers to classify each driver’s behavior, for example, what drivers are doing, and
whether they are distracted.

*Link to the file in repository: [Dataset Visualization](Dataset%20Visualization.ipynb)*

The dataset used in this project was provided
by State Farm through a Kaggle competition. The dataset contains a total of 102150 images
split into a training set of 22424 images and
a testing set of 79726 images (640 × 480 full
color).
There are 10 classes including safe driving in the dataset:

* c0 Safe driving.
* c1 Texting (right hand).
* c2 Talking on the phone (right hand).
* c3 Texting (left hand).
* c4 Talking on the phone (left hand).
* c5 Operating the radio.
* c6 Drinking.
* c7 Reaching behind.
* c8 Hair and makeup.
* c9 Talking to passenger(s).

This dataset is available on Kaggle, under the State Farm competition: https://www.kaggle.com/c/state-farm-distracted-driver-detection


## Methodology
* [Transfer Learning and KNN implementation](Transfer%20Learning%20Models%20and%20KNN.ipynb)
* [Network Ensemble](Ensemble%20on%20Test%20set.ipynb)
* For generating predictions and Undersampled dataset training, different versions of Kaggle notebook: https://www.kaggle.com/k163665/to-detect-distracted-drivers


#### Transfer Learning
We initially explored creating our model from "scratch" but quickly realized our training set was limited. In the deep learning world, 20,000 images is a rather small dataset. After switching to transfer learning, we saw a dramatic improvement in model performance. We considered other pre-trained models such as VGG-19, MobileNet, Xception and ResNet-50. These models are usually trained on millions of images which helps especially when the training set is small. 
#### Network Ensemble
It was found that using an ensemble of different models yielded better results than using a
single model as over fitting is a major concern
in this problem. Thus, instead of relying on the
predictions of one single model, we averaged
the results of 4 of our models namely VGG-19,
Xception, MobileNet and ResNet50 to get the
final prediction values. By averaging different
models, the variance of the trained model can be
reduced and a lower loss can be achieved.
#### Temporal Context
Since the images are taken from a video clip,
there exist many similar pictures which should
belong to the same categories. The idea is that after the CNN models complete predictions for all testing images, for each
test image, we find its K most similar images
(including itself) based on pixel-wise L2 euclidean norm. Due to the large quantity of
images to be processed, we had to shrink the
original test images to 100 × 100 to shorten the
computation tie. To find these K neighbors,
K Nearest-neighbor classification based on K
Dimensional-tree is used as KNN is computationally expensive as compared to other methods.

## Result

After exploring various CNN models combined
with network ensemble and KNN, our best performing results were achieved by calculating
the weighted trimmed average of models predictions and finally calculated trimmed average
for predictions across its K Nearest Neighbors
with (K = 10).

The final LB score (log loss) we
got is 0.18110 which ranked 40 among 1438
total participants (top 2%) on Kaggle’s public
leaderboard.
![Final predictions on test set](pred%20on%20test.PNG)
