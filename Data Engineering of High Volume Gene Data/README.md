# Introduction
Here a unlabeled gene expression data is used with 13177 rows and 13166 columns. The goal is to divide the genes into different clusters.
# Data Preprocessing
The dataset has 13166 features, considering all the features to build a model is not desirable as it will not result in best predictions and accuracy, training all the 13166 features will lead to a model with vague knowledge in the dataset. Therefore reducing the number of features is the only best way to fix them all. .  In the project we have reduced features in two steps, one is by writing a custom function, then by passing this reduced data to further decomposition using FastICA and from there we were able to get the desired number of reduced features to get the best representation of the entire dataset.
# Custom Function for Data Cleaning
The custom function takes up all the features and reduces the features by taking the average of every feature and selecting only the features that has the average value of three(3) or more than three(3) i.e, >=3, and all the other features with average less than that is completely dropped. 
# Model
The model chosen to train the data is sklearn K-Means clustering model. K is the number of groups that we think the data should be divided into. For this project, the number of clusters are set to 16. 
# Model Internal Evaluation
We used Silhouette as our internal evaluation metric. The silhouette score is used to calculate the cluster quality which is created by the K-Means algorithm. Silhouette score ranges from [–1,1]. If the score is negative, then the clusters are not well assigned. Zero score indicates overlapping of clusters.
# Model Final Evaluation
The final evaluation of the model is done by calling an API which is mentioned in the code.
