NaiveBayes
==========

Naive Bayes Text Classifier

This repository contains two Naieve Bayes text classifiers.

The first is a Multi-Variate Bernoulli NB classifier and the other is a Multinomial NB classifier.

Description of how each algorithm works.

Results

Both make an assumption of conditional independence.

the training and testing data must be formatted as in the accompanying examples, though changing the code to fit other data formats would be trivial in most cases.

Usage:


kNN
===

k-Nearest-Neighbors Text Classifier
Jared Kramer

This script takes training and test files (formatted as in the accompanying samples), a k-value that specifies the desired number of neighbors, a similarity value that specifies the similarity measure (1 for Euclidean, 2 for Cosine), and the name of an output file. The confusion matrix is printed to stdout.

The classifier works by calculating the distances between a given test instance and every training instance using the specified similarity metric.  Note that Euclidean distance here is actually a dissimilarity measure, where as Cosine is a similarity measure. The labels of k nearest training neighbors each cast a vote and the test instance is assigned the label with the most votes.

As noted above, the training and testing data must be formatted as in the accompanying examples, though changing the code to fit other data formats would be trivial in most cases.

This code classifies 900 test instances in approximately 2 minutes with accuracies in the 60-70 range depending on metric and k-value. The running time is all testing and the training step for kNN is non-existant.



Usage: The command line arguments are as follows: 
1 = training_data, 
testing_data, 
class_prior_delta, 
cond_prob_delta, 
model_file, sys_output = sys.argv[1:]


