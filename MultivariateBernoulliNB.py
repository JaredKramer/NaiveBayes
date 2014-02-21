#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Jared Kramer & TJ Trimble

This script is my implementation of a Multivariate Bernoulli NB classifier. It takes training
and test files (formatted as in the accompanying samples), class prior and conditional
probability values used in smoothing, and outputs a model file, system output and prints
a confusion matrix to stdout.

"""

import sys
import time
from numpy import sum
from numpy import log10 as log
from collections import defaultdict

start_time = time.time()

if len(sys.argv) > 1:
    # Command line arguments
    training_data, testing_data, class_prior_delta, cond_prob_delta, model_file, sys_output = sys.argv[1:]
else:
    # Debugging arguments
    training_data = "../examples/train.vectors.txt"
    testing_data = "../examples/test.vectors.txt"
    class_prior_delta = 0.1 # for smoothing
    cond_prob_delta = 0.1 # for smoothing
    model_file = "model_file_NB1"
    sys_output = "sys_output_NB1"

# Cast for use with shell scripts
class_prior_delta = float(class_prior_delta)
cond_prob_delta = float(cond_prob_delta)

#Load in data and initialize
training_data = open(training_data, 'r').read().strip().split("\n")
testing_data = open(testing_data, 'r').read().strip().split("\n")

class_counts = defaultdict(int)
word_counts = defaultdict(lambda: defaultdict(int))
cond_probs = defaultdict(lambda: defaultdict(int))
class_probs = defaultdict(int)
doc_count = 0
every_train_word = set()

### Training
#get word and class counts
for vector in training_data:
    doc_count += 1
    c = vector.split()[0]
    class_counts[c] += 1
    for w in vector.split()[1:]:
        word_counts[c][w.split(":")[0]] += 1

every_train_word = {k for c in word_counts for k in word_counts[c]}
labels = set(class_counts.keys())

# Get conditional probabilities
for c in labels:
    denominator = float((2*cond_prob_delta) + class_counts[c])
    for w in every_train_word:
        cond_probs[c][w] = (cond_prob_delta + word_counts[c][w]) / denominator

# get prior probabilities
for c in labels:
    class_probs[c] = log((class_prior_delta + class_counts[c]) / ((len(class_counts)*class_prior_delta) + doc_count)) # this is the log of the class prob

### Testing
def test(data, matrix, test_or_train):
    doc_index = 0
    correct = 0
    for vector in data:
        true_label = vector.split()[0]
        tups = []
        for c in labels:
            second_term = 0
            for w in vector.split()[1:]:
                word = w.split(":")[0]
                if word in every_train_word:
                    prob = cond_probs[c][word]
                    if prob != 0:
                        second_term += log(prob / (1 - prob))
            tups.append((c, float(class_probs[c] * second_term * third_term[c])))

        sys_output.write("".join([test_or_train, str(doc_index), "\t", true_label]))
        max_val = max(tups, key=lambda item: item[1])
        difference = float(max_val[1])
        projected_label = max_val[0]
        # Sorted by value
        tups = sorted([(c, float(p-difference)) for c, p in tups], key=lambda item: item[1], reverse=True)
        values = [pow(10, tup[1]) for tup in tups]
        denominator = sum(values)
        for i in range(len(tups)):
            sys_output.write("\t".join(["", tups[i][0], str(values[i]/denominator)]))
        sys_output.write("\n")
        doc_index += 1
        matrix[true_label][projected_label] += 1
        if projected_label == true_label: correct += 1

# Initialize
third_term = defaultdict(float)
for c in labels:
    third_term[c] = sum([log(1-cond_probs[c][w]) for w in every_train_word])

train_confusion_matrix = defaultdict(lambda: defaultdict(int))
test_confusion_matrix = defaultdict(lambda: defaultdict(int))

# Begin testing
sys_output = open(sys_output, 'w')
test(training_data, train_confusion_matrix, "train")
print
test(testing_data, test_confusion_matrix, "test")
sys_output.close()

def print_matrix(matrix, test_or_train):
    print "".join(["class_num=", str(len(class_counts)), "feat_num=", str(len(every_train_word))])
    print "".join(["Confusion matrix for the ", test_or_train, "ing data:"])
    print "row is the truth, column is the system output\n"
    total_docs_classified = 0
    docs_classified_correctly = 0
    for label1 in labels:
        print label1,
        for label2 in labels:
            result = matrix[label1][label2]
            print result,
            if label1 == label2:
                docs_classified_correctly += result
            total_docs_classified += result
        print
    print "".join(["\n", test_or_train, "ing accuracy = ", str(float(docs_classified_correctly)/total_docs_classified), '\n\n'])

print_matrix(train_confusion_matrix, "train")
print_matrix(test_confusion_matrix, "test")

model_file = open(model_file, 'w')
model_file.write("%%%%% prior prob P(c) %%%%%\n")
for c in class_counts:
    model_file.write(" ".join([c, str(float(class_counts[c]) / doc_count), str(log(float(class_counts[c]) / doc_count)) + "\n"]))

model_file.write("%%%%% conditional prob P(f|c) %%%%%")
for c in cond_probs:
    model_file.write("".join(["%%%%% conditional prob P(f|c) c=", c, "%%%%%\n"]))
    for w in cond_probs[c]:
        model_file.write("\t".join([w, c, str(cond_probs[c][w]), str(log(cond_probs[c][w])), "\n"])) #this is failing because of the -inf values
model_file.close()

print time.time() - start_time, "seconds"
