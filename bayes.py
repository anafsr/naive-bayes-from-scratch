
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:11:23 2019

@author: ana fisher
"""

import pandas as pd
import numpy as np
import math
import os 

#we will use sklearn and scipy to track accuracy of custom calculations

from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


"""
Implementation of Bayes class
Class is initialized with training and test sets
For each outcome cathegory class implements method to calculate prior probabilities from from the Train Set
For each outcome cathegory class implements method to calculate likelihood using norm_pdf_multivariate method
Class implements method to calculate naive Bayess predictions for the test data
Class implements method to calculate confusion matrix and metods to calculate classifier performance
"""

class Bayes:
    def __init__(self, X_train, Y_train):
        
        self.X_train = X_train
        self.Y_train = Y_train
#        self.X_test = X_test
#        self.Y_test = Y_test
        self.probability = []
        self.prediction = []
        
    def mean_var_calculate(self, data):
        
        n_points = data.shape[0]
        n_features = self.X_train.shape[1]
        
        mu = data.sum(axis=0)/n_points
        sigma = np.zeros((n_features, n_features))
        
        x_mu = data - mu.T
        
        for i in range(n_features):
            sigma[i, i] = np.dot(x_mu[:, i].T, x_mu[:, i])
            sigma[i, i] /= n_points
            
        return mu, sigma
    
    """
    Multivariate normal pdf calculation
    Input: x = feature vector
           mu = mean of the data
           sigma = std of the data
    """
    
    def norm_pdf_multivariate(self, x, mu, sigma):
    
        dim = len(x)
        
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        
        norm_const = 1.0/( math.pow((2*math.pi),float(dim)/2) * math.pow(det,1.0/2) )
        x_mu = x - mu
        result = math.pow(math.e, -0.5 * ((np.dot(np.dot(x_mu, inv ), x_mu.T))))
        
        return norm_const * result
    
    """
    Calculate prior probabilites
    Output: dictionary in the format {class, propability}
    """
    def get_priors(self): 
        
        labels, counts = np.unique(self.Y_train, return_counts=True)
        
        output = {}
        
        for i, label in enumerate(labels):
            output[int(label)] = float(counts[i])/len(self.Y_train)
        return output  
    
    """
    Calculate likelihood of feature vector x, given class label
    Input: x = feature vector
           label = output class
    Output: probability x conditioned on label
    """
    
    def calculate_likelihood(self, x, label):
        
        sub_array = np.where(self.Y_train == label)
        mu, sigma = self.mean_var_calculate(self.X_train[sub_array])
        
        likelihood = self.norm_pdf_multivariate(x, mu, sigma)
        return likelihood
    
    """
    Classifier function
    Outputs a vector of class predictions that has the same dimension 
    as a vector of class outcomes in test data
    """
    
    def naive_bayes_classifier(self, X_test):
        
        assert X_test.shape[1] == self.X_train.shape[1], "Number of features doesn't match train set!"

    
        prior = self.get_priors(); 
        possible_labels = list(prior.keys());
        self.prediction = np.zeros(len(X_test))
        self.probability = np.zeros(len(X_test))
            
        for i, x_test_vector in enumerate(X_test):
#            prediction = 0
            evidence = sum(self.calculate_likelihood(x_test_vector, label)*prior[label] \
                                for label in possible_labels)
            for label in possible_labels:
                posterior = self.calculate_likelihood(x_test_vector, label)*prior[label]   \
                            / evidence 
                if posterior > self.probability[i]:
                    self.probability[i] = posterior
                    self.prediction[i] = label
        return self.prediction
    
    
    def plot_prob_hist(self, Y_test):
        
        for label in np.unique(self.Y_train):
            idx = np.where(Y_test == label)
            plt.hist(x=self.probability[idx], bins=math.floor(len(self.probability)/10), 
                     label = label, alpha=0.7, rwidth=0.85)        
        
        plt.grid(axis='y', alpha=0.75)
        plt.legend(loc='upper left')
        plt.xlabel('P of prediction'); plt.ylabel('Frequency')
        plt.title('Probability Frequency Plot')
        
    
    def calc_confusion_matrix(self, Y_pred, Y_test): #confusion matrix
   
        assert len(Y_pred) == len(Y_test), "Vectors of predictions and outcomes should be the same length!"
        
        conf_mtrx = np.empty([2, 2], dtype = int)
        
        TN = Y_pred[(Y_pred == Y_test) & (Y_test == 0)]
        FP = Y_pred[(Y_pred !=Y_test) & (Y_test == 0)]
        FN = Y_pred[(Y_pred != Y_test) & (Y_test == 1)]
        TP = Y_pred[(Y_pred == Y_test) & (Y_test == 1)]
        
        conf_mtrx[0,0] = len(TN); conf_mtrx[0,1] = len(FP)
        conf_mtrx[1,0] = len(FN); conf_mtrx[1,1] = len(TP)
        
        return conf_mtrx
    
    def calc_accuracy(self, Y_pred, Y_test): #Acuracy
        
        conf_mtrx = self.calc_confusion_matrix(Y_pred, Y_test)
        
        return (conf_mtrx[0,0] + conf_mtrx[1,1])/ np.sum(conf_mtrx)
    
    def calc_error(self, Y_pred, Y_test): #Error rate = 1-accuracy
        
        conf_mtrx = self.calc_confusion_matrix(Y_pred, Y_test)
        
        return (conf_mtrx[0,1] + conf_mtrx[1,0])/ np.sum(conf_mtrx)
    
    def calc_sens(self, Y_pred, Y_test): #Sensitivity
        
        conf_mtrx = self.calc_confusion_matrix(Y_pred, Y_test)

        return conf_mtrx[1,1]/(conf_mtrx[1,0] + conf_mtrx[1,1])
    
    def calc_spec(self, Y_pred, Y_test): #Specificity
        
        conf_mtrx = self.calc_confusion_matrix(Y_pred, Y_test)
        
        return conf_mtrx[0,0]/(conf_mtrx[0,0] + conf_mtrx[0,1])


if __name__ == "__main__":

    #Define a helper function to load the data
    def data_loader(path): 

        data = pd.read_csv(path, header=None, engine='python').values
        feature = data[:, :-1]
        outcome = data[:, -1]
        
        return feature, outcome
    
    path = os.getcwd()

    X_train, Y_train = data_loader(os.path.join(path, "data/train.csv"))
    X_test, Y_test = data_loader(os.path.join(path, "data/test.csv"))
   
    
    
    print("Prediction using custom Bayes classifier:\n")
    
    custom = Bayes(X_train, Y_train) #create custom Bayes classifier
    clf_predict=custom.naive_bayes_classifier(X_test) #predict values
    conf = custom.calc_confusion_matrix(clf_predict, Y_test) #calculate confusion matix
    custom.plot_prob_hist(Y_test)
    print('Accuracy score (ability rate of correctly classifying cases in general): ', custom.calc_accuracy(clf_predict, Y_test))
    print('Error score: (failure rate of correctly classifying cases in general)', custom.calc_error(clf_predict, Y_test))
    print('Sensitivity score (ability rate of correctly classifying cases with the disease): ', custom.calc_sens(clf_predict, Y_test))
    print('Specificity score: (ability rate of correctly classifying cases without the disease)', custom.calc_spec(clf_predict, Y_test))
    print('\nCustom confusion matrix:\n', conf)
        
    """
    Compare to sklearn
    """
    
    print("\nPredication metric using scikit-learn matches our custom classifier metric:\n")
    
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)

    print('Scikit-learn confusion matrix:\n', cm)
    print('Accuracy score scikit-learn: ', accuracy_score(Y_test, y_pred))
    
