'''
Naive Bayes Classifier Module
TODO FINISH THIS CLASSIFIER
'''

import re

class naiveBayes():

    # ctor
    def __init__(self):
        return


    # Function to tokenize given string
    def tokenize(self, string):
        string = string.lower()
        tokens = re.findall("[a-z0-9]+", string)

        return set(tokens)

