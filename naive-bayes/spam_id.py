#!/usr/bin/env python

from __future__ import print_function
import csv
import random
import nltk
import time
import re

#DATA_FILE="/home/Development/datascience/repos/datasets/sms_spam_or_ham.csv"
DATA_FILE="/home/vagrant/repos/datasets/sms_spam_or_ham.csv"

DEBUG_FEATURES = False
PRINT_CLASSIFIER_ERRORS = False
TRAIN_SET_PERCENTAGE = 0.2

TRIGGER_WORDS = \
'''valued customer customers complimentary urgent offer offers award awards win won winner guaranteed selected claim \
claims code codes draw wkly weekly upgrade upagrades loyal loyals term terms condition conditions expire expires expired alert alerts charge charges horny sex naked www http
'''
TRIGGER_WORDS_LIST = TRIGGER_WORDS.split()

PROMPT_WORDS_REGEX = "call|send|text|txt"

POUND_SIGN = u'\u00A3'
DOLLAR_SIGN = "$"

def tokenize(text):
    return text.lower().split()

def remove_special_chars(text, delimiters):
    regex = "[" + re.escape(delimiters) + "*]"
    clean_text = re.sub(regex, ' ', text)
    return clean_text

def contains_phone_number(text):
    # Not very robust, works on cleaned text only
    return re.search('([0-9]{3,8}\s*)+', text)

def contains_text_call_prompt(text):
    if re.search(PROMPT_WORDS_REGEX, text.lower(), re.IGNORECASE) and contains_phone_number(text):
        return True
    return False

def contains_price(text):
    regex = re.escape(POUND_SIGN) + "|" + re.escape(DOLLAR_SIGN)
    return re.search(regex, text)

def contains_coupon(text_list):
    #..9f...
    #..f9...
    for word in text_list:
        if re.search('[a-z]*[0-9][a-z]', word):
            return True
    return False

def contains_trigger_word(text_list, trigger_words_list):
    for word in text_list:
        for trigger_word in trigger_words_list:
            if trigger_word == word:
                if DEBUG_FEATURES: print("Trigger word detected:", trigger_word, " in word:", word)
                return True
    return False

def spam_features(text):
    clean_text = remove_special_chars(text,",-!:./")
    tokenized_text = tokenize(clean_text)
    features_dict = {}
    features_dict["has_price"] = True if contains_price(clean_text) else False
    features_dict["has_prompt"] = True if contains_text_call_prompt(clean_text) else False
    features_dict["has_coupon"] = True if contains_coupon(tokenized_text) else False
    features_dict["has_spam_word"] = True if contains_trigger_word(tokenized_text, TRIGGER_WORDS_LIST) else False
    
    return features_dict

def get_feature_sets(data_file):
    f = open(data_file, 'rb')

    # Read in the rows from the csv file
    rows = []
    for row in csv.reader(f):
        rows.append(row)
    
    output_data = []

    for row in rows:
        if DEBUG_FEATURES: print("ROW:", row)
        if len(row) != 2:
            continue
        # row[0] is the label, either 0 or 1 and the rest is the msg
        #msg_data = row.split(",")
        msg_label = 1 if row[0] == "spam" else 0
        msg_body  = row[1]
        
        msg_feature_dict = spam_features(msg_body)

        if DEBUG_FEATURES: print("FEATURES:",msg_feature_dict)

        # add the tuple of feature_dict, label to output_data
        data = (msg_feature_dict, msg_label)
        output_data.append(data)

        if DEBUG_FEATURES: print("DATA:", data, "\n")
        
    # close the file
    f.close()
    return output_data


def get_training_and_validation_sets(feature_sets):
    random.shuffle(feature_sets)
    
    count = len(feature_sets)
    
    slicing_point  = int(TRAIN_SET_PERCENTAGE * count)

    # the training set will be the first segment
    training_set = feature_sets[:slicing_point]

    # the validation set will be the second segment
    validation_set = feature_sets[slicing_point:]
    return training_set, validation_set

def run_classification(training_set, validation_set):
    # train the NaiveBayesClassifier on the training_set
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # let's see how accurate it was
    accuracy = nltk.classify.accuracy(classifier, validation_set)
    print("***The accuracy was.... {}".format(accuracy))
    return classifier

def predict(classifier, new_message):
    """
    Given a trained classifier and a fresh data point (a message),
    this will predict its label, either 0 or 1.
    """
    return classifier.classify(spam_features(new_message))

def find_classifier_errors(data_file, classifier):
    f = open(data_file, 'rb')
    total = 0.
    miss = 0.
    legit_msg_miss = 0.
    
    for row in csv.reader(f):
        if len(row) != 2:
            continue
        total += 1
        msg_label = 1 if row[0] == "spam" else 0
        msg_body  = row[1]
        msg_feature_dict = spam_features(msg_body)
        classification_result = classifier.classify(msg_feature_dict)
        if classification_result != msg_label:
            if PRINT_CLASSIFIER_ERRORS: print("Message misclassified:", row[0], msg_body, msg_feature_dict, "\n\n")
            miss += 1
            if row[0] == "ham":
                legit_msg_miss += 1
    f.close()
    print("Total misclassifications:", miss, " out of ", total, " messages")
    print("Real messages classified as spam: ", legit_msg_miss)
    print("Spam messages escaped filter: ", miss-legit_msg_miss)
    print("Misclassification ratio: ", (miss/total)*100., "%")
    print("Legitimate messages blocked ratio:", (legit_msg_miss/total)*100, "%")

    
def main():
    start_time = time.time()
    print("Let's use Naive Bayes!")
    data_file = DATA_FILE
    our_feature_sets = get_feature_sets(data_file)
    our_training_set, our_validation_set = get_training_and_validation_sets(our_feature_sets)
    print("Size of our data set: {}".format(len(our_feature_sets)))
    print("Size of our training set: {}".format(len(our_training_set)))
    print("Size of our validation set: {}".format(len(our_validation_set)))

    print("Now training the classifier and testing the accuracy...")
    classifier = run_classification(our_training_set, our_validation_set)

    end_time = time.time()
    completion_time = end_time - start_time
    print("It took {} seconds to run the algorithm".format(completion_time))

    print("Classifier errors:")
    find_classifier_errors(data_file, classifier)

    classifier.show_most_informative_features()
main()
