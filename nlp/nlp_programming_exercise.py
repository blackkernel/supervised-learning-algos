
# coding: utf-8

# In[46]:

'''
Write a function `unpunctuate` that takes a string and removes all punctuation, e.g.
unpunctuate("Hey there! How's it going?")
will output the following
"Hey there Hows it going"
'''
from collections import defaultdict
import re
DELIMITERS = ",!:.?'"
regex_punctuations=  "[" + re.escape(DELIMITERS) + "*]"
compiled_regex_punctuations = re.compile(regex_punctuations) # makes re.sub much faster

def unpunctuate(text):
    clean_text = re.sub(compiled_regex_punctuations, '', text)
    return clean_text.lower()
    
print unpunctuate("Hey there! How's it going???.")


# In[14]:

'''
Write a function `get_bag_of_words_for_single_document` that, given any strings (also called documents), e.g. "John also likes to watch football games.", returns its bag of words:

get_bag_of_words_for_single_document("John also likes to watch football games.")

# outputs the following. Hint: it doesn't have to be a dictionary exactly but could be an object that acts like a dictionary.
{
    'games.': 1,
    'to': 1,
    'football': 1,
    'watch': 1,
    'also': 1,
    'likes': 1,
    'John': 1
}
'''

def get_bag_of_words_for_single_document(doc):
    new_doc = unpunctuate(doc)
    l = new_doc.split(" ")
    bag = defaultdict(int)
    for word in l:
        bag[word] += 1
    return bag
        

print get_bag_of_words_for_single_document("John also likes to watch football games games.")


# In[15]:

'''
Write a function `get_bag_of_words` that uses the above function to achieve the following: 
given a list of strings, it returns the total bag of words
for all of the documents.

get_bag_of_words([
    "John likes to watch movies. Mary likes movies too.",
    "John also likes to watch football games.",
])

# ouputs the following:
{
    "John": 1,
    "likes": 2,
    "to": 3,
    "watch": 4,
    "movies": 5,
    "also": 6,
    "football": 7,
    "games": 8,
    "Mary": 9,
    "too": 10
}
'''

def get_bag_of_words(doc_list):
    total_bag = defaultdict(int)
    for doc in doc_list:
        single_bag = get_bag_of_words_for_single_document(doc)
        for word in single_bag:
            total_bag[word] += single_bag[word] 
    return total_bag

print get_bag_of_words([
    "John likes to watch movies. Mary likes movies too.",
    "John also likes to watch football games. And pandas",
    "John also likes to code in pandas.",
    "pandas"
])


# In[32]:

'''
Given a bag of words for all of the documents in our data set, write a function `turn_words_into_indices`
take the keys in the bag of words and alphabetize them,

e.g.
bag_of_words_data = {
    "John": 1,
    "likes": 2,
    "to": 3,
    "watch": 4,
    "movies": 5,
    "also": 6,
    "football": 7,
    "games": 8,
    "Mary": 9,
    "too": 10
}

turn_words_into_indices(bag_of_words_data)

# outputs the following:
["also", "football", "games", "John", "likes", "Mary", "movies", "to", "too", "watch"]
'''

def turn_words_into_indices(bag):
    indices = sorted([key for key in bag.keys()], key=str.lower)
    return indices

bag_of_words_data = {
    "John": 1,
    "likes": 2,
    "to": 3,
    "watch": 4,
    "movies": 5,
    "also": 6,
    "football": 7,
    "games": 8,
    "Mary": 9,
    "too": 10
}

print turn_words_into_indices(bag_of_words_data)


# In[45]:

'''
Given a document, write a function `vectorize` that turns the document into a list (also will be called a vector)
the same length as the number of keys of bag of words where for each index of the list will be 1 only if the word
at that index in the word list is contained in the document and 0 otherwise.

e.g.
Given this word list: ["also", "football", "games", "John", "likes", "Mary", "movies", "to", "too", "watch"]
Given this document: "The sun also rises"
vectorize("The sun also rises. Let's go to the movies")
# Since it contains 'also', 'to' and 'movies, the vectorize call will output the following
[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
'''
def vectorize_document_for_bag(doc, bag):
    doc_dict = defaultdict(int)
    for word in doc.split():
        doc_dict[word] += 1
    vector = [ 1 if doc_dict[word] > 0 else 0 for word in bag  ]
    return vector


vectorize_document_for_bag("The sun also rises. Let's go to the movies", ["also", "football", "games", "John", "likes", "Mary", "movies", "to", "too", "watch"])


# In[ ]:



