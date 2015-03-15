
# coding: utf-8

# In[68]:

import csv
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tools as st

# read and in the data and show summary statistics
data = pd.read_csv('/home/vagrant/repos/datasets/admission_data.csv')
print "\n****DESCRIPTION OF THE DATA****"

print data.describe()

#print data['gre'].hist()

# show histogram of the data
#data.hist()
#pl.show()

plt.subplots_adjust(wspace=1.6)


## Analysis of the gre variable (not very high discrimination power)

# In[69]:

# --------------Analysis of the gre variable
fig1 = plt.figure('gre analysis', figsize=(18,18))

# scatter plot of all gre data, irrespective if student was addmitted or not
gre_vs_admit_ax = fig1.add_subplot(2,1,1)
gre_vs_admit_ax.set_title("GRE scores vs admit")
gre_vs_admit_ax.set_xlabel("GRE scores")
gre_vs_admit_ax.set_ylabel("Admitted")
plt.scatter(data['gre'], data['admit'], s=100)

# gre distribution broken down for admitted and not admitted students
gre_vs_admit_yes_no_ax = fig1.add_subplot(2,1,2)
gre_vs_admit_yes_no_ax.set_title("GRE score distributions for admitted and not admitted")
gre_vs_admit_yes_no_ax.set_xlabel("GRE scores")
plt.hist(data[data['admit'] > 0]["gre"].as_matrix(), bins=20, color='r', alpha=0.3) #as_matrix returns the numpy array representation
plt.hist(data[data['admit'] < 1]["gre"].as_matrix(), bins=20, color='k', alpha=0.3) 


# In[69]:




## Analysis of the gpa variable (gpa distribution of admitted students has higher mean)

# In[70]:

# --------------Analysis of the gpa variable
fig2 = plt.figure('gpa analysis', figsize=(18,18))

# scatter plot of all gpa data, irrespective if student was addmitted or not
gpa_vs_admit_ax = fig2.add_subplot(2,1,1)
gpa_vs_admit_ax.set_title("GPA vs admitted")
gpa_vs_admit_ax.set_xlabel("GPA")
gpa_vs_admit_ax.set_ylabel("Admitted")
plt.scatter(data['gpa'], data['admit'], s=100)

# gpa distribution broken down for admitted and not admitted students
gpa_vs_admit_yes_no_ax = fig2.add_subplot(2,1,2)
gpa_vs_admit_yes_no_ax.set_title("GPA distributions for admitted and not admitted")
gpa_vs_admit_yes_no_ax.set_xlabel("GPA")
plt.hist(data[data['admit'] > 0]["gpa"].as_matrix(), bins=20, color='r', alpha=0.3) #as_matrix returns the numpy array representation
plt.hist(data[data['admit'] < 1]["gpa"].as_matrix(), bins=20, color='k', alpha=0.3) 


## Analysis of the prestige variable: discovered a much higher probability of admission for prestige==1 

# In[71]:

# --------------Analysis of the prestige variable
fig3 = plt.figure('prestige analysis', figsize=(18,18))

# scatter plot of all prestige data, irrespective if student was addmitted or not
prestige_vs_admit_ax = fig3.add_subplot(2,2,1)
prestige_vs_admit_ax.set_title("prestige vs admit")
prestige_vs_admit_ax.set_xlabel("prestige")
prestige_vs_admit_ax.set_ylabel("admission result")
plt.scatter(data['prestige'], data['admit'], s=100)

# unique values of prestige: 1, 2, 3, 4
unique_prestige = data['prestige'].unique() # unique values, type is ndarray
unique_prestige_list = sorted(unique_prestige.tolist()) # convert it to sorted list

# for each prestige score, sum all the times there was admission
num_admitted_by_prestige = []
for prestige in unique_prestige_list:
    num_admitted_by_prestige.append( (data[data['admit'] > 0]['prestige'] == prestige).sum() * 1.0 )

# for each prestige score, sum all the times there was *no* admission
num_not_admitted_by_prestige = []
for prestige in unique_prestige_list:
    num_not_admitted_by_prestige.append( (data[data['admit'] < 1]['prestige'] == prestige).sum() * 1.0 )

# calculate probability of admission by prestige number (these probabilities don't add up to 1, they belong to different groups)
admitted_probability_by_prestige = []
for prestige in range(0,len(unique_prestige_list)):
    num_total = num_admitted_by_prestige[prestige] + num_not_admitted_by_prestige[prestige]
    admitted_probability_by_prestige.append(num_admitted_by_prestige[prestige]/num_total)

# bar chart of admitted and not admitted grouped by prestige number
ind = np.array(unique_prestige_list)   # the x locations for the groups
width = 0.35      # the width of the bars
prestige_vs_admit_yes_no_ax = fig3.add_subplot(2,2,2)
prestige_vs_admit_yes_no_ax.set_title("Number of admitted vs not admitted for each prestige group")
prestige_vs_admit_yes_no_ax.set_xlabel("prestige")
prestige_vs_admit_yes_no_ax.set_xticks(ind)
rects1 = prestige_vs_admit_yes_no_ax.bar(ind, num_not_admitted_by_prestige, width, color='y' )
rects2 = prestige_vs_admit_yes_no_ax.bar(ind+width, num_admitted_by_prestige, width, color='r' )
prestige_vs_admit_yes_no_ax.legend( (rects1[0], rects2[0]), ('Not admitted', 'Admitted') )

# bar chart of probability of admission as a function of prestige number
prestige_probability_ax = fig3.add_subplot(2,2,3)
prestige_probability_ax.set_title("Probability of admission for each prestige group")
prestige_probability_ax.set_xlabel("prestige")
prestige_probability_ax.set_ylabel("probability of admission")
prestige_probability_ax.set_xticks(ind)
rects3 = prestige_probability_ax.bar(ind, admitted_probability_by_prestige, width, color='y' )




## Logistic regression

### 1. Based on the above we will drop the GRE variable. Even if we keep it, it has absolutely no explanatory value based on the results of the fit  

### 2. Unlike the example in http://blog.yhathq.com/posts/logistic-regression-and-python.html, we will discard prestige_4 instead of prestige_1, because it has a lot of explanatory power

# In[72]:

# this is just pandas notation to get columns 1...n
# we want to do this because our input variables are in columns 1...n
# while our target is in column 1 (0=not admitted, 1=admitted)
#training_columns = data.columns[1:]

# dummify rank
dummy_ranks = pd.get_dummies(data['prestige'], prefix='prestige')
#print dummy_ranks.head()

# create a clean data frame for the regression, join the dummy prestige ranks
cols_to_keep = ['admit', 'gpa']
regression_data = data[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_1': 'prestige_3'])

# Add intercept or use the tookit function to add a constant
#regression_data['intercept'] = 1.0
regression_data = st.tools.add_constant(regression_data, False) # equivalent to adding intercept as above

print "after adding constant:\n", regression_data.head()

train_cols = regression_data.columns[1:]
print "\n\n**Training columns:\n", train_cols
print "\n\n**Regression data:\n", regression_data[train_cols].head()

# Create a Logit model
logit = sm.Logit(regression_data['admit'], regression_data[train_cols])
 
# Fit the model
result = logit.fit()

# Print results
print result.summary()

# odds ratios only
print "\n\n**Odds ratios:\n", np.exp(result.params)

# look at the confidence interval of each coeffecient
print "\n\n**Confidence intervals:\n", result.conf_int()


def predict(result, gpa, prestige):
    """
    Outputs predicted probability of admission to graduate program
    given gre, gpa and prestige of the institution where
    the student did their undergraduate
    """
    vec = []
    if prestige == 1: vec = [gpa,1,0,0,1]
    elif prestige == 2: vec = [gpa,0,1,0,1]
    elif prestige == 3: vec = [gpa,0,0,1,1]
    #####elif prestige == 4: vec = ???         
    return result.predict(vec)[0]

print "\nPrediction for GPA: 3.59, and Tier 3 Undergraduate degree is..."
print predict(result, 3.59, 3)

print "\nPrediction for GPA: 3.59, and Tier 1 Undergraduate degree is..."
print predict(result, 3.59, 1)

print "\nPrediction for GPA: 3.0, and Tier 1 Undergraduate degree is..."
print predict(result, 4.0, 3)


## Define helper cartesian function for plotting the results

# In[73]:

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


## Generate cartesian combinations of the training set

# In[74]:

gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 100)
print "\n\n**gpas:\n", gpas
print "\n\n**gpas size:\n", gpas.size

# array([ 2.26      ,  2.45333333,  2.64666667,  2.84      ,  3.03333333,
#         3.22666667,  3.42      ,  3.61333333,  3.80666667,  4.        ])
 


# In[86]:

# enumerate all possibilities
cartesian_combo = pd.DataFrame(cartesian([gpas, [1, 2, 3, 4], [1.]]))

# recreate the dummy variables
cartesian_combo.columns = ['gpa', 'prestige', 'const']
print "\n\n**Cartesian combo:\n", cartesian_combo.head()

dummy_ranks = pd.get_dummies(cartesian_combo['prestige'], prefix='prestige')
dummy_ranks.columns = ['prestige_1', 'prestige_2', 'prestige_3', 'prestige_4']
print "\n\n**Cartesian dummy ranks:\n", dummy_ranks.head()

# keep only what we need for making predictions
cols_to_keep = ['gpa', 'const']
cartesian_combo = cartesian_combo[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_1':'prestige_3'])

print "\n\n**Cartesian combo with dummy ranks and prestige 1,2,3:\n", cartesian_combo.head()
 
print "\n\n**Train columns:\n", train_cols

print "\n\n**Cartesian columns:\n", cartesian_combo.columns

# make predictions on the enumerated dataset
cartesian_combo['admit_pred'] = result.predict(cartesian_combo[train_cols])

print "\n\n Cartesian combo with admission probabilities:\n", cartesian_combo.head()


## Plot the results

# In[103]:

fig4 = plt.figure('Analysis of results', figsize=(10,10))

# gpa distribution broken down for admitted and no
prob_admit_vs_gpa_ax = fig4.add_subplot(1,1,1)
prob_admit_vs_gpa_ax.set_title("Probability of admission vs GPA score for prestige=1,2,3 schools")
prob_admit_vs_gpa_ax.set_xlabel("GPA")
prob_admit_vs_gpa_ax.set_ylabel("Probability of admission")

prestige_1_series = cartesian_combo['prestige_1'] == 1
prestige_2_series = cartesian_combo['prestige_2'] == 1
prestige_3_series = cartesian_combo['prestige_3'] == 1

p1 = plt.scatter(cartesian_combo[prestige_1_series]["gpa"], cartesian_combo[prestige_1_series]["admit_pred"], s=50, color='r')
p2 = plt.scatter(cartesian_combo[prestige_2_series]["gpa"], cartesian_combo[prestige_2_series]["admit_pred"], s=50, color='g')
p3 = plt.scatter(cartesian_combo[prestige_3_series]["gpa"], cartesian_combo[prestige_3_series]["admit_pred"], s=50, color='b')

prob_admit_vs_gpa_ax.legend( (p1, p2, p3), ('prestige_1', 'prestige_2', 'prestige_3') )


# In[99]:




# In[ ]:



