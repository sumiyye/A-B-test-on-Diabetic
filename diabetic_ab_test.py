# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Comparison of AB Test - Is Diabetes related with age ?
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Settings
# statsmodels library is needed to install. We are going to use functions from this library.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project Tasks
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Task 1: Preparing and Analyzing Data

df = pd.read_csv('datasets/diabetes.csv')

def check_df(dataframe, head=5):
    print('############### shapes ###############')
    print(dataframe.shape)
    print('############### Types  ###############')
    print(dataframe.dtypes)
    print('############### Head   ###############')
    print(dataframe.head(head))
    print('############### Tail   ###############')
    print(dataframe.tail(head))
    print('###############  NA   ################')
    print(dataframe.isnull().sum())
    print('############### Quantites #############')
    print(dataframe.describe().T)

check_df(df)
# Outcome result of 0 means no diabetes, and a score of 1 indicates diabetes.
# Look at the mean of age groupby outcome.

df.groupby('Outcome').agg({'Age':'mean'})

# there is a difference between mean of age but is this difference meaningful statistically ? 
# Done ab test to explain that .

# Task 2: Set up the Hypothesis.
# H0 : M1==M2 There is no difference whether have diabetic related with age.
# p-value < 0.05 H0 Reject
# H1 : M1!=M2 There IS difference whether have diabetic related with age.
# p-value > 0.05 H0 Can NOT Reject
# first of all we need to understand the data are whether normally distributed and have equal variances.
# for normal distribution test  === Shapiro test

test_stat, pvalue = shapiro(df.loc[df['Outcome'] == 0 , 'Age'])
print('test Stat = %.4f, p value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df['Outcome'] == 1 , 'Age'])
print('test Stat = %.4f, p value = %.4f' % (test_stat, pvalue))

### p-value is < 0.05 and HO reject.
### It means there is no normal distrubition.
### we do non-parametric Mann-Whitney U test

test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 0, "Age"],
                              df.loc[df["Outcome"] == 1 , "Age"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

## p-value is smaller than 0.05 so H0 reject.
# what are we rejecting, our hipothesis is taht There is no difference whether have diabetic related with age.
# we reject this hipothesis becouse p-value < 0.05 so there is a difference related with age.




