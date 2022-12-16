# coursework

###1

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


fp = r"/Users/yurisofiyan/Desktop/Economics /ThirdYear/NumericalMethods/pwt100.csv"
df = pd.read_csv(fp,encoding='UTF-8')

###2

#Creating a subset for variables in interest

df_subset = df[['cgdpo', 'emp', 'avh', 'hc', 'countrycode', 'year', 'country', 'cn', 'pop', 'ctfp', 'labsh']]
df_subset.to_csv("pwt_subset.csv", index = "country")
 
#Choosing a year to perform ananlysis on

for year in range (2010, 2020):
   x = df_subset[df_subset["year"] == year].describe()
 
max_observ = df["country"].value_counts()

