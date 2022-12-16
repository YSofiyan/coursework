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

df_subset_2019 = df_subset[(df_subset['year'] == 2019)].dropna()
df_subset_2019_2 = df_subset_2019[['cgdpo', 'emp', 'avh', 'hc', 'countrycode', 'year', 'country', 'cn', 'pop', 'ctfp', 'labsh']].dropna()


###3
 
#Gettinmg the countriy with highest and lowest GDP at current PPP

df_rich_country = df_subset_2019.loc[df_subset['cgdpo'] == df_subset_2019['cgdpo'].max()]
df_poor_country = df_subset_2019.loc[df_subset['cgdpo'] == df_subset_2019['cgdpo'].min()]

richest_income_per_worker = df_rich_country["cgdpo"] / df_rich_country["emp"]
print ("for the richest country, income per worker is", richest_income per worker)
poorest_income_per_worker = df_poor_country["cgdpo"] / df_poor_country["emp"]
print ("for the poorest country, income per worker is", poorest_income_per_worker)
richest_income_per_hour_worked = df_rich_country["cgdpo"] / (df_rich_country["emp"] * df_rich_country["avh"])
print ("for the richest country, income per hour worked is", richest_income_per_hour_worked)
poorest_income_per_hour_worked = df_poor_country["cgdpo"] / (df_poor_country["emp"] * df_poor_country["avh"])
print ("for the poorest country, income per hour worked is", poorest_income_per_hour_worked)
richest_income_per_unit_of_human_capital = df_rich_country['cgdpo']/ (df_rich_country['emp'] * df_rich_country['hc'])
print ("for the richest country, income per unit of human capital  is", richest_income_per_unit_of_human_capital)
poorest_income_per_unit_of_human_capital = df_poor_country['cgdpo']/ (df_poor_country['emp'] * df_poor_country['hc'])
print ("for the poorest country, income per worker is", poorest_income_per_unit_of_human_capital)
richest_income_per_hour_of_human_capital = df_rich_country['cgdpo']/ (df_rich_country['emp'] * df_rich_country['hc'] * df_rich_country['avh'])
print ("for the richest country, income per hour of human capital is", richest_income_per_hour_of_human_capital)
poorest_income_per_hour_of_human_capital = df_poor_country['cgdpo']/ (df_poor_country['emp'] * df_poor_country['hc'] * df_poor_country['avh'])
print ("for the poorest country, income per hour of human capital is", poorest_income_per_hour_of_human_capital)

#Countries in different quantiles 

df_subset_q = {}
income_per_worker = {}
income_per_hour_worked = {}
income_per_unit_of_human_capital = {}
income_per_hour_of_human_capital = {}
for q in (0.05, 0.1, 0.9, 0.95):
    df_subset_q[q] = (df_subset_2019.loc[df_subset['cgdpo'] == df_subset_2019.quantile((q), interpolation='nearest')['cgdpo']])
    income_per_worker[q] = (df_subset_q[q]['cgdpo']/ df_subset_q[q]['emp'])
    print ("for percentile", (q), "income per worker is", float(income_per_worker[q]))
    income_per_hour_worked[q] = (df_subset_q[q]['cgdpo']/ (df_subset_q[q]['emp'] * df_subset_q[q]['avh']))
    print ("for percentile", (q), "income per hour worked is", float(income_per_hour_worked[q]))
    income_per_unit_of_human_capital[q] = (df_subset_q[q]['cgdpo']/ (df_subset_q[q]['emp'] * df_subset_q[q]['hc']))
    print ("for percentile", (q), "income per unit of human capital is", float(income_per_unit_of_human_capital[q]))
    income_per_hour_of_human_capital[q] = (df_subset_q[q]['cgdpo']/ (df_subset_q[q]['emp'] * df_subset_q[q]['hc'] * df_subset_q[q]['avh']))
    print ("for percentile", (q), "income per hour of human capital is", float(income_per_hour_of_human_capital[q]))
 
#Ratios between richest and poorest countries and between percentiles 

GDP_ratio_between_richest_and_poorest = df_rich_country['cgdpo'] / df_poor_country['cgdpo']
print ('The GDP ratio between the richest and poorest countries is', GDP_ratio_between_richest_and_poorest, ': 1')
GDP_ratio_between_5th_95th_percentile = float(df_subset_q[0.95]['cgdpo']) / float(df_subset_q[0.05]['cgdpo'])
print ("The GDP ratio between the countries in the 95th and 5th percentiles is", GDP_ratio_between_5th_95th_percentile, ":1")
GDP_ratio_between_10th_90th_percentile = float(df_subset_q[0.9]['cgdpo']) / float(df_subset_q[0.1]['cgdpo'])
print ("The GDP ratio between the countries in the 90th and 10th percentiles is", GDP_ratio_between_10th_90th_percentile, ":1")
GDP_per_worker_ratio_richest_and_poorest = richest_income_per_worker / poorest_income_per_worker
print ("The GDP per worker ratio between the richest and poorest countries is", GDP_per_worker_ratio_richest_and_poorest, ":1")
GDP_per_worker_ratio_10th_90th_percentile = income_per_worker[0.9] / income_per_worker[0.1]
print ("The GDP per worker ratio between the countries in the 90th and 10th percentiles is", GDP_per_worker_ratio_10th_90th_percentile, ":1")
GDP_per_worker_ratio_5th_95th_percentile = income_per_worker[0.95] / income_per_worker[0.05]
print ("The GDP per worker ratio between the countries in the 95th and 5th percentiles is", GDP_per_worker_ratio_5th_95th_percentile, ":1")

###5

import matplotlib.pyplot as plt
import statistics

df_subset_2019_2['log_gdp'] = np.log((df_subset_2019_2["cgdpo"]))
df_subset_2019_2['gdp_per_worker'] = df_subset_2019_2["cgdpo"] / df_subset_2019_2['emp']
df_subset_2019_2['log_gdp_per_capita'] = np.log(df_subset_2019_2["cgdpo"] / df_subset_2019_2['pop'])
df_subset_2019_2['log_gdp_per_worker'] = np.log(df_subset_2019_2["cgdpo"] / df_subset_2019_2['emp'])
df_subset_2019_2['log_gdp_per_hour_worked'] = np.log(df_subset_2019_2["cgdpo"] / (df_subset_2019_2['avh'] / df_subset_2019['emp']))
df_subset_2019_2['log_gdp_per_hour_human_capital'] = np.log(df_subset_2019_2["cgdpo"] / (df_subset_2019_2['hc'] / df_subset_2019['emp']))
df_subset_2019_2['share_of_labour_compensation_in_GDP'] = ((-1 * df_subset_2019_2['labsh']) + 1)
x_variables = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])

import itertools

for x_variables, y_variables in itertools.product(x_variables, y_variables):
    ax = df_subset_2019_2.plot (x= x_variables, y = y_variables, kind='scatter')
    plt.xlabel(x_variables, fontsize=12 )
    plt.ylabel(y_variables, fontsize=12)
    df_subset_2019_2[[x_variables, y_variables, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
plt.show()
plt.close()

###6

import statistics

var_log_y_kh = statistics.variance(np.log((df_subset_2019["cgdpo"] / df_subset_2019["emp"]) / df_subset_2019["ctfp"]))
var_log_y = statistics.variance(np.log((df_subset_2019["cgdpo"]) / df_subset_2019['emp']))
df_subset_2019_2['ykh'] = (df_subset_2019["cgdpo"] / (df_subset_2019["emp"])) / df_subset_2019["ctfp"]
df_subset_2019_2['y'] = df_subset_2019["cgdpo"] / df_subset_2019['emp']

success1 = var_log_y_kh / var_log_y
print ("success 1 is", success1)

df_subset_q1_success = {}
df_subset_q2_success = {}
success_2 = {}
df_subset_y_success = {}
df_subset_ykh_success = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success[q] = df_subset_2019_2['gdp_per_worker'].quantile(q)
    df_subset_q1_success[r] = df_subset_2019_2['gdp_per_worker'].quantile(r)
    df_subset_y_success[q,r] = (df_subset_q1_success[q]/df_subset_q1_success[r])
    df_subset_q2_success[q] = df_subset_2019_2['ykh'].quantile(q)
    df_subset_q2_success[r] = df_subset_2019_2['ykh'].quantile(r)
    df_subset_ykh_success[q,r] = (df_subset_q2_success[q]/df_subset_q2_success[r])
    success_2[q] = (df_subset_ykh_success[q,r]/ df_subset_y_success[q,r])
    #print("for FIRST percentiles", q, "and", r, "the ratio is", (success_2[q]))
    
