# Numerical Methods Coursework

#Question 1:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fp = "https://raw.githubusercontent.com/jivizcaino/PWT_10.0/main/pwt100.csv"
df = pd.read_csv(fp,encoding='latin-1')

#Question 2:

#Creating a subset for variables in interest
df_subset = df[['cgdpo', 'emp', 'avh', 'hc', 'countrycode', 'year', 'country', 'cn', 'pop', 'ctfp','labsh']].dropna()
print(df_subset.dropna().describe())
df_subset.to_csv("pwt_subset.csv", index = "country")
#Choosing a year that maximises the number of observations for variables of interest
for year in range (2010, 2020):
  x = df_subset[df_subset["year"] == year].describe()
  print(x.dropna())
  max_observ = df["country"].value_counts()
  print(max_observ)
 
#As can be seen from tables printed, the number of observations
#is very consistent across countries and overall provides negligible
#difference between 2010 and 2019. Therefore it makes sense to pick
#the latest year available - 2019

#Question 3:
 
#Getting the country with the highest and lowest GDP at current PPP:

df_subset_2019 = df_subset[(df_subset['year'] == 2019)].dropna()
print (df_subset_2019)

df_rich_country = df_subset_2019.loc[df_subset['cgdpo'] == df_subset_2019['cgdpo'].max()]
df_poor_country = df_subset_2019.loc[df_subset['cgdpo'] == df_subset_2019['cgdpo'].min()]

richest_income_per_worker = float(df_rich_country["cgdpo"] / df_rich_country["emp"])
print ("for the richest country, income per worker is", richest_income_per_worker)
poorest_income_per_worker = float(df_poor_country["cgdpo"] / df_poor_country["emp"])
print ("for the poorest country, income per worker is", poorest_income_per_worker)
richest_income_per_hour_worked = float(df_rich_country["cgdpo"] / (df_rich_country["emp"] * df_rich_country["avh"]))
print ("for the richest country, income per hour worked is", richest_income_per_hour_worked)
poorest_income_per_hour_worked = float(df_poor_country["cgdpo"] / (df_poor_country["emp"] * df_poor_country["avh"]))
print ("for the poorest country, income per hour worked is", poorest_income_per_hour_worked)
richest_income_per_unit_of_human_capital = float(df_rich_country['cgdpo']/ (df_rich_country['emp'] * df_rich_country['hc']))
print ("for the richest country, income per unit of human capital  is", richest_income_per_unit_of_human_capital)
poorest_income_per_unit_of_human_capital = float(df_poor_country['cgdpo']/ (df_poor_country['emp'] * df_poor_country['hc']))
print ("for the poorest country, income per worker is", poorest_income_per_unit_of_human_capital)
richest_income_per_hour_of_human_capital = float(df_rich_country['cgdpo']/ (df_rich_country['emp'] * df_rich_country['hc'] * df_rich_country['avh']))
print ("for the richest country, income per hour of human capital is", richest_income_per_hour_of_human_capital)
poorest_income_per_hour_of_human_capital = float (df_poor_country['cgdpo']/ (df_poor_country['emp'] * df_poor_country['hc'] * df_poor_country['avh']))
print ("for the poorest country, income per hour of human capital is", poorest_income_per_hour_of_human_capital)

#Countries in different percentiles: 

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
 
#Ratios between richest and poorest countries and between percentiles:

GDP_ratio_between_richest_and_poorest = float(df_rich_country['cgdpo']) / float(df_poor_country['cgdpo'])
print ('The GDP ratio between the richest and poorest countries is', GDP_ratio_between_richest_and_poorest, ': 1')
GDP_ratio_between_5th_95th_percentile = float(df_subset_q[0.95]['cgdpo']) / float(df_subset_q[0.05]['cgdpo'])
print ("The GDP ratio between the countries in the 95th and 5th percentiles is", GDP_ratio_between_5th_95th_percentile, ":1")
GDP_ratio_between_10th_90th_percentile = float(df_subset_q[0.9]['cgdpo']) / float(df_subset_q[0.1]['cgdpo'])
print ("The GDP ratio between the countries in the 90th and 10th percentiles is", GDP_ratio_between_10th_90th_percentile, ":1")
GDP_per_worker_ratio_richest_and_poorest = float(richest_income_per_worker) / float(poorest_income_per_worker)
print ("The GDP per worker ratio between the richest and poorest countries is", GDP_per_worker_ratio_richest_and_poorest, ":1")
GDP_per_worker_ratio_10th_90th_percentile = float(income_per_worker[0.9]) / float(income_per_worker[0.1])
print ("The GDP per worker ratio between the countries in the 90th and 10th percentiles is", GDP_per_worker_ratio_10th_90th_percentile, ":1")
GDP_per_worker_ratio_5th_95th_percentile = float(income_per_worker[0.95]) / float(income_per_worker[0.05])
print ("The GDP per worker ratio between the countries in the 95th and 5th percentiles is", GDP_per_worker_ratio_5th_95th_percentile, ":1")

#Tabulating results: 

from tabulate import tabulate
table1_data = [["Country", "Income per Worker", "Income per Hour Worked", "Income per Unit of Human Capital","Income per Hour of Human Capital"],
["United States", float(richest_income_per_worker), float(richest_income_per_hour_worked), float(richest_income_per_unit_of_human_capital), float(richest_income_per_hour_of_human_capital)],
["Japan", float(income_per_worker[0.95]), float(income_per_hour_worked[0.95]), float(income_per_unit_of_human_capital[0.95]), float(income_per_hour_of_human_capital[0.95])],
["Indonesia", float(income_per_worker[0.9]), float(income_per_hour_worked[0.9]), float(income_per_unit_of_human_capital[0.9]), float(income_per_hour_of_human_capital[0.9])],
["Uruguay", float(income_per_worker[0.1]), float(income_per_hour_worked[0.1]), float(income_per_unit_of_human_capital[0.1]), float(income_per_hour_of_human_capital[0.1])],
["Estonia", float(income_per_worker[0.05]), float(income_per_hour_worked[0.05]), float(income_per_unit_of_human_capital[0.05]), float(income_per_hour_of_human_capital[0.05])],
["Malta", float(poorest_income_per_worker), float(poorest_income_per_hour_worked), float(poorest_income_per_unit_of_human_capital), float(poorest_income_per_hour_of_human_capital)]]
print(tabulate(table1_data, headers='firstrow', tablefmt='fancy_grid'))

table2_data = [["Percentiles", "Countries" , "GDP ratio", "GDP per worker ratio"], ["Minimum and Maximum value for GDP", "Malta and United States",GDP_ratio_between_richest_and_poorest,GDP_per_worker_ratio_richest_and_poorest],["95th and 5th percentiles", "Japan and Estonia",GDP_ratio_between_5th_95th_percentile,GDP_per_worker_ratio_5th_95th_percentile ], ["90th and 10th percentiles", "Indonesia and Uruguay",GDP_ratio_between_10th_90th_percentile,GDP_per_worker_ratio_10th_90th_percentile]]
print(tabulate(table2_data, headers='firstrow', tablefmt='fancy_grid'))

#Question 4:

#Standard of living for each country can be denoted by the income per worker. The differences in human capital and hours worked can be a useful way to explain differences in standard of living. Having greater access to human capital suggests that a country has better education resources, improving the economies innovation and social well-being, helping the economy grow, improving standard of living as income per worker will rise as the economy grows. As well as this, having a greater income per hour worked will also increase the standard of living because it means that the workers will get more benefit from their time spent working, increasing the income per capita. This is supported by the results in our table 1 and table 2. As we have seen in table 1, for the richest and poorest countries, the US has greater income per Hour Worked, income per Unit of Human Capital, compared to Malta resulting in a standard of living that is 1.65 times bigger as the magnitude of the differences in income per worker is 1.65. This helps contribute to the fact that the GDP ratio between the USA and Malta is 1186.69. For the countries in the 95th and 5th percentile, the country in the 95th percentile, Japan, has greater income per Hour Worked, income per Unit of Human Capital, compared to the country in the 5th percentile, Estonia, resulting in a standard of living that is 1.07 times bigger as the magnitude of the differences in income per worker is 1.07, this small due to the fact that these countries are similar in terms of these measures. However, the GDP ratio of Japan to Estonia is 111.62, this is because Japan’s population is significantly bigger, hence reflecting their superior GDP, whilst having a similar standard of living. For the countries in the 90th and 10th percentile, the country in the 90th percentile, Indonesia, has lower income per Hour Worked, income per Unit of Human Capital, compared to the country in the 10th percentile, Uruguay, resulting in a standard of living that is 0.55 times smaller as the magnitude of the differences in income per worker is 0.55. Despite having a significant GDP, Indonesia’s standard of living is proved to be extremely poor due to their weak income per hour worked and poor human capital. The GDP ratio between Indonesia and Uruguay is 43.7577, Indonesia has a greater GDP due to their much larger population, making up for their lack of income per capita. A better indicator of standard of living may be to use the HDI index because it considers other factors that will affect the standard of living in a country.

#Question 5:

import matplotlib.pyplot as plt
import statistics

df_subset_2019['log_gdp'] = np.log((df_subset_2019["cgdpo"]))
df_subset_2019['gdp_per_worker'] = df_subset_2019["cgdpo"] / df_subset_2019['emp']
df_subset_2019['log_gdp_per_capita'] = np.log(df_subset_2019["cgdpo"] / df_subset_2019['pop'])
df_subset_2019['log_gdp_per_worker'] = np.log(df_subset_2019["cgdpo"] / df_subset_2019['emp'])
df_subset_2019['log_gdp_per_hour_worked'] = np.log(df_subset_2019["cgdpo"] / (df_subset_2019['avh'] / df_subset_2019['emp']))
df_subset_2019['log_gdp_per_hour_human_capital'] = np.log(df_subset_2019["cgdpo"] / (df_subset_2019['hc'] / df_subset_2019['emp']))
df_subset_2019['share_of_labour_compensation_in_GDP'] = ((-1 * df_subset_2019['labsh']) + 1)
x_variables = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])

import itertools

for x_variables, y_variables in itertools.product(x_variables, y_variables):
    ax = df_subset_2019.plot (x= x_variables, y = y_variables, kind='scatter')
    plt.xlabel(x_variables, fontsize=12 )
    plt.ylabel(y_variables, fontsize=12)
    df_subset_2019[[x_variables, y_variables, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
plt.show()
plt.close()

#Question 6:

import statistics

var_log_y_kh = statistics.variance(np.log((df_subset_2019["cgdpo"] / df_subset_2019["emp"]) / df_subset_2019["ctfp"]))
var_log_y = statistics.variance(np.log((df_subset_2019["cgdpo"]) / df_subset_2019['emp']))
df_subset_2019['ykh'] = (df_subset_2019["cgdpo"] / (df_subset_2019["emp"])) / df_subset_2019["ctfp"]
df_subset_2019['y'] = df_subset_2019["cgdpo"] / df_subset_2019['emp']

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
    df_subset_q1_success[q] = df_subset_2019['y'].quantile(q)
    df_subset_q1_success[r] = df_subset_2019['y'].quantile(r)
    df_subset_y_success[q,r] = (df_subset_q1_success[q]/df_subset_q1_success[r])
    df_subset_q2_success[q] = df_subset_2019['ykh'].quantile(q)
    df_subset_q2_success[r] = df_subset_2019['ykh'].quantile(r)
    df_subset_ykh_success[q,r] = (df_subset_q2_success[q]/df_subset_q2_success[r])
    success_2[q] = (df_subset_ykh_success[q,r]/ df_subset_y_success[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2[q]))
    print(("percentile", q, df_subset_2019.loc[df_subset_2019['y'] == df_subset_2019.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_subset_2019.loc[df_subset_2019['y'] == df_subset_2019.quantile((r), interpolation='nearest')['y']]))  
  
#Question 8:

df_subset_q2_success_TFP = {}
success_2_TFP = {}
df_subset_TFP_success = {}

for (q, r) in zip (q_list, r_list):
    df_subset_q1_success[q] = df_subset_2019['y'].quantile(q)
    df_subset_q1_success[r] = df_subset_2019['y'].quantile(r)
    df_subset_y_success[q,r] = (df_subset_q1_success[q]/df_subset_q1_success[r])
    df_subset_q2_success_TFP[q] = df_subset_2019['ctfp'].quantile(q)
    df_subset_q2_success_TFP[r] = df_subset_2019['ctfp'].quantile(r)
    df_subset_TFP_success[q,r] = (df_subset_q2_success_TFP[q]/df_subset_q2_success_TFP[r])
    success_2_TFP[q] = (df_subset_TFP_success[q,r]/ df_subset_y_success[q,r])
    print("for percentiles", q, "and", r, "success 2", (success_2_TFP[q]))
    print(("percentile", q, df_subset_2019.loc[df_subset_2019['y'] == df_subset_2019.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_subset_2019.loc[df_subset_2019['y'] == df_subset_2019.quantile((r), interpolation='nearest')['y']]))

var_log_y_kh = statistics.variance(np.log((df_subset_2019["cgdpo"] / df_subset_2019["emp"]) / df_subset_2019["ctfp"]))
var_log_TFP = statistics.variance(np.log((df_subset_2019["ctfp"])))
df_subset_2019['ykh'] = (df_subset_2019["cgdpo"] / (df_subset_2019["emp"])) / df_subset_2019["ctfp"]
df_subset_2019['y'] = df_subset_2019["cgdpo"] / df_subset_2019['emp']

success1_TFP = var_log_TFP/ var_log_y
print ("success 1 TFP is", success1_TFP)

#OECD
OECD_countries = ["Luxembourg", "Ireland", "Switzerland", "Norway", "United States", "Iceland", "Netherlands", "Austria", "Denmark", "Australia", "Germany", "Belgium", "Finland", "Canada", "United Kingdom", "France", "Japan", "Italy", "New Zealand", "Israel", "Czech Republic", "Spain", "Slovenia", "Estonia", "Slovakia", "Portugal", "Poland", "Hungary", "Greece", "Turkey", "Chile", "Mexico", "Sweden"]
df_OECD = df_subset_2019[df_subset_2019["country"].isin(OECD_countries)].copy()
df_non_OECD = df_subset_2019[df_subset_2019["country"].isin(OECD_countries) == False].copy()

var_log_y_kh = statistics.variance(np.log((df_OECD["cgdpo"] / df_OECD["emp"]) / df_OECD["ctfp"]))
var_log_y = statistics.variance(np.log((df_OECD["cgdpo"]) / df_OECD['emp']))
df_OECD['ykh'] = (df_OECD["cgdpo"] / (df_OECD["emp"])) / df_OECD["ctfp"]
df_OECD['y'] = df_OECD["cgdpo"] / df_OECD['emp']

success1 = var_log_y_kh / var_log_y
print ("success 1 is", success1)

df_subset_q1_success_OECD = {}
df_subset_q2_success_OECD = {}
success_2_OECD = {}
df_subbset_y_success_OECD = {}
df_subset_ykh_success_OECD = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_OECD[q] = df_OECD['y'].quantile(q)
    df_subset_q1_success_OECD[r] = df_OECD['y'].quantile(r)
    df_subbset_y_success_OECD[q,r] = (df_subset_q1_success_OECD[q]/df_subset_q1_success_OECD[r])
    df_subset_q2_success_OECD[q] = df_OECD['ykh'].quantile(q)
    df_subset_q2_success_OECD[r] = df_OECD['ykh'].quantile(r)
    df_subset_ykh_success_OECD[q,r] = (df_subset_q2_success_OECD[q]/df_subset_q2_success_OECD[r])
    success_2_OECD[q] = (df_subset_ykh_success_OECD[q,r]/ df_subbset_y_success_OECD[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_OECD[q]))
    print(("percentile", q, df_OECD.loc[df_OECD['y'] == df_OECD.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_OECD.loc[df_OECD['y'] == df_OECD.quantile((r), interpolation='nearest')['y']]))  
    
#Question 9:

#Graphs

df_above_median = (df_subset_2019.loc[df_subset['cgdpo'] > df_subset_2019.quantile((0.5), interpolation='nearest')['cgdpo']]).copy()
df_below_median = (df_subset_2019.loc[df_subset['cgdpo'] < df_subset_2019.quantile((0.5), interpolation='nearest')['cgdpo']]).copy()

OECD_countries = ["Luxembourg", "Ireland", "Switzerland", "Norway", "United States", "Iceland", "Netherlands", "Austria", "Denmark", "Australia", "Germany", "Belgium", "Finland", "Canada", "United Kingdom", "France", "Japan", "Italy", "New Zealand", "Israel", "Czech Republic", "Spain", "Slovenia", "Estonia", "Slovakia", "Portugal", "Poland", "Hungary", "Greece", "Turkey", "Chile", "Mexico", "Sweden", "Republic of Korea"].copy()
df_OECD = df_subset_2019[df_subset_2019["country"].isin(OECD_countries)].copy()
df_non_OECD = df_subset_2019[df_subset_2019["country"].isin(OECD_countries) == False].copy()

European_countries = ["Austria", "Belgium", "Bulgaria", "Switzerland", "Cyprus", "Czech Republic", "Germany", "Denmark", "Spain", "Estonia", "Finland", "France", "United Kingdom", "Greece", "Croatia", "Hungary", "Ireland", "Iceland", "Italy", "Lithuania", "Luxembourg", "Latvia", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Russian Federation", "Sweden", "Turkey", "Malta", "Slovenia", "Slovakia"].copy()
Asian_and_oceanic_countries = ["Australia", "China", "China, Hong Kong SAR", "Indonesia", "India", "Israel", "Japan", "Republic of Korea", "Malaysia", "New Zealand", "Philippines", "Singapore", "Thailand", "Taiwan", "Sri Lanka"]
African_countries = ["South Africa"].copy()
Americas_countries = ["Argentina", "Brazil", "Canada", "Chile", "Colombia", "Costa Rica", "Dominican Republic", "Ecuador", "Peru", "Uruguay", "United States", "Mexico"].copy()
df_European = df_subset_2019[df_subset_2019["country"].isin(European_countries)].copy()
df_Asian_Oceanic = df_subset_2019[df_subset_2019["country"].isin(Asian_and_oceanic_countries)].copy()
df_African = df_subset_2019[df_subset_2019["country"].isin(African_countries)].copy()
df_Americas = df_subset_2019[df_subset_2019["country"].isin(Americas_countries)].copy()

x_variables_above_median = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables_above_median = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])
for x_variables_above_median, y_variables_above_median in itertools.product(x_variables_above_median, y_variables_above_median):
    ax = df_above_median.plot (x= x_variables_above_median, y = y_variables_above_median, kind='scatter')
    plt.xlabel(x_variables_above_median, fontsize=12 )
    plt.ylabel(y_variables_above_median, fontsize=12)
    df_above_median[[x_variables_above_median, y_variables_above_median, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
    plt.title ("above median")
    plt.show()
    plt.close()

x_variables_below_median = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables_below_median = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])
for x_variables_below_median, y_variables_below_median in itertools.product(x_variables_below_median, y_variables_below_median):
    ax = df_above_median.plot (x= x_variables_below_median, y = y_variables_below_median, kind='scatter')
    plt.xlabel(x_variables_below_median, fontsize=12 )
    plt.ylabel(y_variables_below_median, fontsize=12)
    df_above_median[[x_variables_below_median, y_variables_below_median, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
    plt.title ("below median")
    plt.show()
    plt.close()

x_variables_OECD = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables_OECD = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])
for x_variables_OECD, y_variables_OECD in itertools.product(x_variables_OECD, y_variables_OECD):
    ax = df_OECD.plot (x= x_variables_OECD, y = y_variables_OECD, kind='scatter')
    plt.xlabel(x_variables_OECD, fontsize=12 )
    plt.ylabel(y_variables_OECD, fontsize=12)
    df_OECD[[x_variables_OECD, y_variables_OECD, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
    plt.title ("OECD countries")
    plt.show()
    plt.close()

x_variables_non_OECD = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables_non_OECD = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])
for x_variables_non_OECD, y_variables_non_OECD in itertools.product(x_variables_non_OECD, y_variables_non_OECD):
    ax = df_non_OECD.plot (x= x_variables_non_OECD, y = y_variables_non_OECD, kind='scatter')
    plt.xlabel(x_variables_non_OECD, fontsize=12 )
    plt.ylabel(y_variables_non_OECD, fontsize=12)
    df_non_OECD[[x_variables_non_OECD, y_variables_non_OECD, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
    plt.title ("non-OECD countries")
    plt.show()
    plt.close()

x_variables_African = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables_African = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])
for x_variables_African, y_variables_African in itertools.product(x_variables_African, y_variables_African):
    ax = df_African.plot (x= x_variables_African, y = y_variables_African, kind='scatter')
    plt.xlabel(x_variables_African, fontsize=12 )
    plt.ylabel(y_variables_African, fontsize=12)
    df_African[[x_variables_African, y_variables_African, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
    plt.title ("African Countries")
    plt.show()
    plt.close()


x_variables_Americas = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables_Americas = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])
for x_variables_Americas, y_variables_Americas in itertools.product(x_variables_Americas, y_variables_Americas):
    ax = df_Americas.plot (x= x_variables_Americas, y = y_variables_Americas, kind='scatter')
    plt.xlabel(x_variables_Americas, fontsize=12 )
    plt.ylabel(y_variables_Americas, fontsize=12)
    df_Americas[[x_variables_Americas, y_variables_Americas, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
    plt.title ("Countries in the Americas")
    plt.show()
    plt.close()

x_variables_Asian_Oceanic = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables_Asian_Oceanic = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])
for x_variables_Asian_Oceanic, y_variables_Asian_Oceanic in itertools.product(x_variables_Asian_Oceanic, y_variables_Asian_Oceanic):
    ax = df_Asian_Oceanic.plot (x= x_variables_Asian_Oceanic, y = y_variables_Asian_Oceanic, kind='scatter')
    plt.xlabel(x_variables_Asian_Oceanic, fontsize=12 )
    plt.ylabel(y_variables_Asian_Oceanic, fontsize=12)
    df_Asian_Oceanic[[x_variables_Asian_Oceanic, y_variables_Asian_Oceanic, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
    plt.title ("Asian and Oceanic Countries")
    plt.show()
    plt.close()

x_variables_European = list(['log_gdp_per_capita', 'log_gdp_per_worker', 'log_gdp_per_hour_worked', 'log_gdp_per_hour_human_capital'])
y_variables_European = list(['cn', 'hc', 'avh', 'ctfp', 'share_of_labour_compensation_in_GDP'])
for x_variables_European, y_variables_European in itertools.product(x_variables_European, y_variables_European):
    ax = df_European.plot (x= x_variables_European, y = y_variables_European, kind='scatter')
    plt.xlabel(x_variables_European, fontsize=12 )
    plt.ylabel(y_variables_European, fontsize=12)
    df_European[[x_variables_European, y_variables_European, "countrycode"]].apply(lambda x: ax.text(*x), axis=1)   
    plt.title ("European Countries")
    plt.show()
    plt.close()

#Success 1 values:
#OECD

var_log_y_kh = statistics.variance(np.log((df_OECD["cgdpo"] / df_OECD["emp"]) / df_OECD["ctfp"]))
var_log_y = statistics.variance(np.log((df_OECD["cgdpo"]) / df_OECD['emp']))
df_OECD['ykh'] = (df_OECD["cgdpo"] / (df_OECD["emp"])) / df_OECD["ctfp"]
df_OECD['y'] = df_OECD["cgdpo"] / df_OECD['emp']

success1_OECD = var_log_y_kh / var_log_y
print ("success 1 is", success1_OECD)

df_subset_q1_success_OECD = {}
df_subset_q2_success_OECD = {}
success_2_OECD = {}
df_subbset_y_success_OECD = {}
df_subset_ykh_success_OECD = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_OECD[q] = df_OECD['y'].quantile(q)
    df_subset_q1_success_OECD[r] = df_OECD['y'].quantile(r)
    df_subbset_y_success_OECD[q,r] = (df_subset_q1_success_OECD[q]/df_subset_q1_success_OECD[r])
    df_subset_q2_success_OECD[q] = df_OECD['ykh'].quantile(q)
    df_subset_q2_success_OECD[r] = df_OECD['ykh'].quantile(r)
    df_subset_ykh_success_OECD[q,r] = (df_subset_q2_success_OECD[q]/df_subset_q2_success_OECD[r])
    success_2_OECD[q] = (df_subset_ykh_success_OECD[q,r]/ df_subbset_y_success_OECD[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_OECD[q]))
    print(("percentile", q, df_OECD.loc[df_OECD['y'] == df_OECD.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_OECD.loc[df_OECD['y'] == df_OECD.quantile((r), interpolation='nearest')['y']]))  

#Non-OECD

var_log_y_kh = statistics.variance(np.log((df_non_OECD["cgdpo"] / df_non_OECD["emp"]) / df_non_OECD["ctfp"]))
var_log_y = statistics.variance(np.log((df_non_OECD["cgdpo"]) / df_non_OECD['emp']))
df_non_OECD['ykh'] = (df_non_OECD["cgdpo"] / (df_non_OECD["emp"])) / df_non_OECD["ctfp"]
df_non_OECD['y'] = df_non_OECD["cgdpo"] / df_non_OECD['emp']

success1_non_OECD = var_log_y_kh / var_log_y
print ("success 1 is", success1_non_OECD)

df_subset_q1_success_non_OECD = {}
df_subset_q2_success_non_OECD = {}
success_2_non_OECD = {}
df_subbset_y_success_non_OECD = {}
df_subset_ykh_success_non_OECD = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_non_OECD[q] = df_non_OECD['y'].quantile(q)
    df_subset_q1_success_non_OECD[r] = df_non_OECD['y'].quantile(r)
    df_subbset_y_success_non_OECD[q,r] = (df_subset_q1_success_non_OECD[q]/df_subset_q1_success_non_OECD[r])
    df_subset_q2_success_non_OECD[q] = df_non_OECD['ykh'].quantile(q)
    df_subset_q2_success_non_OECD[r] = df_non_OECD['ykh'].quantile(r)
    df_subset_ykh_success_non_OECD[q,r] = (df_subset_q2_success_non_OECD[q]/df_subset_q2_success_non_OECD[r])
    success_2_non_OECD[q] = (df_subset_ykh_success_non_OECD[q,r]/ df_subbset_y_success_non_OECD[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_non_OECD[q]))
    print(("percentile", q, df_non_OECD.loc[df_non_OECD['y'] == df_non_OECD.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_non_OECD.loc[df_non_OECD['y'] == df_non_OECD.quantile((r), interpolation='nearest')['y']]))  
    
#Above and below median 

var_log_y_kh = statistics.variance(np.log((df_above_median["cgdpo"] / df_above_median["emp"]) / df_above_median["ctfp"]))
var_log_y = statistics.variance(np.log((df_above_median["cgdpo"]) / df_above_median['emp']))
df_above_median['ykh'] = (df_above_median["cgdpo"] / (df_above_median["emp"])) / df_above_median["ctfp"]
df_above_median['y'] = df_above_median["cgdpo"] / df_above_median['emp']

success1_above_median = var_log_y_kh / var_log_y
print ("success 1 is", success1_above_median)

df_subset_q1_success_above_median = {}
df_subset_q2_success_above_median = {}
success_2_above_median = {}
df_subbset_y_success_above_median = {}
df_subset_ykh_success_above_median = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_above_median[q] = df_above_median['y'].quantile(q)
    df_subset_q1_success_above_median[r] = df_above_median['y'].quantile(r)
    df_subbset_y_success_above_median[q,r] = (df_subset_q1_success_above_median[q]/df_subset_q1_success_above_median[r])
    df_subset_q2_success_above_median[q] = df_above_median['ykh'].quantile(q)
    df_subset_q2_success_above_median[r] = df_above_median['ykh'].quantile(r)
    df_subset_ykh_success_above_median[q,r] = (df_subset_q2_success_above_median[q]/df_subset_q2_success_above_median[r])
    success_2_above_median[q] = (df_subset_ykh_success_above_median[q,r]/ df_subbset_y_success_above_median[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_above_median[q]))
    print(("percentile", q, df_above_median.loc[df_above_median['y'] == df_above_median.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_above_median.loc[df_above_median['y'] == df_above_median.quantile((r), interpolation='nearest')['y']]))  

#Below

var_log_y_kh = statistics.variance(np.log((df_below_median["cgdpo"] / df_below_median["emp"]) / df_below_median["ctfp"]))
var_log_y = statistics.variance(np.log((df_below_median["cgdpo"]) / df_below_median['emp']))
df_below_median['ykh'] = (df_below_median["cgdpo"] / (df_below_median["emp"])) / df_below_median["ctfp"]
df_below_median['y'] = df_below_median["cgdpo"] / df_below_median['emp']

success1_below_median = var_log_y_kh / var_log_y
print ("success 1 is", success1_below_median)

df_subset_q1_success_below_median = {}
df_subset_q2_success_below_median = {}
success_2_below_median = {}
df_subbset_y_success_below_median = {}
df_subset_ykh_success_below_median = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_below_median[q] = df_below_median['y'].quantile(q)
    df_subset_q1_success_below_median[r] = df_below_median['y'].quantile(r)
    df_subbset_y_success_below_median[q,r] = (df_subset_q1_success_below_median[q]/df_subset_q1_success_below_median[r])
    df_subset_q2_success_below_median[q] = df_below_median['ykh'].quantile(q)
    df_subset_q2_success_below_median[r] = df_below_median['ykh'].quantile(r)
    df_subset_ykh_success_below_median[q,r] = (df_subset_q2_success_below_median[q]/df_subset_q2_success_below_median[r])
    success_2_below_median[q] = (df_subset_ykh_success_below_median[q,r]/ df_subbset_y_success_below_median[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_below_median[q]))
    print(("percentile", q, df_below_median.loc[df_below_median['y'] == df_below_median.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_below_median.loc[df_below_median['y'] == df_below_median.quantile((r), interpolation='nearest')['y']]))  
    
#Continents 
#Eurpoe

var_log_y_kh = statistics.variance(np.log((df_European["cgdpo"] / df_European["emp"]) / df_European["ctfp"]))
var_log_y = statistics.variance(np.log((df_European["cgdpo"]) / df_European['emp']))
df_European['ykh'] = (df_European["cgdpo"] / (df_European["emp"])) / df_European["ctfp"]
df_European['y'] = df_European["cgdpo"] / df_European['emp']

success1_European = var_log_y_kh / var_log_y
print ("success 1 is", success1_European)

df_subset_q1_success_European = {}
df_subset_q2_success_European= {}
success_2_European = {}
df_subbset_y_success_European = {}
df_subset_ykh_success_European = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_European[q] = df_European['y'].quantile(q)
    df_subset_q1_success_European[r] = df_European['y'].quantile(r)
    df_subbset_y_success_European[q,r] = (df_subset_q1_success_European[q]/df_subset_q1_success_European[r])
    df_subset_q2_success_European[q] = df_European['ykh'].quantile(q)
    df_subset_q2_success_European[r] = df_European['ykh'].quantile(r)
    df_subset_ykh_success_European[q,r] = (df_subset_q2_success_European[q]/df_subset_q2_success_European[r])
    success_2_European[q] = (df_subset_ykh_success_European[q,r]/ df_subbset_y_success_European[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_European[q]))
    print(("percentile", q, df_European.loc[df_European['y'] == df_European.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_European.loc[df_European['y'] == df_European.quantile((r), interpolation='nearest')['y']]))  
    
#Asian and Oceanic

var_log_y_kh = statistics.variance(np.log((df_Asian_Oceanic["cgdpo"] / df_Asian_Oceanic["emp"]) / df_Asian_Oceanic["ctfp"]))
var_log_y = statistics.variance(np.log((df_Asian_Oceanic["cgdpo"]) / df_Asian_Oceanic['emp']))
df_Asian_Oceanic['ykh'] = (df_Asian_Oceanic["cgdpo"] / (df_Asian_Oceanic["emp"])) / df_Asian_Oceanic["ctfp"]
df_Asian_Oceanic['y'] = df_Asian_Oceanic["cgdpo"] / df_Asian_Oceanic['emp']

success1_Asian_Oceanic = var_log_y_kh / var_log_y
print ("success 1 is", success1_Asian_Oceanic)

df_subset_q1_success_Asian_Oceanic = {}
df_subset_q2_success_Asian_Oceanic = {}
success_2_Asian_Oceanic = {}
df_subbset_y_success_Asian_Oceanic = {}
df_subset_ykh_success_Asian_Oceanic = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_Asian_Oceanic[q] = df_Asian_Oceanic['y'].quantile(q)
    df_subset_q1_success_Asian_Oceanic[r] = df_Asian_Oceanic['y'].quantile(r)
    df_subbset_y_success_Asian_Oceanic[q,r] = (df_subset_q1_success_Asian_Oceanic[q]/df_subset_q1_success_Asian_Oceanic[r])
    df_subset_q2_success_Asian_Oceanic[q] = df_Asian_Oceanic['ykh'].quantile(q)
    df_subset_q2_success_Asian_Oceanic[r] = df_Asian_Oceanic['ykh'].quantile(r)
    df_subset_ykh_success_Asian_Oceanic[q,r] = (df_subset_q2_success_Asian_Oceanic[q]/df_subset_q2_success_Asian_Oceanic[r])
    success_2_Asian_Oceanic[q] = (df_subset_ykh_success_Asian_Oceanic[q,r]/ df_subbset_y_success_Asian_Oceanic[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_Asian_Oceanic[q]))
    print(("percentile", q, df_Asian_Oceanic.loc[df_Asian_Oceanic['y'] == df_Asian_Oceanic.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_Asian_Oceanic.loc[df_Asian_Oceanic['y'] == df_Asian_Oceanic.quantile((r), interpolation='nearest')['y']]))  

#Americas

var_log_y_kh = statistics.variance(np.log((df_Americas["cgdpo"] / df_Americas["emp"]) / df_Americas["ctfp"]))
var_log_y = statistics.variance(np.log((df_Americas["cgdpo"]) / df_Americas['emp']))
df_Americas['ykh'] = (df_Americas["cgdpo"] / (df_Americas["emp"])) / df_Americas["ctfp"]
df_Americas['y'] = df_Americas["cgdpo"] / df_Americas['emp']

success1_Americas = var_log_y_kh / var_log_y
print ("success 1 is", success1_Americas)

df_subset_q1_success_Americas= {}
df_subset_q2_success_Americas = {}
success_2_Americas = {}
df_subbset_y_success_Americas = {}
df_subset_ykh_success_Americas = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_Americas[q] = df_Americas['y'].quantile(q)
    df_subset_q1_success_Americas[r] = df_Americas['y'].quantile(r)
    df_subbset_y_success_Americas[q,r] = (df_subset_q1_success_Americas[q]/df_subset_q1_success_Americas[r])
    df_subset_q2_success_Americas[q] = df_Americas['ykh'].quantile(q)
    df_subset_q2_success_Americas[r] = df_Americas['ykh'].quantile(r)
    df_subset_ykh_success_Americas[q,r] = (df_subset_q2_success_Americas[q]/df_subset_q2_success_Americas[r])
    success_2_Americas[q] = (df_subset_ykh_success_Americas[q,r]/ df_subbset_y_success_Americas[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_Americas[q]))
    print(("percentile", q, df_Americas.loc[df_Americas['y'] == df_Americas.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_Americas.loc[df_Americas['y'] == df_Americas.quantile((r), interpolation='nearest')['y']]))  
    
#African

#There is only one data point here for Africa so cannot workout the variance required for success 1
#This will also mean that success 2 is just 1

df_African['ykh'] = (df_African["cgdpo"] / (df_African["emp"])) / df_African["ctfp"]
df_African['y'] = df_African["cgdpo"] / df_African['emp']

df_subset_q1_success_African = {}
df_subset_q2_success_African = {}
success_2_African = {}
df_subbset_y_success_African = {}
df_subset_ykh_success_African = {}

q_list = [0.99, 0.95, 0.9, 0.75]
r_list = [0.01, 0.05, 0.1, 0.25]
for (q, r) in zip (q_list, r_list):
    df_subset_q1_success_African[q] = df_African['y'].quantile(q)
    df_subset_q1_success_African[r] = df_African['y'].quantile(r)
    df_subbset_y_success_African[q,r] = (df_subset_q1_success_African[q]/df_subset_q1_success_African[r])
    df_subset_q2_success_African[q] = df_African['ykh'].quantile(q)
    df_subset_q2_success_African[r] = df_African['ykh'].quantile(r)
    df_subset_ykh_success_African[q,r] = (df_subset_q2_success_African[q]/df_subset_q2_success_African[r])
    success_2_African[q] = (df_subset_ykh_success_African[q,r]/ df_subbset_y_success_African[q,r])
    print("for percentiles", q, "and", r, "success 2 is", (success_2_African[q]))
    print(("percentile", q, df_African.loc[df_African['y'] == df_African.quantile((q), interpolation='nearest')['y']]))
    print(("percentile", r, df_African.loc[df_African['y'] == df_African.quantile((r), interpolation='nearest')['y']]))  
    
#Summary table:

summarytable_data = [['Subsample', "Number of Observations:", "Var[log(y)]:", "Var[log(ykh)]:", "Success 1:"], ["Above Median", len(df_above_median), statistics.variance(np.log(df_above_median['y'])), statistics.variance(np.log(df_above_median['ykh'])), success1_above_median],["Below Median", len(df_below_median), statistics.variance(np.log(df_below_median['y'])), statistics.variance(np.log(df_below_median['ykh'])), success1_below_median],["OECD Countries", len(df_OECD), statistics.variance(np.log(df_OECD['y'])), statistics.variance(np.log(df_OECD['ykh'])), success1_OECD],["Non-OECD Countries", len(df_non_OECD), statistics.variance(np.log(df_non_OECD['y'])), statistics.variance(np.log(df_non_OECD['ykh'])), success1_non_OECD],["African Countries", len(df_African), "N/A", "N/A", "N/A"],["Countries in the Americas", len(df_Americas), statistics.variance(np.log(df_Americas['y'])), statistics.variance(np.log(df_Americas['ykh'])), success1_Americas],["Asian and Oceanic Countries", len(df_Asian_Oceanic), statistics.variance(np.log(df_Asian_Oceanic['y'])), statistics.variance(np.log(df_Asian_Oceanic['ykh'])), success1_Asian_Oceanic],["European Countries", len(df_European), statistics.variance(np.log(df_European['y'])), statistics.variance(np.log(df_European['ykh'])), success1_European]]
print(tabulate(summarytable_data, headers='firstrow', tablefmt='fancy_grid'))
