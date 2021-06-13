import pandas as pd 
import numpy as np
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os
from warnings import filterwarnings
filterwarnings('ignore')



# Download if data is not available
from kaggle.api.kaggle_api_extended import KaggleApi
if not os.path.exists("edstats-csv-zip-32-mb-/EdStatsData.csv"):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('theworldbank/education-statistics', unzip=True)
    
    
#read from the downloaded folder:
df = pd.read_csv("edstats-csv-zip-32-mb-/EdStatsData.csv")

#select the columns and the data:

main_cols = ['Country Code', 'Country Name', 'Indicator Code',
             'Indicator Name'] + [str(i) for i in range(1990, 2015)]
df = df[main_cols]

###############Grouping and ranking the data###########################

# ask if there is a numeric value
num_cols = [col for col in df.columns if np.char.isnumeric(col)]

ind_count_year = df[num_cols + ["Indicator Code", "Indicator Name"]].groupby(["Indicator Code",  "Indicator Name"]).count()

###ranking the data###
sorted_ind = ind_count_year.sum(axis=1).reset_index(
    name="count").sort_values(['count'], ascending=False)
sorted_ind['rank'] = sorted_ind['count'].rank(ascending=0)
sorted_ind_top_20 = sorted_ind.head(20)

##pisa indicator filter####
pd.set_option('display.max_colwidth', None)
sorted_ind_pisa_top_10 = sorted_ind[sorted_ind["Indicator Code"].str.contains(
    'PISA')].head(10)

# selection the indicators and create the new data frame:

# now let's choose our 16 top indicators
indicators = [
    # Population, total, male, female
    "SP.POP.TOTL.MA.IN", "SP.POP.TOTL.FE.IN", "SP.POP.TOTL",
    # Population of the official age for primary education total, male, female
    "SP.PRM.TOTL.FE.IN", "SP.PRE.TOTL.MA.IN", "SP.PRM.TOTL.IN",
    # PISA: Mean performance on the mathematics scale. total, male, female
    "LO.PISA.MAT", "LO.PISA.MAT.FE", "LO.PISA.MAT.MA",
    # PISA: Mean performance on the science scale,  total, male, female
    "LO.PISA.SCI", "LO.PISA.SCI.FE", "LO.PISA.SCI.MA",
    # PISA: Mean performance on the reading scale,  total, male, female
    "LO.PISA.REA.MA", "LO.PISA.REA.FE", "LO.PISA.REA",
    "NY.GDP.PCAP.PP.KD"  # GDP per capita, PPP
]


# Country filtering

df_filter_ind = df[df['Indicator Code'].isin(indicators)]

######### Summary table info for indicators #########################

# file info:
num_countries_after_filter_ind = len(df_filter_ind['Country Code'].unique())
num_indicators_after_filter_ind = len(df_filter_ind['Indicator Code'].unique())


##############same format of summary table################################
# initialize list
summary_table_df_filter_ind = [["Countries", num_countries_after_filter_ind], [
    "Indicators", num_indicators_after_filter_ind], ["Years", "1990-2014"]]
# Create the pandas DataFrame
summary_table_df_filter_ind = pd.DataFrame(
    summary_table_df_filter_ind, columns=['Variables', 'Observations'])

############## top and bottom countries filtering############################

number_of_countries = 6
list_countries = []  # initialize the list of countries

# function for create a list:


def update_list_countries(dataframe):
    for i in range(number_of_countries):
        list_countries.append(dataframe.iloc[i, 0])
    return list_countries

# # top
search_cols = "LO.PISA.MAT"
projection = ["Country Name", "2012"]
pisa_mat_total = df[(df["Indicator Code"] == search_cols) & (df["2012"].notna())]
top6_highestPISA_2012 = pisa_mat_total.sort_values("2012", ascending=False)[
    projection].head(number_of_countries)

# # update list:
update_list_countries(top6_highestPISA_2012)

# # bottom
bottom6_lowestPISA_2012 = pisa_mat_total.sort_values("2012")[projection].head(number_of_countries)
# # update list country:
update_list_countries(bottom6_lowestPISA_2012)


######################
#    1) TOP 20 overall indicators (LISTO)
#    2) Top 10 pisa indicators (listo)
#    3) summary table with the variables and observations
#    3) FOURTH: Top 50 indicators
#    4)  TOP 6, BOTTOM 6,
#    5) TOP 6 WITH HIGHEST GROWTH RATE, BOTTOM 6 WITH LOWEST GROWTH RATE
#    6) Summary table with the variables and observations
# """"""""




############let's create a growth rate and filter the countries###

df['PISA.MAT_growth'] = df[["2012", "2000"]].apply(
    lambda row: (row.iloc[0] - row.iloc[1]) / row.iloc[0] * 100, axis=1)
df['PISA.MAT_last_year_growth'] = df[["2012", "2009"]].apply(
    lambda row: (row.iloc[0] - row.iloc[1]) / row.iloc[0] * 100, axis=1)

# the projection data frame with variables to include
projection = ["Country Name", "2000", "2012", "PISA.MAT_growth"]

# creating the new data frame which include the top and bottom 6 with highest and lowest growth between the years span

pisa_mat_total = df[(df["Indicator Code"] == search_cols) & (df["PISA.MAT_growth"].notna())]
# top6:
top6_overall = pisa_mat_total.sort_values("PISA.MAT_growth", ascending=False)[
    projection].head(number_of_countries)
update_list_countries(top6_overall)  # update list

# bottom 6:
bottom6_overall = pisa_mat_total.sort_values(
    "PISA.MAT_growth")[projection].head(number_of_countries)


update_list_countries(bottom6_overall)

# final list of countries:
countries_to_analyse = []
for i in list_countries:
    if i not in countries_to_analyse:
        countries_to_analyse.append(i)
        
        
final_df = df_filter_ind[
    (df_filter_ind["Country Name"] == countries_to_analyse[0]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[1]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[2]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[3]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[4]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[5]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[6]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[7]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[8]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[9]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[10]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[11]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[12]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[13]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[14]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[15]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[16]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[17]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[18]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[19]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[20]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[21]) |
    (df_filter_ind["Country Name"] == countries_to_analyse[22])
]
