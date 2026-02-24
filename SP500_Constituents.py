# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:18:26 2025

@author: Patrick

Create a DataFrame that for each month displays all the constituents (stock_id = PERMNO)
of the S&P 500.
"""

#%% Libraries
import pandas as pd
import sqlite3

path = "C:/Users/patri/Desktop/ML/"

#%% Monthly & All-Time Constituents

SP500_Constituents = sqlite3.connect(path + "Data/SP500_Constituents.db")

#Read in DataSet
df = pd.read_csv(path + "Data/SP500_Constituents.csv")

#Convert to dates
df["MbrStartDt"] = pd.to_datetime(df["MbrStartDt"])
df["MbrEndDt"] = pd.to_datetime(df["MbrEndDt"])

#Get List of months
month_ends = pd.date_range("1950-01-31", "2024-12-31", freq="M")

#For each Month, filter the stocks that are in the S&P 500
records = []
for _, row in df.iterrows():
    mask = (month_ends >= row["MbrStartDt"]) & (month_ends <= row["MbrEndDt"])
    valid_months = month_ends[mask]
    records.extend([(row["PERMNO"], d) for d in valid_months])

#Create DataFrame
df_sp500 = (pd.DataFrame(records, columns=["PERMNO", "MonthEnd"])
            .rename(columns = {'MonthEnd':'eom'})
            .reset_index(drop = True)
            )

#ID of every stock that is, was or will be in the S&P 500
ids_alltime = pd.Series(df_sp500['PERMNO'].unique(), name='id')

#Save to DataBase
df_sp500.to_sql(name = "SP500_Constituents_monthly", if_exists = 'replace', 
                con = SP500_Constituents, index = False)

ids_alltime.to_sql(name = "SP500_Constituents_alltime", if_exists = "replace",
                   con = SP500_Constituents, index = False)


#%% Cumulative & Forward-looking Constituents

#---- Forward-looking ----
"""
For each eom, include next period's constitutents as well, since at the end
of each month the S&P500 constituents of next month are also known.
"""

#Save first date
first_date = df_sp500.eom.min()

#Lag eom
df_sp500_lag1 = df_sp500.assign(eom = lambda df: df['eom'] + pd.offsets.MonthEnd(-1))

#Merge to dataframe and drop duplicates to create one period look-ahead
df_sp500_fl = (pd.concat([df_sp500,df_sp500_lag1])
            .drop_duplicates(subset = ['PERMNO','eom'])
            #Ensure first date remains the same
            .pipe(lambda df: df.loc[df['eom']>= first_date])
            .sort_values(by = ['eom','PERMNO'])
            )

#---- Cumulative Constituents ----
"""
At each 'eom' contains the list of stocks that are, will be in the next month,
or ever have been in the S&P500.
"""

# Find the very first time an PERMNO appeared in the dataset.
first_appearance = df_sp500_fl.groupby('PERMNO')['eom'].min().reset_index()
first_appearance.rename(columns={'eom': 'start_date'}, inplace=True)

#Create a list of all unique dates in the dataset
all_dates = pd.DataFrame({'eom': df_sp500_fl['eom'].unique()})

# Generate a Cross Join (Cartesian Product)
#   creates a row for every PERMNO for every single Date.
universe_grid = first_appearance.merge(all_dates, how='cross')

#Remove all stocks prior to their start date (no forward-looking bias)
df_sp500_cumulative = universe_grid[universe_grid['eom'] >= universe_grid['start_date']]

#Create an is_active dummy
df_sp500_cumulative = (df_sp500_cumulative.merge(df_sp500.assign(in_sp500 = 1),
                                                on = ['PERMNO','eom'], how = 'left')
                       .assign(is_active=lambda df: df['in_sp500'].fillna(0).astype(bool))
                       )

# Clean up
df_sp500_cumulative = df_sp500_cumulative.sort_values(['eom', 'PERMNO']).reset_index(drop=True)

df_sp500_cumulative.to_sql("SP500_Constituents_FLandExpanding",
                           con = SP500_Constituents,
                           if_exists = 'replace',
                           index = False)

SP500_Constituents.close()
