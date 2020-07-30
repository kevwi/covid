import os
import csv
import sys
import wget
import shutil
import numpy as np
import pandas as pd
from os import path
from finta import TA
from math import floor
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from pathlib import Path

today = datetime.now()
print("Today's date:", today)

#import pandas_bokeh
#pandas_bokeh.output_notebook()
#pd.set_option('plotting.backend', 'pandas_bokeh')
import warnings
warnings.filterwarnings('ignore')

#pd.set_option('display.max_rows', 5000)

home_path = str(Path.home())
original_path = os.getcwd()

#home_state = "Michigan"
#home_county = "Wayne"
#home_state = "California"
#home_county = "Los Angeles"
#home_state = "New York"
#home_county = "New York"
#home_state = "Texas"
#home_county = "Denton"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("home_state")
parser.add_argument("home_county")
args = parser.parse_args()
home_state = args.home_state
home_county = args.home_county

caseName = "Cases"
dName = "Deaths"
cur_date = datetime.now()
yester_date = cur_date - timedelta(days=1)
yester_date = yester_date.strftime('%Y%m%d%M%H%S')

downloads_path = os.path.join(home_path,"Downloads")
base0_path = os.path.join(home_path,"Downloads/COVID-19")
base_path = os.path.join(base0_path,"csse_covid_19_data/csse_covid_19_time_series/")
base_archive = os.path.join(base_path, "archive")

fileGlobalConfirmed = "time_series_covid19_confirmed_global"
fileUSConfirmed = "time_series_covid19_confirmed_US"
fileGlobalDeaths = "time_series_covid19_deaths_global"
fileUSDeaths = "time_series_covid19_deaths_US"
_csvFilenameGlobalConfirmed = os.path.join(base_path, fileGlobalConfirmed)
_csvFilenameUSConfirmed = os.path.join(base_path, fileUSConfirmed)
_csvFilenameGlobalDeaths = os.path.join(base_path, fileGlobalDeaths)
_csvFilenameUSDeaths = os.path.join(base_path, fileUSDeaths)

file_suffix = ".csv"
file1 = fileGlobalConfirmed
file2 = fileUSConfirmed
file1b = fileGlobalDeaths
file2b = fileUSDeaths
file1full = file1 + file_suffix
file2full = file2 + file_suffix
file1bfull = file1b + file_suffix
file2bfull = file2b + file_suffix
file1date = file1 + "_" + yester_date + file_suffix
file2date = file2 + "_" + yester_date + file_suffix
file1bdate = file1b + "_" + yester_date + file_suffix
file2bdate = file2b + "_" + yester_date + file_suffix
file1path = os.path.join(base_path,file1full)
file2path = os.path.join(base_path,file2full)
file1bpath = os.path.join(base_path,file1bfull)
file2bpath = os.path.join(base_path,file2bfull)

if os.path.exists(base0_path):
    shutil.rmtree(base0_path)

os.chdir(downloads_path)
#os.system("git checkout .")
os.system("git clone https://github.com/CSSEGISandData/COVID-19.git")
#os.system("git reset --hard origin/master")
os.chdir(base_path)

def rewrite_with_doublequotes(fileName=_csvFilenameGlobalConfirmed):
    with open(fileName+file_suffix, 'r', newline='') as f_input, open(fileName+"_output"+file_suffix, 'w', newline='') as f_output:
        csv_input = csv.reader(f_input)
        csv_output = csv.writer(f_output, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)

        for row_input in csv_input:
            row_output = []
            for col in row_input:
                row_output.append(col)
            csv_output.writerow(row_output)

rewrite_with_doublequotes(_csvFilenameGlobalConfirmed)
rewrite_with_doublequotes(_csvFilenameUSConfirmed)
rewrite_with_doublequotes(_csvFilenameGlobalDeaths)
rewrite_with_doublequotes(_csvFilenameUSDeaths)

fileNameGlobalConfirmed = _csvFilenameGlobalConfirmed + "_output" + file_suffix
fileNameUSConfirmed = _csvFilenameUSConfirmed + "_output" + file_suffix
fileNameGlobalDeaths = _csvFilenameGlobalDeaths + "_output" + file_suffix
fileNameUSDeaths = _csvFilenameUSDeaths + "_output" + file_suffix

dfGlobalConfirmed = pd.read_csv(fileNameGlobalConfirmed, error_bad_lines=False, doublequote=True, sep=';')

dfGlobalDeaths = pd.read_csv(fileNameGlobalDeaths, error_bad_lines=False, doublequote=True, sep=';')

dfUSConfirmed = pd.read_csv(fileNameUSConfirmed, error_bad_lines=False, doublequote=True, sep=';')

dfUSDeaths = pd.read_csv(fileNameUSDeaths, error_bad_lines=False, doublequote=True, sep=';')

dfGlobalConfirmed = dfGlobalConfirmed[( (dfGlobalConfirmed['Country/Region']!='China') & (dfGlobalConfirmed['Country/Region']!='Diamond Princess') & (dfGlobalConfirmed['Country/Region']!='Grand Princess') )]
dfGlobalDeaths = dfGlobalDeaths[( (dfGlobalDeaths['Country/Region']!='China') & (dfGlobalDeaths['Country/Region']!='Diamond Princess') & (dfGlobalDeaths['Country/Region']!='Grand Princess') )]
dfUSConfirmed = dfUSConfirmed[( (dfUSConfirmed['Country_Region']!='China') & (dfUSConfirmed['Country_Region']!='Diamond Princess') & (dfUSConfirmed['Country_Region']!='Grand Princess') )]
dfUSDeaths = dfUSDeaths[( (dfUSDeaths['Country_Region']!='China') & (dfUSDeaths['Country_Region']!='Diamond Princess') & (dfUSDeaths['Country_Region']!='Grand Princess') )]


# inputs could be: 
#   'Country/Region', dfGlobalConfirmed, dfGlobalDeaths
#    
def get_cases_summary(principality, confirmed, ds):
    if principality == 'Country/Region':
        casessummary = confirmed.iloc[:,[1,-1]].groupby(principality).sum()
        dsummary = ds.iloc[:,[1,-1]].groupby(principality).sum()
        mostrecentdatecases = casessummary.columns[0]
        mostrecentdatecases2 = mostrecentdatecases + " cases"
        mostrecentdateds = dsummary.columns[0]
        casessummary = casessummary.rename(columns={mostrecentdatecases: mostrecentdatecases2})
        dsummary = dsummary.rename(columns={mostrecentdateds: dName})
        casessummary = casessummary.join(dsummary)
        casessummary['Death_rate_pct'] = round((casessummary[dName]/casessummary[mostrecentdatecases2])*100,2)

        casessummary = casessummary[casessummary[mostrecentdatecases2] >= 100]
        casessummary = casessummary.sort_values(by=casessummary.columns[0], ascending = False)
        summary = casessummary.dropna().sort_values(by='Deaths', ascending=False)[:7]
    elif principality == 'Province_State':
        uscases = confirmed
        usds = ds
        # US states lookup from https://code.activestate.com/recipes/577305-python-dictionary-of-us-states-and-territories/
        # with DC added
        states = { 'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AS': 'American Samoa', 'AZ': 'Arizona', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'GU': 'Guam', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'MP': 'Northern Mariana Islands', 'MS': 'Mississippi', 'MT': 'Montana', 'NA': 'National', 'NC': 'North Carolina', 'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'PR': 'Puerto Rico', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VI': 'Virgin Islands', 'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming', 'D.C.': 'District of Columbia'}
        for index, row in uscases.iterrows():
            location = row[principality]
            try:
                if ',' in location:
                    result = [x.strip() for x in location.split(',')]
                    statename = states[result[1]]
                    row['State'] = statename
                    uscases.loc[index, principality] = statename
            except:
                print('Error parsing US state:', location)
        for index, row in usds.iterrows():
            location = row[principality]
            try:
                if ',' in location:
                    result = [x.strip() for x in location.split(',')]
                    statename = states[result[1]]
                    row['State'] = statename
                    usds.loc[index, principality] = statename
            except:
                print('Error parsing US state:', location)
        usstatecasesummary = uscases.iloc[:,[0,-1]].groupby(principality).sum()
        usstatedsummary = usds.iloc[:,[0,-1]].groupby(principality).sum()
        mostrecentdatecases = usstatecasesummary.columns[0]
        mostrecentdatecases2 = mostrecentdatecases + " cases"
        mostrecentdateds = usstatedsummary.columns[0]
        usstatecasesummary = usstatecasesummary.rename(columns={mostrecentdatecases: mostrecentdatecases2})
        usstatecasesummary = usstatecasesummary.sort_values(by = mostrecentdatecases2, ascending = False)
        usstatecasesummary = usstatecasesummary[usstatecasesummary[mostrecentdatecases2] > 0]
        usstatedsummary = usstatedsummary.sort_values(by = mostrecentdateds, ascending = False)
        usstatedsummary = usstatedsummary[usstatedsummary[mostrecentdateds] > 0]

        usstatedsummary = usstatedsummary.rename(columns={mostrecentdateds: dName})
        usstatecasesummary = usstatecasesummary.join(usstatedsummary)
        usstatecasesummary['Death_rate_pct'] = round((usstatedsummary[dName]/usstatecasesummary[mostrecentdatecases2])*100,2)
        summary = usstatecasesummary.dropna().sort_values(by='Deaths', ascending=False)[:7]
    elif principality == 'County':
        statecasesummary = confirmed.iloc[:,[0,-1]].groupby(principality).sum()
        statedsummary = ds.iloc[:,[0,-1]].groupby(principality).sum()
        mostrecentdatecases = statecasesummary.columns[0]
        mostrecentdatecases2 = mostrecentdatecases + " cases"
        mostrecentdateds = statedsummary.columns[0]
        statecasesummary = statecasesummary.rename(columns={mostrecentdatecases: mostrecentdatecases2})
        statecasesummary = statecasesummary.sort_values(by = mostrecentdatecases2, ascending = False)
        statecasesummary = statecasesummary[statecasesummary[mostrecentdatecases2] > 0]
        statedsummary = statedsummary.sort_values(by = mostrecentdateds, ascending = False)
        statedsummary = statedsummary[statedsummary[mostrecentdateds] > 0]
        statedsummary = statedsummary.rename(columns={mostrecentdateds: dName})
        statecasesummary = statecasesummary.join(statedsummary)
        statecasesummary['Death_rate_pct'] = round((statedsummary[dName]/statecasesummary[mostrecentdatecases2])*100,2)
        summary = statecasesummary.dropna().sort_values(by='Deaths', ascending=False)[:7]
    return summary, mostrecentdatecases

# could be 'Country/Region', dfGlobalConfirmed, dfGlobalDeaths 
# could be 'Province_State', dfUSConfirmed, dfUSDeaths 
# could be 'County', dfUSConfirmed, dfUSDeaths 
def get_cases_ds(principality, dfCases, dfDs):
    if principality == 'Country/Region':
        dfCases = dfCases.drop(columns=['Province/State','Lat','Long'])
        cases = dfCases.copy()
        dfDs = dfDs.drop(columns=['Province/State','Lat','Long'])
        ds = dfDs.copy()
    elif principality == 'Province_State':
        cases = dfCases.drop(columns=['UID','iso2','iso3','code3','FIPS','Admin2','Combined_Key','Country_Region','Lat','Long_']).copy()
        ds = dfDs.drop(columns=['UID','iso2','iso3','code3','FIPS','Admin2','Combined_Key','Country_Region','Lat','Long_','Population']).copy()
    elif principality == 'County':
        dfCases = dfCases[( (dfCases['Province_State']==home_state) & (dfCases['Lat']>0.0) )]
        dfCases = dfCases.drop(columns=['UID','iso2','iso3','code3','FIPS','Combined_Key','Province_State','Country_Region','Lat','Long_'])
        cases = dfCases.rename(columns={"Admin2": principality}).copy()
        dfDs = dfDs[( (dfDs['Province_State']==home_state) & (dfDs['Lat']>0.0) )]
        dfDs = dfDs.drop(columns=['UID','iso2','iso3','code3','FIPS','Combined_Key','Province_State','Country_Region','Lat','Long_','Population'])
        ds = dfDs.rename(columns={'Admin2': principality}).copy()
    return cases, ds

# casegroup = cases.groupby(principality).sum().reset_index()
# meltee = casegroup, principality = 'Province_State'
# valueName = either caseName or dName
def melt_dataframes_date(meltee, principality = 'Country/Region', valueName = caseName):
    meltCases = pd.melt(meltee,id_vars=[principality],value_vars=meltee.columns[1:len(meltee.columns)].tolist(),var_name='Date',value_name=valueName)
    meltCases['date'] = pd.to_datetime(meltCases['Date'], format='%m/%d/%y')
    mc = meltCases.sort_values(by=[principality,'date'], ascending=[True,True]).reset_index().drop(columns=['index','Date'])
    return mc

def get_joined_melted_cases_deaths(mc, md, principality):
    joinedDF = pd.merge(mc, md, how='left', on=[principality, 'date'])
    #joinedDF = joinedDF[(joinedDF['Ds']>0)]
    #joinedDF['D_Rate_pct'] = round((joinedDF['Ds']/joinedDF['Cases'])*100,2)
    joinedDF['principality'] = joinedDF[principality]
    joinedDF[dName] = joinedDF[dName]/1.0
    joinedDF['principality_shift_past1'] = joinedDF[principality].shift(-1)
    joinedDF['principality_shift_past2'] = joinedDF[principality].shift(-2)
    joinedDF['principality_shift_future1'] = joinedDF[principality].shift(1)
    joinedDF['Ds_past1'] = joinedDF[dName].shift(-1)
    joinedDF['Ds_past2'] = joinedDF[dName].shift(-2)
    joinedDF['Ds_future1'] = joinedDF[dName].shift(1)
    #joinedDF['low'] = np.where(((joinedDF.principality_shift_past2.notnull()) & (joinedDF.principality_shift_past2 == joinedDF.principality)), joinedDF.Ds_past2, False)
    #joinedDF['open'] = np.where(((joinedDF.principality_shift_past1.notnull()) & (joinedDF.principality_shift_past1 == joinedDF.principality)), joinedDF.Ds_past1, False)
    #joinedDF['close'] = joinedDF[dName]
    #joinedDF['high'] = np.where(((joinedDF.principality_shift_future1.notnull()) & (joinedDF.principality_shift_future1 == joinedDF.principality)), joinedDF.Ds_future1, False)
    #joinedDF['volume'] = joinedDF[caseName] - joinedDF[caseName].shift(1)
    #joinedDF = joinedDF[( (joinedDF['principality_shift_past2']==joinedDF[principality]) & (joinedDF[caseName]>=3) & (joinedDF['high']!=0.00) & (joinedDF['close']!=0.00) )]
    joinedDF = joinedDF[( (joinedDF['principality_shift_past2']==joinedDF[principality]) & (joinedDF[caseName]>=3) )]
    joinedDF = joinedDF.drop(columns=['principality_shift_past1','principality_shift_past2','principality_shift_future1','Ds_past1','Ds_past2','Ds_future1','principality'])
    return joinedDF.dropna()

#principalities include:  'Country/Region', 'Province/State', 'County'
#localNames include: 'US', 'California'
def dfLocale(dfSource, principality='Country/Region', localeName='US', interval='1d', multiplier=10.0):
    df = dfSource.copy()
    df = df[( (df[principality]==localeName) )]
    #df = df.drop(columns=['Dividends','Stock Splits'])
    #df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    #df['AO_TYPICAL'] = TA.AO(df,34,5)
    #df['AO'] = TA.AO(df,floor(34*multiplier),floor(5*multiplier))
    #if (interval == '1d'):
        #vama_period = 14
    #else:
        #vama_period = 30
    #df['VAMA'] = TA.VAMA(df,vama_period)
    #df['STOCH'] = TA.STOCH(df,period=14)
    #df['STOCHD'] = TA.STOCHD(df,period=3,stoch_period=14)
    #df['CaseVar'] = (df['Cases']-df['Cases'].shift(1))/df['Cases'].shift(1)
    df['DeathsDiff'] = (df['Deaths']-df['Deaths'].shift(1))
    df['CaseDiff'] = (df['Cases']-df['Cases'].shift(1))
    #df['Rolling14Average'] = (df['CaseDiff'].shift(0)+df['CaseDiff'].shift(1)+df['CaseDiff'].shift(2)+df['CaseDiff'].shift(3)+df['CaseDiff'].shift(4)+df['CaseDiff'].shift(5)+df['CaseDiff'].shift(6)+df['CaseDiff'].shift(7)+df['CaseDiff'].shift(8)+df['CaseDiff'].shift(9)+df['CaseDiff'].shift(10)+df['CaseDiff'].shift(11)+df['CaseDiff'].shift(12)+df['CaseDiff'].shift(13))/14.0
    df['Rolling7Average'] = (df['CaseDiff'].shift(0)+df['CaseDiff'].shift(1)+df['CaseDiff'].shift(2)+df['CaseDiff'].shift(3)+df['CaseDiff'].shift(4)+df['CaseDiff'].shift(5)+df['CaseDiff'].shift(6))/7.0
    #df['Rolling3Average'] = (df['CaseDiff'].shift(0)+df['CaseDiff'].shift(1)+df['CaseDiff'].shift(2))/3.0
    #df['Rolling1Average'] = (df['CaseDiff'].shift(0)+df['CaseDiff'].shift(1))/2.0
    #df['R0'] = round((df['Rolling14Average'].shift(1)*2).astype(float),2)
    return df

