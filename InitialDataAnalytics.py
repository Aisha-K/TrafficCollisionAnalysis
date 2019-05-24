# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:28:46 2019

@author: hudil
"""
import re
import pandas as pd
import matplotlib.pyplot as plt

"""

Helper Functions

"""

def normalize_traffic_lights(input):
    if input == " ":
        return "No Control"
    if re.search("^Traffic.*", input) != None:
        return "Traffic Light"
    else:
        return input

def normalize_road_types(input):
    if input in ('Pending', ' '):
        return "Other"
    else:
        return input

def format_acc_class(input):
    if input == "Fatal":
        return 1
    else:
        return 0

def output_incident_table(file_name):    
    df = pd.read_excel(file_name)
    df.drop("OFFSET", axis = 1, inplace=True) 
    df.rename(columns={"TIME":"MINUTES"}, inplace=True)

    df["MINUTES"] = df["MINUTES"]%100
    df["TRAFFCTL"] = df["TRAFFCTL"].map(normalize_traffic_lights)
    df["ROAD_CLASS"] = df["ROAD_CLASS"].map(normalize_road_types)
    df["ACCLASS"] = df["ACCLASS"].map(format_acc_class)
    df.rename(columns={"ACCLASS": "IS_FATAL"}, inplace= True)
    df = df.drop_duplicates(['ACCNUM'])
    df.drop(incidentDf.columns[19:-3], axis=1, inplace=True)
    df.drop(incidentDf.columns[-1], axis = 1, inplace=True)
    
    return df

incidentDf = output_incident_table('KSI_data.xlsx')

print(incidentDf.columns.tolist())

print(incidentDf['VISIBILITY'].unique().tolist())

years = incidentDf["YEAR"].value_counts().sort_index()


plot = years.plot(kind='bar')

print(incidentDf.dtypes)




    
    

