import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as skp
import sklearn.metrics as skm

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

#----------------------------------Machine learning, Linear Regression using scikit learn
#predict number of future KSI collisions
#features: weekday, month, hour label: number of collisions


def output_incident_table(file_name):    
    df = pd.read_excel(file_name)
    df.rename(columns={"TIME":"MINUTES"}, inplace=True)

    df["MINUTES"] =df["MINUTES"]%100
    df["MONTH"]=pd.DatetimeIndex(df['DATE']).month
    df["WEEKDAY"]=pd.DatetimeIndex(df['DATE']).weekday

    df = df.drop_duplicates(['ACCNUM'])
    df=df[['YEAR','Hour','MONTH','WEEKDAY','DATE', 'ACCNUM']]
    return df

df = output_incident_table('KSI_data.xlsx')
print(list(df))
print(df.head())

#Aggregate df so each row is a specific day (year,month,day) with column ACCNUM nor repr number of accidents that day
grouped_df=df.groupby(['DATE']).agg({'ACCNUM':'count','YEAR':'first','MONTH':'first','WEEKDAY':'first'})
print(grouped_df.head())



#TODO:
def cyclical_features_transform(df):
    df['hr_sin']=np.sin(df.Hour*(2.*np.pi/24))


#TODO: one hot encode categorical data


def runModel(df):
    #split and train the data
    x_train, x_test, y_train, y_test= ms.train_test_split(df.drop('ACCNUM', axis=1), df.ACCNUM, test_size=0.2)
    #rescale
    x_train[:,2]=skp.StandardScaler().fit(x_train[:,2:])
    model=LinearRegression(normalize=False).fit(x_train, y_train)

    #evaluate model
    predictions=model.predict(x_test)
    print(y_test.head(10))
    print(predictions[0:10])
    printErrorMetrics(y_true=y_test, predictions=predictions)
    

def printErrorMetrics(y_true, predictions):
    print('MAE:'+ str(skm.mean_absolute_error(y_true, predictions)))
    print('R2:'+ str(skm.r2_score(y_true,predictions)))


runModel(grouped_df)

#TODO: Next Steps
#Filter predictions by division
#filter by hour to predict only wether a crash will happen or not (0 or 1)

