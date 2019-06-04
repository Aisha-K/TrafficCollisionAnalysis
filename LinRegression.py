import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as skp
import sklearn.metrics as skm
from sklearn.preprocessing import PolynomialFeatures


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
grouped_df=df.groupby(['DATE']).agg({'ACCNUM':'count','YEAR':'first','MONTH':'first','WEEKDAY':'first'}).reset_index()
grouped_df.drop('DATE', axis=1, inplace=True)


#TODO:
def cyclical_features_transform(df):
    df['day_sin']=np.sin(df.WEEKDAY*(2.*np.pi/7))
    df['day_cos'] = np.cos(df.WEEKDAY*(2.*np.pi/7))
    df['mnth_sin'] = np.sin((df.MONTH-1)*(2.*np.pi/12))
    df['mnth_cos'] = np.cos((df.MONTH-1)*(2.*np.pi/12))
    

cyclical_features_transform(grouped_df)
print(grouped_df.head(20))
grouped_df.drop(['YEAR','MONTH','WEEKDAY'],axis=1,inplace=True)
print(grouped_df.head(20))


def runModel(df):
    #split and train the data
    x_train, x_test, y_train, y_test= ms.train_test_split(df.drop('ACCNUM', axis=1), df.ACCNUM, test_size=0.2)
    poly = PolynomialFeatures(degree=8)

    x_train= poly.fit_transform(x_train)
    x_test = poly.fit_transform(x_test)

    model=LinearRegression(normalize=False).fit(x_train, y_train)


    #evaluate model
    print('EVALUATIONS')
    predictions=model.predict(x_test)
    print(y_test.head(20))
    print(predictions[0:20])
    printErrorMetrics(y_true=y_test, predictions=predictions)
    #predictions.vectorize(round(x,0))
    plt.scatter(predictions[0:800],y_test.head(800), alpha=0.4, s=3)
    plt.show()
    

def printErrorMetrics(y_true, predictions):
    print('MAE:'+ str(skm.mean_absolute_error(y_true, predictions)))
    print('R2:'+ str(skm.r2_score(y_true,predictions)))


runModel(grouped_df)

#TODO: Next Steps
#Filter predictions by division
#filter by hour to predict only wether a crash will happen or not (0 or 1)



