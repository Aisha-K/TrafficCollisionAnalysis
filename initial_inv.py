import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')





#-----------#---------#---------#----------DATA CLEANING
df=pd.read_excel('KSI_data.xlsx')

#dropping unnecessary columns
inv_df=df.drop(df.columns[1:20], axis=1)
inv_df.drop(['Hood_ID', 'Division','REDLIGHT','INITDIR'], axis=1, inplace=True)

#filling in blanks
t=inv_df['INJURY'].replace(r'^\s*$',np.nan, regex=True, inplace=True)

#mapping age categories to ordered ints
mapping={'unknown':0,'0 to 4':1, '5 to 9':2,'10 to 14':3, '15 to 19':4,'20 to 24':5, '25 to 29':6,'30 to 34':7, '35 to 49':8,'50 to 54:':9}
inv_df.replace({'INVAGE':mapping})


#-----------#---------#---------#----------FUNCTIONS

#counts the number of occurences of unique values in a column, shown as a bar graph
def count_col(col_name, df):
    temp=df[col_name].value_counts()
    plt.figure()
    plt.title(col_name)
    temp.plot(kind='bar')

#show injury types based on age categories
def injury_by_cat(col):
    injury_type=['Fatal','Major','Minor','Minimal']

    fig=plt.figure()
    x=1

    #make a graph for each injury type
    for i_type in injury_type:
        temp_df=inv_df.loc[inv_df['INJURY']==i_type]

        counts_df=temp_df[col].value_counts()
        #print(counts_df)

        fig.add_subplot(2,2,x)
        plt.title(i_type)
        counts_df.plot(kind='bar')
        x=x+1

#breaks down a category into injury types(x axis)
def cat_by_injury(cat, types):
    x=1
    fig=plt.figure()

    for i_type in types:

        temp_df=inv_df.loc[inv_df[cat]==i_type]
        counts_df=temp_df['INJURY'].value_counts()
        
        fig.add_subplot(3,3,x)
        plt.title(i_type)
        colors=['red','green','blue', 'yellow','white']
        counts_df.columns=["Type","Count"]
        print(counts_df)

        #plt.pie(
            #counts_df['Type'],
            #labels=['None', 'Major', 'Minor', 'Minimal','Fatal'],
            #colors=colors,
            #startangle=90,
            #autopct='%.1f%%',
            #explode=(0,0,0,0,0)
        #)
        #counts_df.plot(kind='bar')
        x=x+1
    




#------#---------#---------#------------DISPLAY


#count_col('INVTYPE', inv_df)
#count_col('INJURY', inv_df)
#count_col('INVAGE', inv_df)
injury_by_cat('INVAGE')
injury_by_cat('DRIVCOND')

cat_by_injury('INVTYPE', ['Driver','Pedestrian','Cyclist','Motorcycle Driver','Passenger'])


plt.show()
print('\n\nALL COLUMNS')
print(list(inv_df))

