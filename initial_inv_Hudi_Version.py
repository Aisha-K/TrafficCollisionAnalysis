
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Tarun_Work import traffic_lights
plt.style.use('seaborn-white')


#-----------#---------#---------#----------Vars
col_list_categorical=['SPEEDING','AG_DRIV','ALCOHOL','DISABILITY']

mapping={'unknown':0,'0 to 4':1, '5 to 9':2,'10 to 14':3, '15 to 19':4,'20 to 24':5, '25 to 29':6,'30 to 34':7, '35 to 39':8,'40 to 44':9, 
'45 to 49':10, '50 to 54':11, '55 to 59':12,'60 to 64':13,'65 to 69':14,'70 to 74':15,'75 to 79':16,'70 to 74':17,'75 to 79':18,'80 to 84':19,
'85 to 89':20,'90 to 94':21,'Over 95':22}

mapping2={'other':'unknown', '':'unknown', r'^\s*$':'unknown'}

mapping3={'None':0,'Minimal':1, 'Minor':1,'Major':2,'Fatal':3, '':0, np.nan:0}

inv_type_unwanted=['Witness','Pedestrian - Not Hit', 'Trailer Owner', 'In-Line Skater', 'Wheelchair','Driver - Not Hit', 'Runaway - No Driver', 
'','Other','Cyclist Passenger','Moped Driver','Motorcycle Passenger', None]

colors_five=['#E13F29', 'lightseagreen','lightskyblue', 'lightyellow', 'limegreen']

col_unwanted=['Division','INITDIR','ObjectId','VEHTYPE','YEAR','TIME','STREET1','STREET2','OFFSET','District','ACCLOC','LIGHT','LOCCOORD', 'RDSFCOND','VISIBILITY']

##-----------#---------#---------#----------DATA CLEANING FUNCTIONS

def drop_rows(df, column, row_vals, regex_b=False):
    new_df = df   
    #dataframe dropna won't replace empty values only NaN and NaT so convert blank space to NaN then drop
    new_df[column].replace(row_vals, np.nan, inplace=True, regex=regex_b) #dropna will take care of np.nan
    new_df = new_df.dropna(subset=[column])
    return new_df

def drop_unwanted(orig_df):
        inv_df=orig_df.drop(orig_df.columns[26:41], axis=1)
        inv_df.drop(col_unwanted, axis=1, inplace=True)
        #inv_df.drop(['Division','REDLIGHT','INITDIR','ObjectId','VEHTYPE'], axis=1, inplace=True)
        return inv_df

#-----------#---------#---------#----------DATA CLEANING
orig_df=pd.read_excel('KSI_data.xlsx')
inv_df=drop_unwanted(orig_df)

#filling in blanks/replacing
inv_df['INJURY'].replace(r'^\s*$',np.nan, regex=True, inplace=True) #blanks
for col_r in col_list_categorical:
        inv_df[col_r].replace({'Yes':1,'No':0, r'^\s*$':0}, regex=True, inplace=True) #blanks

inv_df['MANOEUVER'].replace(['Other','',r'^\s*$' ], 'Unknown', inplace=True, regex=True) 
inv_df['ROAD_CLASS'].replace(['Pending','',r'^\s*$' ], 'Other', inplace=True, regex=True) 

#dropping unwanted rows
inv_df['INVTYPE'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
drop_rows(inv_df, 'INVTYPE', inv_type_unwanted)


#mapping age categories to ordered ints
inv_df.replace({'INVAGE':mapping}, inplace=True)
inv_df.replace({'INJURY':mapping3}, inplace=True)


#convert columns to appropriate types with astype()
inv_df.INVAGE.astype('int32', copy=False)
inv_df.INJURY.astype('int32', copy=False)
inv_df.INVTYPE.astype(str)

print(inv_df.head())
print('\n')


#-----------#---------#---------#----------FUNCTIONS

#counts the number of occurences of unique values in a column, shown as a bar graph
def frequency_counts_graph(col_name, df, sort=False):
    temp=freq_counts(col_name,inv_df, sort)
    plt.figure()
    plt.title(col_name)
    temp.plot(kind='bar')

#returns counts of unique values of a column as a series
def freq_counts(col_name, df, sort=False):
    temp=df[col_name].value_counts()
    if sort:
        temp.sort_index(axis=0,inplace=True) 
    return temp 


#show injury types based on categories (each injury type is its own bar graph)
def injury_by_cat(col, df, sort=False):
    injury_type=['Fatal','Major','Minor','Minimal']

    fig=plt.figure()
    x=1

    #make a graph for each injury type
    for i_type in injury_type:
        temp_df=inv_df.loc[inv_df['INJURY']==i_type]

        counts_df=freq_counts(col,temp_df, sort)

        fig.add_subplot(2,2,x)
        plt.title(i_type)
        counts_df.plot(kind='bar')
        x=x+1



#breaks down a category into injury types (pie chart for each item in var types, displaying injuries as percentages)
#types is an array containting unique values from a column given by var cat
def cats_split_into_injuries(cat, types, inv_df=inv_df, grid_l=3):
    pos=1
    fig=plt.figure()
    plt.title(cat)

    for i_type in types:

        temp_df=inv_df.loc[inv_df[cat]==i_type]
        counts_df=freq_counts('INJURY',temp_df, True)

        fig.add_subplot(grid_l,grid_l,pos)
        plt.title(i_type)
        colors=colors_five

        plt.pie(
            counts_df.as_matrix(),  #value_counts returns a series, so need to use as_matrix for array rep
            labels=counts_df.index.values,
            colors=colors,
            startangle=90,
            autopct='%1.1f%%'#,
            #explode=(0,0,0,0,0)
        )
        plt.axis('equal')
        plt.tight_layout()

        pos=pos+1


#each vehicle is a new graph, each graph has the age as the x axis, counts as the y, and color showing injury types
def injury_by_age_invtype(inv_df=inv_df):
    inv_types=['Driver','Pedestrian','Cyclist','Motorcycle Driver','Passenger']
    x=1
    fig=plt.figure()
    

    for i_type in inv_types:
        fig.add_subplot(3,3,x)
        x=x+1
        plt.title(i_type)
        temp_df=inv_df.loc[inv_df['INVTYPE']==i_type]

        for key, grp in temp_df.groupby(['INJURY']):
            temp_counts=freq_counts('INVAGE', grp, True).rename(key)
            #add 0s if no counts found
            temp_counts=temp_counts.reindex(list(range(22)), fill_value=0)
            temp_counts.plot()

        #sum of all injury types
        total_counts=freq_counts('INVAGE',temp_df, True).rename('total').reindex(list(range(22)), fill_value=0)
        total_counts.plot()
              
        plt.legend(loc='best')



#plots longitude and latitude for the soecfied inv types
def scatter_inv_type(df, inv_types):
    fig=plt.figure()

    for i_type in inv_types:
        temp_df=df.loc[df['INVTYPE']==i_type]        
        plt.scatter(temp_df['LONGITUDE'], temp_df['LATITUDE'], alpha=0.5, s=2)
    plt.legend(loc='best')


#------#---------#---------#------------DISPLAY

def display_freq_graphs():
        frequency_counts_graph('INVTYPE', inv_df, False)
        frequency_counts_graph('INJURY', inv_df, False)
        frequency_counts_graph('INVAGE', inv_df, True)
        frequency_counts_graph('SPEEDING', inv_df, True)
        frequency_counts_graph('MANOEUVER', inv_df, False)



#for key, grp in inv_df.groupby(['INVTYPE']):
        #print(freq_counts('MANOEUVER', grp, True).rename(key))


def display_injury_percentages():
    cats_split_into_injuries('INVTYPE', ['Driver','Pedestrian','Cyclist','Motorcycle Driver','Passenger'])
    cats_split_into_injuries('MANOEUVER',inv_df['MANOEUVER'].unique() , inv_df, 4)
    cats_split_into_injuries('ALCOHOL',[1,0], inv_df, 2 )
    cats_split_into_injuries('AG_DRIV', [1,0] , inv_df, 2)
    cats_split_into_injuries('SPEEDING', [1,0] , inv_df, 2)
    cats_split_into_injuries('DISABILITY', [1,0] , inv_df, 2)
    cats_split_into_injuries('ROAD_CLASS', inv_df['ROAD_CLASS'].unique() , inv_df, 4)



#display_freq_graphs()
#injury_by_cat('INVAGE', inv_df,True)
#injury_by_cat('Hood_ID', inv_df)
#display_injury_percentages()
#injury_by_age_invtype()
#scatter_inv_type(inv_df, ['Cyclist', 'Pedestrian', 'Motorcycle'])


#print(inv_df['ROAD_CLASS'].value_counts())
#print(inv_df['TRAFFCTL'].value_counts())

plt.show()


print('\n\nALL COLUMNS')
print(list(inv_df))

def print_freq():
   for l in list(inv_df):
       if (l !='LONGITUDE') & (l!='LATITUDE'):
            print(freq_counts(l,inv_df))
            print('\n')

#print_freq()

def is_fatal(row):
    if row['INJURY'] == 3:
        return 1
    else:
        return 0

#------------------------------------#TODO 

#inv type, zone id, and injury type counts

#supervised clustering for injury types
        #USING COLS: [INVTYPE, MANOEUVER, INVAGE, ALCOHOL, DISABILITY, ]
        # road class?, hood


#accident rate based on density of location, w/ other city datasets

#-----------------------------------Very quickly made random forest model, basic test, NOT TO BE USED FOR ANYTHNG

from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestRegressor 

rf_df=inv_df.drop( ['LATITUDE','LONGITUDE','DATE','ACCNUM', 'ACCLASS'], axis=1)
rf_df['TRAFFCTL'] = rf_df['TRAFFCTL'].apply(traffic_lights)
rf_df=rf_df.dropna()

print(rf_df.head())

types = pd.get_dummies(rf_df['INVTYPE'], prefix = 'type')

rf_df.drop('INVTYPE', inplace=True, axis=1)
rf_df = pd.merge(rf_df, types, left_index = True, right_index = True)

def RF_Verbose(): 
    lblE=LabelEncoder()
    for i in rf_df:
        if rf_df[i].dtype==type(object):
            print(i)
            lblE.fit(rf_df[i].astype(str))
            rf_df[i]=lblE.transform(rf_df[i])

    x_train, x_test, y_train, y_test= ms.train_test_split(rf_df.drop('INJURY', axis=1), rf_df.INJURY, test_size=0.33, random_state=42)

    
    m=RandomForestRegressor(n_estimators=50, oob_score=True)
    m.fit(x_train, y_train)
    
    plt.scatter(m.predict(x_test), y_test, alpha = 0.1)
    plt.show()
    
    
    print('\n R^2')
    print(m.score(x_test, y_test))
    print('\n FEATURE IMPORTANCE')
    print(m.feature_importances_)
    print('\n OOB_SCORE:')
    print(m.oob_score_)

    print('\n TESTING FIRST TEN PREDICTONS')
    print(m.predict(x_test.head(10)))
    print(y_test.head(10))
    
    return m.score(x_test, y_test)

def RF():
    lblE=LabelEncoder()
    for i in rf_df:
        if rf_df[i].dtype==type(object):
            lblE.fit(rf_df[i].astype(str))
            rf_df[i]=lblE.transform(rf_df[i])

    x_train, x_test, y_train, y_test= ms.train_test_split(rf_df.drop('INJURY', axis=1), rf_df.INJURY, test_size=0.33, random_state=42)

    m=RandomForestRegressor(n_estimators=50, oob_score=True)
    m.fit(x_train, y_train)
    
    return m.score(x_test, y_test)
    
def average_trials(numTrials):
    total = 0
    for i in range(numTrials):
        total += RF()
    return(total/numTrials)
        
RF_Verbose()

