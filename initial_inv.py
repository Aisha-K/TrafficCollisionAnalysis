import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')



#-----------#---------#---------#----------DATA CLEANING
orig_df=pd.read_excel('KSI_data.xlsx')

#dropping unnecessary columns
inv_df=orig_df.drop(orig_df.columns[1:20], axis=1)
inv_df.drop(inv_df.columns[14:21], axis=1, inplace=True)
inv_df.drop(['Hood_ID', 'Division','REDLIGHT','INITDIR','ObjectId','VEHTYPE'], axis=1, inplace=True)

#filling in blanks/replacing
inv_df['INJURY'].replace(r'^\s*$',np.nan, regex=True, inplace=True)
col_list=['SPEEDING','AG_DRIV','ALCOHOL','DISABILITY']
for col_r in col_list:
        inv_df[col_r].replace({'Yes':1,'No':0, r'^\s*$':0}, regex=True, inplace=True)

#mapping age categories to ordered ints
mapping={'unknown':0,'0 to 4':1, '5 to 9':2,'10 to 14':3, '15 to 19':4,'20 to 24':5, '25 to 29':6,'30 to 34':7, '35 to 39':8,'40 to 44':9, 
'45 to 49':10, '50 to 54':11, '55 to 59':12,'60 to 64':13,'65 to 69':14,'70 to 74':15,'75 to 79':16,'70 to 74':17,'75 to 79':18,'80 to 84':19,
'85 to 89':20,'90 to 94':21,'Over 95':22}
inv_df.replace({'INVAGE':mapping}, inplace=True)

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
def cat_by_injury(cat, types):
    x=1
    fig=plt.figure()

    for i_type in types:

        temp_df=inv_df.loc[inv_df[cat]==i_type]
        counts_df=freq_counts('INJURY',temp_df, True)

        fig.add_subplot(3,3,x)
        plt.title(i_type)
        colors=['#E13F29', 'lightseagreen','lightskyblue', 'lightyellow', 'limegreen']

        plt.pie(
            counts_df.as_matrix(),  #value_counts returns a series, so need to use as_matrix for array rep
            labels=counts_df.index.values,
            colors=colors,
            startangle=90,
            autopct='%1.1f%%',
            explode=(0,0,0,0,0)
        )
        plt.axis('equal')
        plt.tight_layout()

        x=x+1


#each vehicle is a new graph, each graph has the age as the x axis, counts as the y, and color showing injury types
def injury_by_age_invtype():
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

#returns a multi line graph mapping frequency of the col in cols equaling 1, over age      
def binary_cols_over_age(cols, df, ):
    print()



#------#---------#---------#------------DISPLAY

def display_freq_graphs():
        frequency_counts_graph('INVTYPE', inv_df, False)
        frequency_counts_graph('INJURY', inv_df, False)
        frequency_counts_graph('INVAGE', inv_df, True)
        frequency_counts_graph('SPEEDING', inv_df, True)
        frequency_counts_graph('MANOEUVER', inv_df, True)


for key, grp in inv_df.groupby(['INVTYPE']):
        print(freq_counts('MANOEUVER', grp, True).rename(key))


#display_freq_graphs()

#injury_by_cat('INVAGE', inv_df,True)
#injury_by_cat('DRIVCOND', inv_df)

#cat_by_injury('INVTYPE', ['Driver','Pedestrian','Cyclist','Motorcycle Driver','Passenger'])

#injury_by_age_invtype()

plt.show()


print('\n\nALL COLUMNS')
print(list(inv_df))


#maneouver and inv type
#maneouvre and injury type
#speeding, ag_driving, alcohol by age
#inv type, zone id, and injury type counts w/ total
#unsupervised clustering
