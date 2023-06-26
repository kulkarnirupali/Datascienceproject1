# This is data science Project for the prediction of Home Prices in Banglore Using Various Module such as Sklearn,Pndas,Matplotlip,flask,etc
import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df1 = pd.read_csv('bengaluru_house_prices.csv')
print(df1.head(3))
print(df1.shape)
print(df1.groupby('area_type')['area_type'].agg('count'))

df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
print(df2.head(3))

# Step 2 : Data cleaning

print(df2.isnull().sum())
# Dropping All the null values columns to make dataset more Clean
df3 = df2.dropna()
sum = df3.isnull().sum()
print(sum)
print(df3.head(3))

# In the feature named as size Bedroom & bhk are used for the same meaning so adding new column to solve this .
print(df3['size'].unique())
df3['bhk']=df3['size'].apply(lambda x:int(x.split(' ')[0]))
print(df3.head(3))
print(df3['bhk'].unique())


# In the column total-Square-foot-Area we can observe that few value are present in the form of range,few wit suffix as squarefoot,few valuewith perch as a suffix to solve this data cleaning is necessary by taking median of the range value.

# To check value wheather the value is float or int using is_float function:

def is_float(x):
    try:
        float(x)
    except:
        return False

    return True

print(df3[~df3['total_sqft'].apply(is_float)].head(10))

# Using function named as convert_sqrt_to_num to take mean of range values

def convert_sqrt_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

print(convert_sqrt_to_num('2166'))



# Applying this function to the original dataframe

df4 = df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqrt_to_num)
print(df4.head(3))

# Adding New column 'Price per square foot' in the dataframe

df5 =df4.copy()
df5['Price_per_sqft']=df5['price']*100000/df5['total_sqft']
print(df5.head(3))


# To check unique values in the column 'Location'
print(len(df5.location.unique()))

# To remove extraa spaces betn these individual value in the location columns
location_stats= df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

# Filtering Location value that has less than 10 datapoints
print(len(location_stats[location_stats<=10]))

# keeping this less than 10 datapoints values into the category to make more dataset in the 'other category'
location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

df5.location = df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
print(df5.location.unique())
print(len(df5.location.unique()))

# Step 3 : To remove All the outliers from the dataframe
# By observing excel file ,conclude that The ideal squareft area of a single room is 300 sqaurefoot around so if in the given dataset if the area is less than 300 s.f per room the it can creat a error idealy so removing such points from the dataframe.

print("The rooms where sq.ft area is less than 300 sq.f")
print(df5[df5.total_sqft/df5.bhk<300].head(3))

# After removal of such outliers
df6 = df5[~(df5.total_sqft/df5.bhk<300)]

# Checking another outliers removal criteria in the price columns
print(df6.Price_per_sqft.describe())

# Find out values of mean deviation & Standard deviation for the feature Location so that removing extreme datapoints can make clean dataset for the further use using fallowing function :

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.Price_per_sqft)
        st = np.std(subdf.Price_per_sqft)
        reduced_df = subdf[(subdf.Price_per_sqft >(m-st)) & (subdf.Price_per_sqft <=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)

    return df_out

df7 = remove_pps_outliers(df6)
print(df7.shape)

# To plot graph to see variation between bedroom and House Price

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 =df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.Price_per_sqft,color='blue',label='2bhk',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.Price_per_sqft,color='green',label='3bhk',s=50)
    plt.xlabel('Total Square foot Area')
    plt.ylabel('Price per square foot')
    plt.title(location)
    plt.legend
    plt.show()


print(plot_scatter_chart(df7,"Rajaji Nagar"))

# Removing datapoints where for the same location ,the price of (for example) 3 bedrooms apartments is less than 2 bedrooms apartments (with same sqft area)and for that building a dictionary of stats per bhk so that removing those 2bhk apartments whose price per sqft is less than mean price per sqft of 1bhk aprtments.

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.Price_per_sqft),
                'std':np.std(bhk_df.Price_per_sqft),
                'count':bhk_df.shape[0]
            }
    for bhk,bhk_df in location_df.groupby('bhk'):
        stats =bhk_stats.get(bhk-1)
        if stats and stats['count']>5:
            exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.Price_per_sqft<(stats['mean'])].index.values)

    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)
print(df8.shape)
print(plot_scatter_chart(df8,'Hebbal'))
plt.show()

# Unique values in 'bathroom' feature
print(df8.bath.unique())
print(df8[df8.bath>10])

# Histogram of number of bathrooms and count
plt.hist(df8.bath,rwidth=0.8)
plt.xlabel('Number of bathrooms')
plt.ylabel('count')
plt.show()

# When the number of bathrooms are more than bedrooms it can be removed
print(df8[df8.bath>df8.bhk+2])
df9 = df8[df8.bath<df8.bhk+2]
print(df9.shape)

# Droping some unneccasary feature to train the model
df10 = df9.drop(['size','Price_per_sqft'],axis='columns')
print(df10.head())

# Step 4 : Training of dataset using Diffrent diffrent Algorithms:

# Before training of dataset One hot Encoding is necessary to convert all the string value into Integer

dummies = pd.get_dummies(df10.location)
print(dummies.head(3))

df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
print(df11.head(3))

df11 = df11.drop('location',axis='columns')
print(df11.head(3))

x = df11.drop('price',axis='columns')
y = df11.price
print(x.head(3))

# Splitting of dataset into Training & Testing part
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(len(x_train))

le_clf = LinearRegression()
le_clf.fit(x_train,y_train)
Accuracy =le_clf.score(x_test,y_test)
print("Accuracy of the dataset Using Linear Regression is",Accuracy)

# To check cross_val_score for k-fold validation :
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
print(cross_val_score(LinearRegression(),x,y,cv=cv))

# To check Accuracy Using diffrent algorithms of Machine Learning :
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_fit_using_gradientcv(x,y):
    algos ={
        'Linear Regression':{'model':LinearRegression(),'Params':{'normalize':[True,False]}},
        'Lasso':{'model':Lasso(),'Params':{'alpha':[1,2],'selection':['random','cyclic']}},
        'Decision Tree':{'model':DecisionTreeRegressor(),'Params':{'criterion':['mse','friedman-mse'],'splitter':['best','random']}}
    }

    scores = []
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs = GridSearchCV(config['model'],config['Params'],cv=cv,return_train_score=False)
        gs.fit(x,y)
        scores.append({'model':algo_name,'best_score':gs.best_score_,'best_Params':gs.best_params_})

        return pd.DataFrame(scores,columns=['model','best_score','best_params'])

print(find_best_fit_using_gradientcv(x,y))

# From above findings best fit model is clearly Linear Regression model gives best performance for this dataset:

def predict_price(location,sqft,bath,bhk):
    global loc_
    loc_index = np.where(x.columns == location)[0][0]
    loc_ = np.zeros(len(x.columns))
    loc_[0]= sqft
    loc_[1] =bath
    loc_[2]=bhk

    if loc_index>=0:
        loc_[loc_index]=1

    return le_clf.predict([x][0])

print(predict_price('1st Phase JP Nagar',1000,2,2))


# Export Original Model to 'Picklle' model for use of flask server

import pickle
with open('bengaluru_house_prices.csv.pickle','wb') as f:
    pickle.dump(le_clf,f)

# importing JSON module to collect all the info from columns of a dataset

import json
columns = {'data-columns': [col.lower() for col in x.columns]}
with open('Columns.json','w') as f:
    f.write(json.dumps(columns))
