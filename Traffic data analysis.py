import pandas as pd
pd.set_option('mode.chained_assignment', None)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.dates as mdates

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import matplotlib
import matplotlib.dates as mdates

in_tr=r"traffic_data.csv"  
df_tr=pd.read_csv(in_tr,parse_dates=['measurement_start'],usecols=['zone_id','measurement_start','speed'])
print(df_tr.head(5))

in_dr=r"detector_information.csv"  
df_dr=pd.read_csv(in_dr,usecols=['zone_id','latitude','longitude'])
print(df_dr.head())

df=pd.merge(df_tr,df_dr,on=['zone_id'])
print(df.head())

fig,ax = plt.subplots(figsize=(15,7))

plot_data=df[df['zone_id']==10031]

plot_data=plot_data.sort_values('measurement_start')

monthyearFmt = mdates.DateFormatter('%b %d %H')

ax.xaxis.set_major_formatter(monthyearFmt)
ax.plot(plot_data['measurement_start'],plot_data['speed'],label='zone_id:'+str(10031))
ax.set_xlabel('Time of the day')
ax.set_ylabel('Speed in mph')
ax.set_title('Speed Vs Time of the Day for (Normal Day) \n Before Filtering ')

plt.grid()
ax.legend()
plt.tight_layout()

plt.show()
fig.savefig('output/with_outlier.png')

def find_outliers_iqr(x):
    Q1=np.percentile(x,25)
    Q3=np.percentile(x,75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    outlier_ind=list(x.index[(x<lower)|(x>upper)])
    outlier_value=list(x[outlier_ind])
    return outlier_ind,outlier_value

location=df['zone_id'].unique()

df_filter=pd.DataFrame()  #define a emty data frame to keep the clean data sets

for i in location:
    data=df[df['zone_id']==i]
    data=data.sort_values('measurement_start')
    # call the find_outlier function to find the outliers
    outlier_ind, outlier_value=find_outliers_iqr(data['speed'])   
    for i in outlier_ind:
        data['speed'][i]=np.NAN                                #fill the outlier with nan values
    
    # fill the outliers with previous inputs by backfill method
    data['speed']=(data['speed']).fillna(method='bfill')       
    speed=data.filter(['measurement_start','speed'])              
    speed=speed.set_index('measurement_start')
    
    # apply the rolling average
    rolling = speed.rolling(window=3)                           
    rolling_mean = rolling.mean()
    x=list(rolling_mean['speed'])
    data['speed']=x
    df_filter=df_filter.append(data)           #put the clean data into the empty data frame


df_filter=df_filter.sort_values(['longitude'], ascending= False) #sequetially arrange the data

location=df_filter['longitude'].unique()                     #identify unique locations 

print(location)

data=pd.DataFrame()

# iteratively aranging each link speed into upstream speed, downstream speed, and target speed 
# this is done based on the proposed modeling framework

for i in range(len(location)-2):
    df_ul=df_filter[df_filter['longitude']==location[i]]  #upstream link
    df_dl=df_filter[df_filter['longitude']==location[i+2]] #downstream link
    df_tl=df_filter[df_filter['longitude']==location[i+1]]  #target link
    df_final=pd.merge(pd.merge(df_ul,df_dl,on='measurement_start'),df_tl,on='measurement_start')
    df_final=df_final.sort_values('measurement_start')
    data=data.append(df_final)
data.head()

def data_prep(zone_id, model_data,pred_step):
    df_model=model_data[model_data['zone_id']==zone_id]
    df_model['day']=df_model['measurement_start'].dt.day
    df_model['hour']=df_model['measurement_start'].dt.hour
    
    # shift the data 5 (1 step ahead) ,10 (2 step ahead) or 15 min (3 step ahead) ahead of current time for each target zone
    
    df_model['t_speed']=df_model['speed'].shift(-pred_step)                               
    df_model=df_model.dropna(how='any')                                                   #drop any nan values  
    
    train_size=len(df_model[(df_model['measurement_start']>='2017-11-03 00:00:00' ) & (df_model['measurement_start']<'2017-11-08 00:00:00')])
    
    # keeping the datetime for plotting actual and predicted value w.r.t time 
    datetime=df_model['measurement_start'][train_size:]                                 
    
    # drop unneccessary Column variables
    df_model.drop(['zone_id','measurement_start','latitude','longitude'],1,inplace=True)  
    
    # convert into array pr matrix it will work fast  
    #data_arr=df_model.as_matrix(columns=None)

    data_arr = df_model.to_numpy()
    #data_arr = df_model.values()

    return(data_arr,train_size,datetime)

fig,ax = plt.subplots(figsize=(15,7))

plot_data=df_filter[df_filter['zone_id']==10031]
plot_data=plot_data.sort_values('measurement_start')
monthyearFmt = mdates.DateFormatter('%b %d %H')
ax.xaxis.set_major_formatter(monthyearFmt)
ax.plot(plot_data['measurement_start'],plot_data['speed'],label='zone_id:'+str(10031))
ax.set_xlabel('Time of the data')
ax.set_ylabel('Speed in mph')
ax.set_title('Speed vs. Time of the Day for (Normal Day) \n After Filtering ')
plt.grid()
ax.legend()
plt.tight_layout()

plt.show()
fig.savefig('without_outlier.png')

model_data = data.filter(['zone_id','latitude','longitude','measurement_start','speed_x','speed_y','speed'])

# at the begining we have 6 locations
# since we arrange them as upstream, down stream, and target link, we have 4 target zones or links  

target_zone = model_data['zone_id'].unique()
print(target_zone)

def test_train_split(data_arr,train_size):   
    #spliting input and target variables
    x_vals=data_arr[:,:5]
    y_vals=np.transpose([(data_arr[:,5])])
    x_train=x_vals[:train_size,:]
    x_test=x_vals[train_size:,:]
    y_train=y_vals[:train_size,:]
    y_test=y_vals[train_size:,:]
    return(x_train,x_test,y_train,y_test)

# for each zone, we prepare the model data and split it into test and train set 
data_arr, train_size, datetime = data_prep(10031, model_data,1)    

x_train, x_test, y_train, y_test = test_train_split(data_arr,train_size)   

# define and fit KNN model 
# make prediction on test set and calculate the error in terms of RMSE and MAE
knn = KNeighborsRegressor(n_neighbors=12, weights='uniform') 
knn.fit(x_train, y_train) 
y_pred1=knn.predict(x_test)
RMSE_1= round(sqrt(mean_squared_error(y_test, y_pred1)),2)
MAE_1=round(mean_absolute_error(y_test,y_pred1),2)


# define and fit SVR model 
# make prediction on test set and calculate the error in terms of RMSE and MAE
clf_pol = SVR(kernel='rbf',C=1.0, epsilon=0.2 )
clf_pol.fit(x_train,y_train) 
y_pred2=clf_pol.predict(x_test)
RMSE_2= round(sqrt(mean_squared_error(y_test, y_pred2)),2)
MAE_2=round(mean_absolute_error(y_test,y_pred2),2)

# define and fit ANN model 
# make prediction on test set and calculate the error in terms of RMSE and MAE
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100))
mlp.fit(x_train,y_train)    
y_pred3=mlp.predict(x_test)
RMSE_3= round(sqrt(mean_squared_error(y_test, y_pred3)),2)  
MAE_3=round(mean_absolute_error(y_test,y_pred3),2)

from keras.models import Sequential
from keras.layers import Dense

from keras.layers import LSTM
#create and fit the LSTM network
#in_tr
#def create_dataset(in_tr, look_back=1):
#	a = in_tr[i:(i+look_back), 0]
#	dataX.append(a)
#	dataY.append(in_tr[i + look_back, 0])
#	return numpy.array(dataX), numpy.array(dataY)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

#look_back = 1
#model = Sequential()
##model.add(LSTM(4, input_shape=(1, look_back)))
##model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
##model.add(LSTM(4))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
#model.fit_generator(x_train, y_train, epochs=100, verbose=2)

# make predictions
#trainPredict = model.predict(x_test)
#testPredict = model.predict(x_train)


#data_dim = 13
#timesteps = 13
#num_classes = 1
#batch_size = 32
#model = Sequential()
#model.add(LSTM(32, return_sequences = True, stateful = True,
               #batch_input_shape = (batch_size, timesteps, data_dim)))
#model.add(LSTM(input_shape=(x_train.shape[1],1)))
#model.add(LSTM(32, return_sequences = True, stateful = True))
#model.add(LSTM(32, stateful = True))
#model.add(Dense(1, activation = 'relu'))
#model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#model.summary()


verbose, epochs, batch_size = 0, 70, 16
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs))
model.compile(loss='mse', optimizer='adam')
# fit network
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)



y_pred4= model.predict(x_test)
RMSE_4= round(sqrt(mean_squared_error(y_test, y_pred4)),2)  
MAE_4=round(mean_absolute_error(y_test,y_pred4),2)


#Plot The Results for different zones with accuracy metrices  
fig,ax = plt.subplots(figsize=(15,7))  

monthyearFmt = mdates.DateFormatter('%b %d %H')
ax.xaxis.set_major_formatter(monthyearFmt)
ax.plot(datetime,y_test,label='Actual Data')
ax.plot(datetime,y_pred1,label='KNN '+'RMSE: '+ str(RMSE_1)+' MAE: '+str(MAE_1)) 
ax.plot(datetime,y_pred2,label='SVR '+'RMSE: '+ str(RMSE_2)+' MAE: '+str(MAE_2))     
ax.plot(datetime,y_pred3,label='ANN '+'RMSE: '+ str(RMSE_3)+' MAE: '+str(MAE_3))   
ax.plot(datetime,y_pred4,label='LSTM '+'RMSE: '+ str(RMSE_4)+' MAE: '+str(MAE_4))   

  

ax.set_xlabel('Time of the data')
ax.set_ylabel('Speed in mph')
ax.set_title('Speed Vs. Time of the Day for (Normal Day) \n After Filtering for'+'zone_id:'+str(10031))
plt.grid()
ax.legend()
plt.tight_layout()

plt.show()
fig.savefig('comaprison.png')







