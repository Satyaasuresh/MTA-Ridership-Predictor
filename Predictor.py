#!/usr/bin/env python
# coding: utf-8

# In[5]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv('MTA_Subway_Hourly_Ridership__Beginning_February_2022.csv')


# In[7]:


df.head()


# In[8]:


df['transit_timestamp'] = pd.to_datetime(df['transit_timestamp'])


# In[9]:


original_df = df.copy()


# In[17]:


start_date = pd.Timestamp('2022-11-01')
end_date = pd.Timestamp('2023-11-01')
df = df[(df.transit_timestamp >= start_date) & (df.transit_timestamp < end_date)]


# In[18]:


print(df['transit_timestamp'].min())
print(df['transit_timestamp'].max())


# In[19]:


df['station_complex'] = df['station_complex'].str.replace(r'\(.*\)', '', regex=True)


# In[20]:


df['day_of_week'] = df['transit_timestamp'].dt.day_name()


# In[21]:


df['day_number'] = df['day_of_week'].astype(int)


# In[25]:


day = pd.DataFrame({'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})


# In[26]:


day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}


# In[27]:


day['day_number'] = day['day_week'].map(day_mapping)


# In[28]:


day['day_number'] = day['day_of_week'].map(day_mapping)


# In[29]:


df['day_number'] = df['day_of_week'].astype(int)


# In[30]:


sns.heatmap(df.corr(), annot = True)


# In[34]:


df['hour'] = df['transit_timestamp'].dt.hour


# In[32]:


df['hour_number'] = df['hour'].astype(int)


# In[35]:


sns.heatmap(df.corr(), annot = True)


# In[36]:


df <- subset(df, select = -hour_number)


# In[37]:


del df['hour_number']


# In[38]:


sns.heatmap(df.corr(), annot = True)


# In[41]:


import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd # for geographic data manipulation
from shapely.geometry import Point, Polygon

from sklearn.model_selection import train_test_split # for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[40]:


pip install geopandas


# In[42]:


df['station_complex'] = df['station_complex'].str.replace(r'\(.*\)', '', regex=True)


# In[43]:


gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))

geo_stations_df = gdf.groupby(['station_complex_id', 'station_complex', 'borough', 'latitude','longitude', 'geometry'])['ridership'].sum().reset_index()


# In[44]:


by_station_df = df.groupby(['station_complex_id', 'station_complex', 'borough', 'latitude', 'longitude'])['ridership'].sum().reset_index()


# In[45]:


boroughs = df.groupby('borough')['ridership'].sum()


# In[46]:


by_station_df_normalize = by_station_df.copy()
by_station_df_normalize['ridership'] = by_station_df_normalize['ridership']/1000000

sns.swarmplot(x=by_station_df_normalize['ridership'], y=by_station_df_normalize['borough'], hue=by_station_df_normalize['borough'])
plt.yticks([0, 1, 2, 3], ['Manhattan', 'Brooklyn', 'Queens', 'Bronx']) 
plt.xlabel('Total Ridership in Millions')
plt.ylabel('Borough')
plt.annotate('Times Sq-42 St', xy=(0.76, 0.915), xycoords='axes fraction')
plt.title('Ridership by Borough')
plt.show()


# In[47]:


station_count = by_station_df['borough'].value_counts()
borough_names = ['Brooklyn', 'Manhattan', 'Queens', 'Bronx']


# In[49]:


by_station_df_normalize = by_station_df.copy()
by_station_df_normalize['ridership'] = by_station_df_normalize['ridership']/1000000

sns.stripplot(x=by_station_df_normalize['ridership'], y=by_station_df_normalize['borough'], hue=by_station_df_normalize['borough'])
plt.yticks([0, 1, 2, 3], ['Manhattan', 'Brooklyn', 'Queens', 'Bronx']) 
plt.xlabel('Total Ridership in Millions')
plt.ylabel('Borough')
plt.annotate('Times Sq-42 St', xy=(0.76, 0.915), xycoords='axes fraction')
plt.title('Ridership by Borough')
plt.show()


# In[50]:


by_station_df = df.groupby(['station_complex_id', 'station_complex', 'borough', 'latitude', 'longitude'])['ridership'].sum().reset_index()


# In[51]:


by_station_df.shape


# In[52]:


by_station_df.head()


# In[53]:


by_station_df.drop(df[df.borough.isin(["M","BK","Q","BX","SI"])].index)


# In[54]:


by_station_df.shape


# In[55]:


df = df[~df['borough'].isin(['M', 'BK', 'Q','SI','BX'])]


# In[56]:


by_station_df = by_station_df[~by_station_df['borough'].isin(['M', 'BK', 'Q','SI','BX'])]


# In[57]:


by_station_df.shape


# In[58]:


by_station_df_normalize = by_station_df.copy()
by_station_df_normalize['ridership'] = by_station_df_normalize['ridership']/1000000

sns.swarmplot(x=by_station_df_normalize['ridership'], y=by_station_df_normalize['borough'], hue=by_station_df_normalize['borough'])
plt.yticks([0, 1, 2, 3], ['Manhattan', 'Brooklyn', 'Queens', 'Bronx']) 
plt.xlabel('Total Ridership in Millions')
plt.ylabel('Borough')
plt.annotate('Times Sq-42 St', xy=(0.76, 0.915), xycoords='axes fraction')
plt.title('Ridership by Borough')
plt.show()


# In[59]:


by_station_df = by_station_df[~by_station_df['borough'].isin(['Staten Island'])]


# In[119]:


by_station_df_normalize = by_station_df.copy()
by_station_df_normalize['ridership'] = by_station_df_normalize['ridership']/1000000

sns.stripplot(x=by_station_df_normalize['borough'], y=by_station_df_normalize['ridership'], hue=by_station_df_normalize['borough'], palette = 'rocket', jitter = False)
plt.xticks([0, 1, 2, 3], ['Manhattan', 'Brooklyn', 'Queens', 'Bronx']) 
plt.xlabel('Borough')
plt.ylabel('Ridership (Millions)')
plt.annotate('Times Sq-42 St', xy=(0.15, 0.944), xycoords='axes fraction')
plt.title('Ridership by Borough')
plt.grid(False)
plt.show()


# In[121]:


station_count = by_station_df['borough'].value_counts()
borough_names = ['Brooklyn', 'Manhattan', 'Queens', 'Bronx']

fig = px.bar(x = borough_names, y = station_count.values, 
       color = station_count.index, text = station_count.values, 
       title = 'Number of Stations by Borough', palette = "rocket")

fig.update_layout( xaxis_title = "Boroughs", yaxis_title = "Subway Stations")
fig.show()


# In[122]:


by_station_df_normalize = by_station_df.copy()
by_station_df_normalize['ridership'] = by_station_df_normalize['ridership']/1000000

sns.stripplot(x=by_station_df_normalize['ridership'], y=by_station_df_normalize['borough'], hue=by_station_df_normalize['borough'])
plt.yticks([0, 1, 2, 3], ['Manhattan', 'Brooklyn', 'Queens', 'Bronx']) 
plt.xlabel('Total Ridership in Millions')
plt.ylabel('Borough')
plt.annotate('Times Sq-42 St', xy=(0.76, 0.915), xycoords='axes fraction')
plt.title('Ridership by Borough')
plt.show()


# In[3]:


rfr_df = pd.DataFrame()


# In[4]:


rfr_df['month'] = original_df['transit_timestamp'].dt.month
rfr_df['day'] = original_df['transit_timestamp'].dt.day
rfr_df['hour'] = original_df['transit_timestamp'].dt.hour
rfr_df['ridership'] = original_df['ridership']

rfr_df = rfr_df.groupby(['month', 'day', 'hour'])['ridership'].sum().reset_index()

rfr_df.head()


# In[10]:


rfr_df['month'] = original_df['transit_timestamp'].dt.month
rfr_df['day'] = original_df['transit_timestamp'].dt.day
rfr_df['hour'] = original_df['transit_timestamp'].dt.hour
rfr_df['ridership'] = original_df['ridership']

rfr_df = rfr_df.groupby(['month', 'day', 'hour'])['ridership'].sum().reset_index()

rfr_df.head()


# In[11]:


X = rfr_df[['month', 'day', 'hour']]
y = rfr_df['ridership']

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


from sklearn.model_selection import train_test_split # for machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[13]:


X = rfr_df[['month', 'day', 'hour']]
y = rfr_df['ridership']

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Split the data into features and labels
labels = np.array(rfr_df['ridership'])
features = rfr_df.drop('ridership', axis=1)
feature_list = list(features.columns)
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=38
)

random_forest = RandomForestRegressor(n_estimators=400, random_state=38)

# Train the model using the training data
random_forest.fit(train_features, train_labels)

# Generate predictions on the test set
predictions = random_forest.predict(test_features)

# Compute the absolute errors
errors = abs(predictions - test_labels)

# Display the Mean Absolute Error (MAE)
mae = mean_absolute_error(test_labels, predictions)
print('Mean Absolute Error:', round(mae, 2), 'riders.')

# Calculate the R-squared (R2)
r2 = r2_score(test_labels, predictions)
print('R-squared (R2):', round(r2, 2))

# Create a scatter plot to visualize the predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(test_labels, predictions, alpha=0.7, color = "darkblue")
plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r--')

plt.xlabel('Actual Ridership')
plt.ylabel('Predicted Ridership')
plt.title('Random Forest Regressor: Actual vs. Predicted Ridership')
plt.show()


# In[19]:


def predict_and_show_ridership(month, day, hour):
    input_data = np.array([[month, day, hour]])
    predicted_ridership = rf.predict(input_data)
    actual_ridership = rfr_df[(rfr_df['month'] == month) & (rfr_df['day'] == day) & (rfr_df['hour'] == hour)]['ridership'].values[0]
    
    print(f"Input: Month={month}, Day={day}, Hour={hour}")
    print(f"Predicted Ridership: {round(predicted_ridership[0], 2)}")
    print(f"Actual Ridership: {actual_ridership}")


# In[20]:


predict_and_show_ridership(month=7, day=3, hour=18)


# In[ ]:




