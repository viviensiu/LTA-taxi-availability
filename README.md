# LTA-taxi-availability
Background: The Land Transport Authority (LTA) in Singapore possesses a dataset of historical and real-time taxi availability thru a central system. The API call returns the location coordinates of all taxis that are currently available for hire but does not include "Hired" or "Busy" Taxis.

Source: LTA Taxi Availability (https://data.gov.sg/dataset/taxi-availability)

# Objective: 
We will train a ML model to predict taxi availability at a given time and location.

# Methodology:
1) Explore the taxi dataset and API call method at the LTA taxi availability website.
2) Perform API call to download the dataset required for our model.
3) From the downloaded data, we extract only the required features in JSON format, and convert these JSON objects into usable Dataframe.  
4) feature engineering: 
   - the location coordinates are used to divide the taxi locations throughout Singapore into 9 different sectors.
   - sum up the number of taxis based on their sectors.
   - extract the day of week, minute and hour from the timestamp.
   - drop redundant features from the dataset.
5) Perform EDA on cleaned data. Data is converted into time series for resampling before visualization.
6) Perform further cleaning after EDA as some sectors have no taxi available and therefore can be dropped.
7) Now we are ready to create a ML model to predict taxi availability at designated sectors. We divide the data into 1 set of feature data and 5 sets of target data for 5 remaining sectors. 
8) Then we split them into 5 sets of training and test data.
9) For each set, we train 4 regression models with the training set: DummyRegressor, LinearRegression, DecisionTreeRegressor, RandomForestRegressor. These models are evaluated using the root mean squared error metrics with data from test set.
10) We plot the actual target values vs. predictions using scatter plot to see if the model was able to predict accurately. 

Technical Skills: Pandas, requests, JSON, numpy, Matplotlib, Seaborn, Time Series Data visualization. 

# Conclusion:
Out of the 4 ML models, the RandomForestRegressor model produces the best scores followed by the DecisionTreeRegressor model. The Normalised RMSE scores from the RandomForestRegressor model deviates in the range of 9% to 15% from the actual values.

# Ideas for expansion:
- the taxi availability data can be used for other problem statements, such as to analyse taxi demands across different locations and time periods, which could help taxi companies to anticipate and optimize taxi distributions at peak periods across the country.
- Similarly, the ML model could be adopted for ride-sharing companies as well. 
- The dataset could benefit from additional data such as weather and public transportation availability, as bad weather plus train service disruptions contribute to an increase in taxi demand. Combined with weather forecast, this could help taxi and ride-share companies in demand predictions.

# Useful Reading:
1) Python API Tutorial: Getting Started with APIs (https://www.dataquest.io/blog/python-api-tutorial/)
2) What is Considered a Good RMSE Value? (https://www.statology.org/what-is-a-good-rmse/)
3) 5 ways ML helps in Uber Services Optimization (https://www.analyticssteps.com/blogs/5-ways-ml-helps-uber-services-optimization)
