import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# popular destinations
# self.rides.groupby('to_station_id')['from_station_id'].count().sort_values()

# data from: https://www.kaggle.com/pronto/cycle-share-dataset/home
class Model:

    def __init__(self):
        self.rides = pd.read_csv("trip.csv", error_bad_lines=False)
        self.weather = pd.read_csv("weather.csv", na_values=['-'])
        self.clean_ride_data()
        self.X = None
        self.y = None
        self.set_X_Y()

    def clean_ride_data(self):
        self.rides['Date'] = self.rides['starttime'].apply(lambda x: datetime.strptime(x.split(' ')[0], "%m/%d/%Y"))
        self.rides = self.rides[self.rides['from_station_id'] != 'Pronto shop']
        self.rides = self.rides[self.rides['to_station_id'] != 'Pronto shop']
        self.rides = self.rides[self.rides['from_station_id'] != 'Pronto shop 2']
        self.rides = self.rides[self.rides['to_station_id'] != 'Pronto shop 2']
        self.rides = self.rides[self.rides['from_station_id'] != '8D OPS 02']
        self.rides = self.rides[self.rides['to_station_id'] != '8D OPS 02']
        self.weather['Date'] = self.weather['Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
        self.weather.drop(
            ['Max_Temperature_F', 'Max_Dew_Point_F', 'MeanDew_Point_F', 'Min_Dewpoint_F', 'Max_Humidity',
             'Min_Humidity', 'Max_Sea_Level_Pressure_In',
             'Mean_Sea_Level_Pressure_In', 'Min_Sea_Level_Pressure_In',
             'Max_Visibility_Miles', 'Mean_Visibility_Miles', 'Min_Visibility_Miles',
             'Max_Wind_Speed_MPH', 'Max_Gust_Speed_MPH', 'Events'], axis=1, inplace=True)
        self.weather.interpolate(inplace=True)

    def set_X_Y(self):
        self.y = self.rides.groupby('Date')['tripduration'].sum()
        self.weather.set_index('Date', inplace=True)
        self.X = self.weather
        pass

    # Average the previous and next week and subtract from current week
    def remove_seasonality(self):


    def modeling(self):
        X_train_big, X_test_big, y_train_big, y_test_big = train_test_split(self.X, self.y, test_size=.3, random_state=42)

        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        for degree in range(X_train_big.shape[1] - 1):
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            errors = []
            scores = []

            for train_index, test_index in kf.split(X_train_big):
                X_train = X_train_big.iloc[train_index]
                X_test = X_train_big.iloc[test_index]
                y_train = y_train_big.iloc[train_index]
                y_test = y_train_big.iloc[test_index]

                model.fit(X_train, y_train)
                predicted = model.predict(X_test)
                errors.append(mean_squared_error(y_test, predicted))
                scores.append(model.score(X_test, y_test))
            print("Degree", degree, ": MSE_avg", np.mean(errors), " Score_avg:", np.mean(scores))



model = Model()
model.modeling()
