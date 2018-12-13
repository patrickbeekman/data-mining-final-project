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
        self.weather = pd.read_csv("weather_no_seasonality.csv", na_values=['-'])
        self.clean_ride_data()
        self.X = None
        self.y = None
        self.set_X_Y()
        self.remove_seasonality()

    def clean_ride_data(self):
        self.rides['Date'] = self.rides['starttime'].apply(lambda x: datetime.strptime(x.split(' ')[0], "%m/%d/%Y"))
        self.rides = self.rides[self.rides['from_station_id'] != 'Pronto shop']
        self.rides = self.rides[self.rides['to_station_id'] != 'Pronto shop']
        self.rides = self.rides[self.rides['from_station_id'] != 'Pronto shop 2']
        self.rides = self.rides[self.rides['to_station_id'] != 'Pronto shop 2']
        self.rides = self.rides[self.rides['from_station_id'] != '8D OPS 02']
        self.rides = self.rides[self.rides['to_station_id'] != '8D OPS 02']
        self.weather['Date'] = self.weather['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        self.weather.drop(['Unnamed: 0', 'Mean_Temperature_F', 'Min_TemperatureF'], axis=1, inplace=True)

        # self.weather['Date'] = self.weather['Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
        # self.weather.drop(
        #     ['Max_Temperature_F', 'Max_Dew_Point_F', 'MeanDew_Point_F', 'Min_Dewpoint_F', 'Max_Humidity',
        #      'Min_Humidity', 'Max_Sea_Level_Pressure_In',
        #      'Mean_Sea_Level_Pressure_In', 'Min_Sea_Level_Pressure_In',
        #      'Max_Visibility_Miles', 'Mean_Visibility_Miles', 'Min_Visibility_Miles',
        #      'Max_Wind_Speed_MPH', 'Max_Gust_Speed_MPH', 'Events'], axis=1, inplace=True)
        # self.weather.interpolate(inplace=True)

    def set_X_Y(self):
        self.y = self.rides.groupby('Date')['tripduration'].sum()
        self.weather.set_index('Date', inplace=True)
        self.X = self.weather
        self.X['is_weekend'] = [1 if x.weekday() >= 5 else 0 for x in self.X.index]
        pass

    # Average the previous and next week and subtract from current week
    def remove_seasonality(self):
        # weekly_averages = self.X['Mean_Temperature_F'].resample('W').mean()
        # new_temp = []
        pass


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


'''
Polynomial models, KFold CV, Seasonality included, best is Linear Model (degree 1)
Degree 0 : MSE_avg 96892584964.2  Score_avg: -0.00846191037771
Degree 1 : MSE_avg 51319860032.8  Score_avg: 0.463936116689
Degree 2 : MSE_avg 116035394689.0  Score_avg: -0.198514127468
Degree 3 : MSE_avg 163533390026.0  Score_avg: -0.754070678516
Degree 4 : MSE_avg 8.84053057243e+12  Score_avg: -97.667238757
'''

'''
After Seasonality reduction for the Mean Temperature, comparable but more consistent
Degree 0 : MSE_avg 96892584964.2  Score_avg: -0.00846191037771
Degree 1 : MSE_avg 53700436184.8  Score_avg: 0.439335838008
Degree 2 : MSE_avg 51603721380.0  Score_avg: 0.456934035431
Degree 3 : MSE_avg 71670877991.4  Score_avg: 0.212606155121
'''

model = Model()
model.modeling()
