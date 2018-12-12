import pandas as pd
import numpy as np
from datetime import datetime


class Model:

    def __init__(self):
        self.rides = pd.read_csv("trip.csv", error_bad_lines=False)
        self.weather = pd.read_csv("weather.csv")
        self.clean_ride_data()
        self.data = None
        self.merge_rides_weather()

    def clean_ride_data(self):
        self.rides['Date'] = self.rides['starttime'].apply(lambda x: datetime.strptime(x.split(' ')[0], "%m/%d/%Y"))
        self.rides = self.rides[self.rides['from_station_id'] != 'Pronto shop']
        self.rides = self.rides[self.rides['to_station_id'] != 'Pronto shop']
        self.rides = self.rides[self.rides['from_station_id'] != 'Pronto shop 2']
        self.rides = self.rides[self.rides['to_station_id'] != 'Pronto shop 2']
        self.rides = self.rides[self.rides['from_station_id'] != '8D OPS 02']
        self.rides = self.rides[self.rides['to_station_id'] != '8D OPS 02']
        self.weather['Date'] = self.weather['Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))

    def merge_rides_weather(self):

        pass


model = Model()
