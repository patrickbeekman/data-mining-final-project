import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

rides = pd.read_csv("trip.csv", error_bad_lines=False)
rides['Date'] = rides['starttime'].apply(lambda x: datetime.strptime(x.split(' ')[0], "%m/%d/%Y"))
rides = rides[rides['from_station_id'] != 'Pronto shop']
rides = rides[rides['to_station_id'] != 'Pronto shop']
rides = rides[rides['from_station_id'] != 'Pronto shop 2']
rides = rides[rides['to_station_id'] != 'Pronto shop 2']
rides = rides[rides['from_station_id'] != '8D OPS 02']
rides = rides[rides['to_station_id'] != '8D OPS 02']

weather = pd.read_csv("weather.csv", na_values=['-'])
weather['Date'] = weather['Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
weather.drop(
    ['Max_Temperature_F', 'Max_Dew_Point_F', 'MeanDew_Point_F', 'Min_Dewpoint_F', 'Max_Humidity',
     'Min_Humidity', 'Max_Sea_Level_Pressure_In',
     'Mean_Sea_Level_Pressure_In', 'Min_Sea_Level_Pressure_In',
     'Max_Visibility_Miles', 'Mean_Visibility_Miles', 'Min_Visibility_Miles',
     'Max_Wind_Speed_MPH', 'Max_Gust_Speed_MPH', 'Events'], axis=1, inplace=True)
weather.interpolate(inplace=True)

# rides.to_csv("trips_with_date.csv")
# rides = pd.read_csv("trips_with_date.csv")

'''
# create a plot showing the trend of # of rides for each day
rides_per_day = rides.groupby('date')['to_station_id'].count()
#dates=[datetime.fromtimestamp(ts) for ts in rides_per_day.index]
plt.plot(rides_per_day.index, rides_per_day.values)
plt.title("Count of Rides per day")
# plt.show()
plt.savefig("images/ride_counts_per_day.png")
'''

'''
# create a plot showing the trend of # of rides for each day by Member vs Non-Member

members = rides[rides['usertype'] == "Member"].groupby('date')['to_station_id'].count()
non_member = rides[rides['usertype'] != "Member"].groupby('date')['to_station_id'].count()

plt.plot(members.index, members.values, c="g", label="Member", alpha=.6)
plt.plot(non_member.index, non_member.values, c="b", label="Non-Member", alpha=.6)
plt.title("Members vs Non-Members rides per day")
plt.legend()
# plt.show()
plt.savefig("images/membersVSNon_ride_counts_per_day.png")
'''

# create plot showing arrivals grouped by to_station_id over time
# rides_per_day = rides.groupby(['to_station_id','date'])#['to_station_id'].count()
# for name, group in rides_per_day:
#     plt.plot(group.iloc[0]['date'].to_pydatetime(), len(group), label=name)
#
# plt.legend()
# plt.title("Rides per day grouped by station_id")
# plt.savefig("rides_by_station")

# histogram of the hour of the day
# rides['hour'] = rides['starttime'].apply(lambda x: int(x.split(" ")[1].split(":")[0]))
# members = rides[rides['usertype'] == "Member"].groupby('hour')['to_station_id'].count()
# non_member = rides[rides['usertype'] != "Member"].groupby('hour')['to_station_id'].count()
# plt.bar(members.index, members.values, label='Members', color=(0.2, 0.4, 0.6, 0.6))
# plt.bar(non_member.index, non_member.values, label="Non-Members", color=(0.2, 0.7, 0.3, 0.6))
# plt.legend()
# plt.title("Members vs Non-Members Number of Rides per hour")
# plt.xlabel("Hour of Day")
# plt.ylabel("Number of rides")
# plt.savefig("images/membersVSNon_hourly_rides.png")

'''
# from CBD-13 to all other locations, variance
rides[rides['to_station_id'] == "CBD-13"].boxplot(column='tripduration', by='from_station_id', rot=45)
# plt.show()
plt.title("Boxplot From all stations to station_id CBD-13")
plt.suptitle("")
plt.ylabel("Trip duration (seconds)")
plt.xlabel("From station_id")
plt.show()
# plt.savefig("images/boxplot_from_all_to_CBD-13.png", quality=95)
'''

# what is the variance of the trip duration for each ride from a station to CBD-13?
# rides[rides['to_station_id'] == "CBD-13"].groupby("from_station_id")['tripduration'].var().sort_values()

'''
# Trip duration over time
rides.groupby("date")['tripduration'].sum().plot()
plt.title("Daily total Trip duration over time")
plt.ylabel("Trip duration (seconds)")
plt.show()

rides[rides['usertype'] == "Member"].groupby('date')['tripduration'].sum().plot(label="Member", alpha=.8)
rides[rides['usertype'] != "Member"].groupby('date')['tripduration'].sum().plot(label="Non-Member", alpha=.8)
plt.title("Daily total Trip duration over time")
plt.ylabel("Trip duration (seconds)")
plt.legend()
plt.show()
'''

'''
# Trip duration by week day
members = rides[rides['usertype'] == "Member"]
non = rides[rides['usertype'] != "Member"]

members['weekday'] = [x.weekday() for x in members['Date']]
non['weekday'] = [x.weekday() for x in non['Date']]

mem_weekday_df = members.groupby("weekday")['tripduration'].sum()
#mem_weekday_df.rename(index={0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}, inplace=True)
# mem_weekday_df.plot(kind="bar", rot=45, label="Member", color='b')
plt.bar(mem_weekday_df.index, mem_weekday_df.values, .35, label="Members")

non_weekday_df = non.groupby("weekday")['tripduration'].sum()
#non_weekday_df.rename(index={0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}, inplace=True)
# non_weekday_df.plot(kind="bar", rot=45, label="Non-Member", color='g')
plt.bar(non_weekday_df.index+.35, non_weekday_df.values, .35, label="Non-Members")

plt.ylabel("Total Trip Duration (seconds)")
plt.xlabel("Day of the Week (0=Monday)")
plt.title("Total trip duration for each day of the week")
plt.legend()
plt.show()
'''

'''
# Trip duration by month
rides['day_of_month'] = [x.day for x in rides['date']]
day_df = rides.groupby("day_of_month")['tripduration'].sum()
day_df.plot(kind="bar", rot=45)
plt.ylabel("Total Trip Duration (seconds)")
plt.title("Total trip duration for each day of the month")
plt.show()
'''

'''
# Trip duration with Temperature
# fields: 'Mean_Temperature_F', 'Min_TemperatureF', 'Mean_Humidity',
#         'Mean_Wind_Speed_MPH', 'Precipitation_In'
duration = rides.groupby('date')['tripduration'].sum()
temp = weather.groupby('Date')['Mean_Wind_Speed_MPH'].first()

fig, ax1 = plt.subplots()
plt.title("Trip Duration with Wind")

ax1.plot(duration.index, duration.values, 'b', label="Trip Duration")
ax1.set_ylabel('Trip Duration total (seconds)')

ax2 = ax1.twinx()
ax2.plot(temp.index, temp.values, 'g', label="Wind", alpha=.8)
ax2.set_ylabel('Wind (mph)')

fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
'''

# correlation of weather attributes with Trip Duration
attributes = ['Mean_Temperature_F', 'Min_TemperatureF', 'Mean_Humidity', 'Mean_Wind_Speed_MPH', 'Precipitation_In']
duration = rides.groupby('Date')['tripduration'].sum()
for attr in attributes:
    weather_attr = weather.groupby('Date')[attr].first()
    print(attr, "\n", np.corrcoef(duration, weather_attr))

pass