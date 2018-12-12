import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv("trip.csv", error_bad_lines=False)
df['date'] = df['starttime'].apply(lambda x: datetime.strptime(x.split(' ')[0], "%m/%d/%Y"))
# df.to_csv("trips_with_date.csv")
# df = pd.read_csv("trips_with_date.csv")

'''
# create a plot showing the trend of # of rides for each day
rides_per_day = df.groupby('date')['to_station_id'].count()
#dates=[datetime.fromtimestamp(ts) for ts in rides_per_day.index]
plt.plot(rides_per_day.index, rides_per_day.values)
plt.title("Count of Rides per day")
# plt.show()
plt.savefig("images/ride_counts_per_day.png")
'''

'''
# create a plot showing the trend of # of rides for each day by Member vs Non-Member

members = df[df['usertype'] == "Member"].groupby('date')['to_station_id'].count()
non_member = df[df['usertype'] != "Member"].groupby('date')['to_station_id'].count()

plt.plot(members.index, members.values, c="g", label="Member", alpha=.6)
plt.plot(non_member.index, non_member.values, c="b", label="Non-Member", alpha=.6)
plt.title("Members vs Non-Members rides per day")
plt.legend()
# plt.show()
plt.savefig("images/membersVSNon_ride_counts_per_day.png")
'''

# create plot showing arrivals grouped by to_station_id over time
# rides_per_day = df.groupby(['to_station_id','date'])#['to_station_id'].count()
# for name, group in rides_per_day:
#     plt.plot(group.iloc[0]['date'].to_pydatetime(), len(group), label=name)
#
# plt.legend()
# plt.title("Rides per day grouped by station_id")
# plt.savefig("rides_by_station")

# histogram of the hour of the day
df['hour'] = df['starttime'].apply(lambda x: int(x.split(" ")[1].split(":")[0]))
members = df[df['usertype'] == "Member"].groupby('hour')['to_station_id'].count()
non_member = df[df['usertype'] != "Member"].groupby('hour')['to_station_id'].count()
plt.bar(members.index, members.values, label='Members', color=(0.2, 0.4, 0.6, 0.6))
plt.bar(non_member.index, non_member.values, label="Non-Members", color=(0.2, 0.7, 0.3, 0.6))
plt.legend()
plt.title("Members vs Non-Members Number of Rides per hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of rides")
plt.savefig("images/membersVSNon_hourly_rides.png")

pass