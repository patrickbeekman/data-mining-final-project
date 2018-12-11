import pickle
import pandas as pd
import os

filenames = os.listdir("./followers/")
dict_followers = {}
count = 0

for filename in filenames:
    try:
        df = pd.read_json("./followers/" + filename)
    except ValueError:
        print(filename + ": could not be loaded")
        continue
    screen_name = filename[1:-15]
    print("processing: " + screen_name + " -> " + str(count/len(filenames)))

    temp_list = []
    for follower in df.iterrows():
        temp_list.append(follower[1]['screen_name'])
    dict_followers[screen_name] = temp_list
    count+=1
    del df

#print(dict_followers)
with open("./followers_dict.pkl", 'wb') as f:
    pickle.dump(dict_followers, f, pickle.HIGHEST_PROTOCOL)

print("hi")