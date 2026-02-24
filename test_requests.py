import requests # http request sending
import random # sample data

# fast api endpoint url
API_URL = "http://127.0.0.1:8000/predict" # if we need we can change the port

# random sample input data generator
sample_sequence = []
for i in range(24):
    temp = random.uniform(280, 300) # kelvin 
    rain = 0
    snow = 0
    clouds = random.randint(0, 100) # percentage
    hour = i  # random hour
    dayofweek = random.randint(0, 6) # monday to sunday
    month = 10  # lets assign october
    sample_sequence.append([temp, rain, snow, clouds, hour, dayofweek, month])

print(sample_sequence)    

# send post request (json format)
response = requests.post(API_URL, json= {"sequence": sample_sequence})

if response.status_code == 200:
    result = response.json()
    print(f"Prediction successful.\n Predicted traffic volume: {result["predicted_traffic_volume"]}")
else:
    print("Error")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}" )


