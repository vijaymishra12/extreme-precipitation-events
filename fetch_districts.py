import requests
import json
url = "https://raw.githubusercontent.com/sab99r/Indian-States-And-Districts/master/states-and-districts.json"
try:
    data = requests.get(url).json()
    states_of_interest = [
        "Jammu and Kashmir", "Himachal Pradesh", "Punjab", "Uttarakhand", 
        "Haryana", "Delhi", "Rajasthan", "Uttar Pradesh", "Bihar", 
        "Madhya Pradesh", "West Bengal"
    ]
    res = {}
    for item in data['states']:
        st = item['state']
        if st in states_of_interest:
            if st == "Jammu and Kashmir": st = "Jammu & Kashmir"
            res[st] = item['districts']
    
    with open("district_names.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Done")
except Exception as e:
    print(e)
