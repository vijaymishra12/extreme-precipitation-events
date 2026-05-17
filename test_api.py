import requests
r = requests.post("http://127.0.0.1:5000/predict", 
                  data={"start_date": "2026-01-01", "end_date": "2027-12-31", "states": "Jammu & Kashmir"})
try:
    d = r.json()
    print("Success")
except Exception as e:
    print("ERROR:", r.status_code)
    print(r.text)
