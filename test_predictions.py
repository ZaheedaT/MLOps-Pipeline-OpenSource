import requests

BENTO_API_URL = "http://localhost:3000/predict"

input_datasets = [
    [100000000000, 20.0, 10.5],       # Sample 1
    [540.00, 15.0, 50.0],     # Sample 2
    [21.00, 40.0, 250.25],   # Sample 3
    [0000000000000.0, 0.0, 0.0]            # Sample 4
]

for input_data in input_datasets:

    response = requests.post(
        BENTO_API_URL,
        json={"input_data": [input_data]}
    )

    if response.status_code == 200:
        prediction = response.json()

        print(f"Input: {input_data}")
        print(f"Prediction: {prediction}")

    else:
        print(f"Error {response.status_code}: {response.text}")