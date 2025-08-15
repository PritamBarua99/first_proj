from sklearn.datasets import fetch_california_housing

def load_data():
    # Load California housing data as a pandas DataFrame
    data = fetch_california_housing(as_frame=True)
    X = data.data       # Feature DataFrame (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
    y = data.target     # Target Series (Median House Value in $100,000s)
    return X, y
