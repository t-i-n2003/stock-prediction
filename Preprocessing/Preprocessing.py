from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer

def MinMaxScaler(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    return scaled_prices

def StandardScaler(prices):  
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(prices)
    return scaled_data

def RobustScaler(prices):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(prices)
    return scaled_data

def MaxAbsScaler(prices):
    scaler = MaxAbsScaler()
    scaled_data = scaler.fit_transform(prices)
    return scaled_data

def Normalizer(prices):
    scaler = Normalizer(norm='l2')
    scaled_data = scaler.fit_transform(prices)
    return scaled_data
