
## Time Series forecasting using python


### Sort the timedate

### Analyse the gap in timestamp

````
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.sort_values(by='Timestamp', inplace=True)
time_diff = df['Timestamp'].diff()
expected_interval = pd.Timedelta(hours=1)
gaps = time_diff[time_diff > expected_interval]
if not gaps.empty:
    print("Gaps detected in the time series data.")
    print(gaps)
else:
    print("No gaps detected in the time series data.")
````

### Divide the data to test and train

### moving average
