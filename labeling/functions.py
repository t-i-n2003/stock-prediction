def calculate_average_for_next_period(df, start_date, days_required=7, column=''):
    df_filtered = df[df['time'] > start_date] 
    num_days = min(len(df_filtered), days_required)
    average = df_filtered.head(num_days)[column].mean()
    return average

def calculate_average_for_previous_period(df, end_date, days_required=7, column=''):
    df_filtered = df[df['time'] < end_date]
    df_filtered = df_filtered.sort_values(by='time', ascending=False)
    num_days = min(len(df_filtered), days_required)
    average = df_filtered.head(num_days)[column].mean() 
    return average

def add_previous_and_next_7(df, days_required=7, column=''):
    previous_7 = []
    next_7 = []
    days = 0
    for index, row in df.iterrows():
        prev_avg = calculate_average_for_previous_period(df, row['time'], days_required, column)
        next_avg = calculate_average_for_next_period(df, row['time'], days_required, column)  
        previous_7.append(prev_avg)
        next_7.append(next_avg)
    s1 = column + '_before'
    s2 = column + '_after'
    df[s1] = previous_7
    df[s2] = next_7
    return df
