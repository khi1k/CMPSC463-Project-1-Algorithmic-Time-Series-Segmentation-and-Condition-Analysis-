import pandas as pd
import numpy as np

#loading dataset, will get the first 10000 rows of data
def loadData(filepath, num_rows=10000):
    df = pd.read_csv(filepath, nrows=num_rows)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

#this function will convert the RUL values into the 4 different categories, based on their quantiles
def getRul_categories(df):
    rul_vals = df['rul']

    #calulating the quantiles
    Q10 = np.percentile(rul_values, 10) #lower quantile
    Q50 = np.percentile(rul_values, 50) #median quantile
    Q90 = np.percentile(rul_values, 90) #upper quantile
    print(f"RUL Quantiles - Q10: {Q10:.2f}, Q50: {Q50:.2f}, Q90: {Q90:.2f}")

    # 0=very low, 1=low, 2 = high, 3= very high
    categories = []
    for r in rul_vals:
        if r< Q10:
            categories.append(0)
        elif r< Q50:
            categories.append(1)
        elif r< Q90:
            categories.append(2)
        else:
            categories.append(3)
    df['RUL_Category'] = categories #add a new category as a column in the data frame
    return df, (Q10, Q50, Q90)

#returns a list of all the sensor columns and their names
def getSensors(df):
    return [col for col in df.columns if 'sensor_' in col]