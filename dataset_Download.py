import kagglehub
import os
import pandas as pd

print("Downloading dataset..")

#Downloads to cache
cache_path = kagglehub.dataset_download("anseldsouza/water-pump-rul-predictive-maintenance")
print(f"Downloaded to cache: {cache_path}")

#Locates the CSV file inside of the cache
for file in os.listdir(cache_path):
    if file.endswith('.csv'):
        #Full path to the downloaded file
        source = os.path.join(cache_path, file)

        #Checks that the correct folder exists
        os.makedirs("data", exist_ok=True)

        #Reads and loads only the first 10,000 rows (10,001 including the header)
        print("Reading first 10,000 rows from dataset..")
        df = pd.read_csv(source, nrows=10000)  #This loads only 10k rows

        #Save to our data folder
        destination = os.path.join("data", "water_pump_rul.csv")
        df.to_csv(destination, index=False)

        print(f"Saved first {len(df)} rows to: {destination}")
        print(f"Columns: {len(df.columns)}")
        print(f"Row range: 0 to {len(df) - 1}")
        break

print("Dataset ready in data folder")