import os
import pandas as pd
import numpy as np
from src.dataLoad import loadData, getRul_categories, getSensors
from src.task1Segmentation import *


def main():
    print("Water Pump RUL Analysis - Task 1:")

    #Create results folder for saving visualizations
    os.makedirs("results/task1_plots", exist_ok=True)

    #Load the dataset with the first 10,000 rows as specified
    print("\n1.Loading dataset..")
    df = loadData("data/water_pump_rul.csv", num_rows=10000)
    df, quantiles = getRul_categories(df)
    print(f"RUL Categories distribution:")
    print(df['RUL_Category'].value_counts().sort_index())

    #Get all of the available sensor names
    all_sensors = getSensors(df)
    print(f"\nTotal sensors available: {len(all_sensors)}")

    #Select 10 random sensors (Step 1 of Task 1)
    print("\n2.Selecting 10 random sensors..")
    selected_sensors = sensor_select(all_sensors, seed=42)
    print(f"Selected sensors:")
    for s in selected_sensors:
        print(f"  - {s}")

    #Running segmentation on all selected sensors (Steps 2 & 4 of Task 1)
    print("\n3.Running segmentation algorithm on all 10 sensors..")
    results = analyze_sensors(df, selected_sensors)

    #Visualize the results for all 10 of the sensors (Step 3 of Task 1)
    print("\n4.Generating visualizations for all 10 sensors..")
    for sensor in selected_sensors:
        data = results[sensor]

        #Create the plot
        plt.figure(figsize=(12, 4))
        plt.plot(data["signal"], linewidth=0.8)

        #Add red dashed lines at segment boundaries
        for start, end in data["segments"]:
            plt.axvline(start, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

        plt.title(f"{sensor} - {data['complexity']} segments")
        plt.xlabel("Time Index")
        plt.ylabel("Sensor Value")
        plt.tight_layout()

        #Save before showing the graphs
        save_path = f"results/task1_plots/{sensor}_segmentation.png"
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

        #Then display the graphs
        plt.show()
        plt.close()

    #Analyze relationship between segments and RUL categories (Step 5 of Task 1)
    print("\n5. Analyzing relationship between segments and RUL categories..")
    analyze_rul_relation(results, df['RUL_Category'].values)

    #Discuss temporal dynamics based on the complexity scores (Step 6 of Task 1)
    print("\n6.Analyzing temporal dynamics across sensors..")
    discuss_temporal_dynamics(results)


    print("\nTask 1 Finished")
    print(f"All visualizations have been saved in results folder")

if __name__ == "__main__":
    #First run toy example to verify algorithm works
    print("Task 1 Toy Verification Example:")
    toy_example_verification()

    #Then run main analysis on real data
    print("\nAnalyzing Data Set:")
    main()
