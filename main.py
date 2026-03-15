import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.dataLoad import loadData, getRul_categories, getSensors
from src.task1Segmentation import *
from src.task2Clustering import cluster_sensors, analyze_clusters, discuss_mapping, verify_clustering
from src.task3Kadane import analyze_all_sensors, find_early_indicators, plot_kadane_result, verify_kadane

#running all three tasks through the main function
def main():
    print("Water Pump RUL Analysis")

#Task 1
    print("\nTask 1: Divide and Conquer Segmentation ")

    #Create a results folder for saving the visualizations
    os.makedirs("results/task1_plots", exist_ok=True)

    #Loading of the dataset with the first 10,000 rows as specified
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
    print("\nTask 1 Finished, All visualizations have been saved in the results folder")

#Task 2
    print("\nTask 2: Recursive Clustering")

    #first set up a location to store task 2's results in the results folder
    os.makedirs("results/task2_results", exist_ok=True)

    #Cluster all 10000 time points into 4 clusters
    print("\n1. Running recursive clustering algorithm..")
    clusters = cluster_sensors(df, k=4)

    #Save the cluster info so we can review the results
    print("\n2. Saving cluster information..")
    with open("results/task2_results/cluster_indices.txt", "w") as f:
        for i, cluster in enumerate(clusters):
            f.write(f"Cluster {i}: {len(cluster)} points\n")
            #Show first 20 indices as a sample of the cluster composition
            f.write(f"Sample indices: {cluster[:20]}\n\n")
    print(f"   Saved: results/task2_results/cluster_indices.txt")

    #Analyze majority class per cluster (Step 2 of Task 2)
    print("\n3. Analyzing majority class per cluster..")
    cluster_results = analyze_clusters(clusters, df['RUL_Category'].values)

    #Discuss the cluster mapping (Step 3 of Task 2)
    print("\n4. Discussing cluster mapping..")
    discuss_mapping(cluster_results)

    #Save the summary to results folder
    with open("results/task2_results/cluster_summary.txt", "w") as f:
        for r in cluster_results:
            f.write(f"Cluster {r['cluster']}: {r['size']} points, ")
            f.write(f"majority {r['majority']} ({r['percentage']:.1f}%)\n")
            f.write(f"Distribution: {r['distribution']}\n\n")
    print(f"\n   Summary saved: results/task2_results/cluster_summary.txt")

    print("\nTask 2 Complete")

#Task 3
    print("\nTask 3: Kadane Maximum Subarray")

    #Create results folder for task 3
    os.makedirs("results/task3_results", exist_ok=True)

    #Run Kadane algorithm on all 52 sensors
    print("\n1. Running Kadane's algorithm on all 52 sensors..")
    kadane_results = analyze_all_sensors(df, df['RUL_Category'].values)

    #Finding the early warning indicators
    print("\n2. Identifying early warning indicators..")
    early_indicators = find_early_indicators(kadane_results, threshold_idx=3000)

    if early_indicators:
        print(f"\n   Found {len(early_indicators)} sensors that are early indicators of low RUL:")
        for ind in early_indicators:
            print(f"     - {ind['sensor']}: interval [{ind['start']}-{ind['end']}], "
                  f"{ind['dominant_rul']} ({ind['confidence']:.1f}%)")
    else:
        print("\n   No early indicators found")

    #Save early indicators to results folder
    with open("results/task3_results/early_indicators.txt", "w") as f:
        f.write(f"Early Warning Indicators (start < 3000, dominant RUL = Low/Very Low)\n")
        f.write(f"Found {len(early_indicators)} sensors\n\n")
        for ind in early_indicators:
            f.write(f"{ind['sensor']}: [{ind['start']}-{ind['end']}], "
                    f"{ind['dominant_rul']} ({ind['confidence']:.1f}%)\n")
    print(f"\n   Saved: results/task3_results/early_indicators.txt")

    #Generate visualizations for a few of the sensors as a graphic example
    print("\n3. Generating example Kadane visualizations..")
    example_sensors = list(kadane_results.keys())[:3]  #First 3 sensors
    for sensor in example_sensors:
        result = kadane_results[sensor]
        signal = df[sensor].values
        plot_kadane_result(
            sensor,
            signal,
            result['start'],
            result['end'],
            save_path=f"results/task3_results/{sensor}_kadane.png"
        )

    #Save all results as a CSV in the results folder
    print("\n4. Saving complete results..")
    results_df = pd.DataFrame([
        {
            'sensor': r['sensor'],
            'start': r['start'],
            'end': r['end'],
            'length': r['length'],
            'dominant_rul': r['dominant_rul'],
            'confidence': r['dominant_percentage']
        }
        for r in kadane_results.values()
    ])
    results_df.to_csv("results/task3_results/all_kadane_results.csv", index=False)
    print(f"   Saved: results/task3_results/all_kadane_results.csv")

    print("\n Task 3 Complete")

#Summary of all tasks
    print("\nSummary of All Completed Tasks - ")
    print(f"\n Task 1: Segmented 10 sensors")
    print(f" Task 2: Created {len(clusters)} clusters from {len(df)} time points")
    print(f" Task 3: Analyzed {len(kadane_results)} sensors, found {len(early_indicators)} early indicators")
    print(f"\n All results saved in 'results/' folder")

if __name__ == "__main__":
    #First run the toy examples to verify all three algorithms works as expected
    print("Task 1 Toy Verification Example:")
    toy_example_verification()
    print("\nTask 2 Toy Verification Example:")
    verify_clustering()
    print("\nTask 3 Toy Verification Example:")
    verify_kadane()

    #Then run main analysis on real data
    print("\nAnalyzing Data Set:\n")
    main()
