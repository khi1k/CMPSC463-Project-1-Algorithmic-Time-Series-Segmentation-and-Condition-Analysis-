import numpy as np
import matplotlib.pyplot as plt
import random

min_segments = 100 #first i start by setting the minimum number of segments when splitting, to help mitigate the amount of segments for this large dataset

#this function handles the splitting of task 1 by splitting signals using divide and conquer and the variance threshold
def segmentation_rec(signal, start, end, threshold, segments):
    if end-start <= min_segments: #if the segment is smaller than the min, stop splitting
        segments.append([start,end])
        return
    segment = signal[start:end]
    variance = np.var(segment)#calculate the variance of the current segment

    if variance > threshold: #if the variance > threshold, then we will split in the middle
        mid = (start+end)//2
        #recursion to do the left and right segment halves
        segmentation_rec(signal, start, mid, threshold, segments)
        segmentation_rec(signal, mid, end,threshold, segments )

    else: #else we just mark the mark it as a stable segment
        segments.append([start,end])

def segment_sensor(signal): #begins recursive segmentation for sensors
    threshold = 0.1 * np.var(signal) #basing the threshold on signal variance bc it helps w detection of unstable areas
    segments = []
    segmentation_rec(signal, 0, len(signal), threshold, segments) #beginning segmentation on the signal
    segments.sort(key=lambda x: x[0]) #sorting based on the order segments appear in chronologically
    return segments

def sensor_select(sensor_list, seed=42): #This function will randomly select 10 sensors from the list
    random.seed(seed) #sets a seed so we can use the same sensors each time its ran, for ease of submission
    return random.sample(sensor_list, 10)

def analyze_sensors(df, sensors): #segmentation for each sensor then stores the results of it
    results = {}

    for sensor in sensors:
        signal = df[sensor].values
        segments = segment_sensor(signal)
        results[sensor] = {
            "signal": signal,
            "segments": segments,
            "complexity": len(segments) #calculation of the complexity score needed in step 4 of task 1
        }
        print(f"{sensor}: {len(segments)} segments")
    return results

def plot_sensor(sensor_name, signal, segments): #used to visualise the sensor signals, with red dashed lines for the segment boundaries
    plt.figure(figsize=(12,4))
    plt.plot(signal)

    for start, end in segments: #we mark the boundaries w vertical lines that are red and dashed
        plt.axvline(start, color="red", linestyle="--", alpha=0.5)
    plt.title(f"{sensor_name}, #of segments: {len(segments)}")
    plt.xlabel("time index")
    plt.ylabel("sensor value")

    plt.show()

#step 5: analysis of if the boundaries of segments can correspond to the changes in the RUL categories
def analyze_rul_relation(results, rul_categories):
    print("Step 5: Segmentation vs. RUL Results - ")

    for sensor, data in results.items():
        segments = data["segments"]
        print(f"\n{sensor} (Total segments: {len(segments)})")

        #Analyze each segment's dominant RUL category
        print("  Segment breakdown:")
        for i, (start, end) in enumerate(segments[:5]):  #Show first 5 segments
            segment_rul = rul_categories[start:end]

            #Finds the most frequent category in the segment
            unique, counts = np.unique(segment_rul, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_cat = dominant_idx
            percentage = counts[dominant_idx] / len(segment_rul) * 100

            #Convert category number to label
            cat_labels = ["Very Low", "Low", "High", "Very High"]

            print(f"    Seg {i + 1} ({start}-{end}): mostly {cat_labels[dominant_cat]} ({percentage:.1f}%)")

        #Check if RUL categories change at the segment boundaries
        print("  Boundary analysis:")
        boundaries = [start for start, _ in segments[1:]]#skips first segment
        changes_detected = 0

        for boundary in boundaries[:5]:  #Checks the first 5 boundaries
            if boundary > 5 and boundary < len(rul_categories) - 5:
                before = rul_categories[boundary - 5:boundary]
                after = rul_categories[boundary:boundary + 5]

                before_dominant = np.bincount(before).argmax() if len(before) > 0 else -1
                after_dominant = np.bincount(after).argmax() if len(after) > 0 else -1

                if before_dominant != after_dominant:
                    changes_detected += 1
                    print(
                        f"    Boundary at {boundary}: RUL changes from {cat_labels[before_dominant]} to {cat_labels[after_dominant]}")

        if changes_detected == 0:
            print("    No RUL category changes at checked boundaries")

        print(f"  Summary: Segment boundaries {'DO' if changes_detected > 0 else 'DO NOT'} align with RUL category changes")

#step 6: this function analyzes the complexity across sensors in order to understand and explain their behavior
def discuss_temporal_dynamics(results):

    print("Step 6: Temporal dynamics Analysis - ")
    #Collect complexity scores and rank the sensors by them
    complexities = [(sensor, data["complexity"]) for sensor, data in results.items()]
    complexities.sort(key=lambda x: x[1])  #Sort from low to high scores

    #Display all the sensors in a ranking
    print("\nSensors ranked by complexity (stable → dynamic):")
    for sensor, comp in complexities:
        print(f"  {sensor}: {comp} segments")

    #Calculate the summary statistics
    avg_complexity = np.mean([c for _, c in complexities])
    min_sensor, min_comp = complexities[0]
    max_sensor, max_comp = complexities[-1]

    print(f"\n Statistics:")
    print(f"  Average complexity: {avg_complexity:.1f} segments")
    print(f"  Most stable sensor: {min_sensor} ({min_comp} segments)")
    print(f"  Most dynamic sensor: {max_sensor} ({max_comp} segments)")

    #Discussion of what this means
    print(f"\nInterpretation:")
    print(f"   Sensors with low complexity ({min_comp} segments) are relatively stable")
    print(f"    throughout the 10,000 time points. They don't change behavior much,")
    print(f"    suggesting consistent machine operation in those measurements.")

    print(f"\n  Sensors with a high complexity ({max_comp} segments) are constantly")
    print(f"    changing between stable and unstable regions. This could indicate")
    print(f"    these measurements are more sensitive to machine condition changes.")

    print(f"\n  The average complexity of {avg_complexity:.1f} segments suggests that")
    if avg_complexity < 10:
        print(f"    most sensors are relatively stable with occasional changes.")
    elif avg_complexity < 20:
        print(f"    sensors show moderate fluctuation in behavior.")
    else:
        print(f"    sensors are highly dynamic with frequent state changes.")

#required to verify the algorithm works correctly, using a small array signal and testing the segmentation on it
def toy_example_verification():
    global min_segments
    original_min = min_segments
    min_segments = 5
    #Single test with three distinct regions (5 values each)
    toy_signal = np.array([1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 10, 10, 10, 10, 10])

    print(f"Toy signal: {toy_signal}")
    print(f"Expected: 4 stable regions")
    segments = segment_sensor(toy_signal)#Run segmentation on the toy example

    print(f"\nSegments found: {segments}")
    print(f"Number of segments: {len(segments)}")

    #Visualize the toy example with a graph
    plot_sensor("Toy Example Verification: Task 1", toy_signal, segments)

    #Restore original min_segments
    min_segments = original_min

    return segments