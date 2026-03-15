import numpy as np
import matplotlib.pyplot as plt

#this function finds the max subbarray sum and index's, will return the max sum, starting and ending index
def kadane(arr):
    #initialize the variables with their first element, these will track sums, start, end, etc.
    max_sum = arr[0]
    current_sum = arr[0]
    start = 0
    end = 0
    temp_start = 0

    #finding the max subarray with iterations and dynamic programming
    for i in range(1, len(arr)):
        if current_sum + arr[i] < arr[i]:
            current_sum = arr[i]
            temp_start = i
        else:
            current_sum += arr[i]
        #if the current subarray is better, then we will update the results accordingly
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    return max_sum, start, end

#this function utilizes sensor data in order to calculate the absolute differences to highlight the present deviation patterns
def preprocess_sensor(signal):
    #Absolute first differences calculation
    d = np.abs(np.diff(signal))

    #Subtract the mean for significant changes
    mean_d = np.mean(d)
    x = d - mean_d

    return x, d, mean_d

#this function runs kadane for all the sensors, and performs comparisons with the RUL data
def analyze_all_sensors(df, rul_categories):
    sensor_cols = [col for col in df.columns if 'sensor_' in col]
    results = {}

    for sensor in sensor_cols:
        signal = df[sensor].values

        #Preprocess
        x, raw_diffs, mean_d = preprocess_sensor(signal)

        #Apply Kadane
        max_sum, start, end = kadane(x)

        #Get RUL categories in this interval
        interval_rul = rul_categories[start:end + 1]
        unique, counts = np.unique(interval_rul, return_counts=True)

        if len(unique) > 0:
            dominant_idx = np.argmax(counts)
            dominant = unique[dominant_idx]
            percentage = counts[dominant_idx] / len(interval_rul) * 100
        else:
            dominant = -1
            percentage = 0

        #Map category number to label
        cat_labels = ["Very Low", "Low", "High", "Very High"]
        dominant_label = cat_labels[dominant] if dominant != -1 else "Unknown"

        results[sensor] = {
            'sensor': sensor,
            'start': start,
            'end': end,
            'length': end - start + 1,
            'max_sum': max_sum,
            'mean_diff': mean_d,
            'dominant_rul': dominant_label,
            'dominant_idx': dominant,
            'dominant_percentage': percentage,
            'rul_distribution': dict(zip([cat_labels[c] for c in unique], counts))
        }

        print(f"\n{sensor}:")
        print(f"  Max deviation interval: [{start}, {end}] (length: {end - start + 1})")
        print(f"  Dominant RUL: {dominant_label} ({percentage:.1f}%)")

    return results

#this function filters through all sensors to find the ones that have early occuring max deviation, and signs of low RUL
def find_early_indicators(results, threshold_idx=3000):
    early_indicators = []
    for sensor, result in results.items():
        #Check if interval starts early (first 30% of data)
        if result['start'] < threshold_idx:
            #Check if dominant RUL is low (Very Low or Low)
            if result['dominant_rul'] in ['Very Low', 'Low']:
                early_indicators.append({
                    'sensor': sensor,
                    'start': result['start'],
                    'end': result['end'],
                    'dominant_rul': result['dominant_rul'],
                    'confidence': result['dominant_percentage']
                })
    return early_indicators

#Funtion will make a visualization for task 3, ploting the sensor signals and their max deviations highlighted
def plot_kadane_result(sensor_name, signal, start, end, save_path=None):
    plt.figure(figsize=(12, 4))
    #Plot full signal
    plt.plot(signal, 'b-', linewidth=0.8, alpha=0.7)

    #Highlight the max deviation interval
    plt.axvspan(start, end, alpha=0.3, color='red', label='Max Deviation')

    plt.title(f"{sensor_name} - Max Deviation Interval [{start}:{end}]")
    plt.xlabel("Time Index")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close()

#Toy example for testing the kadane algorithm w a tiny array
def verify_kadane():
    #common kadane test array used for this toy example
    test_arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"Test array: {test_arr}")

    max_sum, start, end = kadane(test_arr)
    print(f"Max subarray: indices [{start}, {end}], sum={max_sum}")
    print(f"Subarray: {test_arr[start:end + 1]}")
    print(f"Expected: [4, -1, 2, 1] w/ sum of 6")

    #Test preprocessing
    print("\nTesting preprocessing:")
    test_signal = [1, 1, 1, 5, 5, 5, 1, 1, 1]
    x, d, mean_d = preprocess_sensor(np.array(test_signal))

    print(f"Original signal: {test_signal}")
    print(f"Absolute differences (d): {d}")
    print(f"Mean of d: {mean_d:.2f}")
    print(f"Preprocessed (x): {[f'{val:.2f}' for val in x]}")

    max_sum, start, end = kadane(x)
    print(f"Max deviation interval: [{start}, {end}]")

    return True
