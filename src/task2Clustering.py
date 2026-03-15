import numpy as np

#computes the euclidean distance btw two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def cluster_recursive(data, indices, k): #this function will use recursion to split our data till we have k number of clusters
    #stops recursion if data is too small or we need only 1 cluster
    if k == 1 or len(data) <= 1:
        return [indices.tolist()]
    #splits into clusters based on their distances from the centroid variable
    centroid = np.mean(data, axis=0) #this will find the current center point in the data
    distances = [euclidean_distance(p, centroid) for p in data]#then calculates the distance from each point to the center
    median_dist = np.median(distances)#finally, we find the median of distances so we can use it to split the data

    left_mask = distances <= median_dist #goes left if it points closer to the median
    right_mask = ~left_mask #goes right if it points farther than the median

    #This will handle edge cases if theres a split that creates empty groups
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return [indices.tolist()]

    #Recursively split into halves of k
    left_k = k // 2
    right_k = k - left_k

    #recursive calls for the left and right clusters
    left_clusters = cluster_recursive(data[left_mask], indices[left_mask], left_k)
    right_clusters = cluster_recursive(data[right_mask], indices[right_mask], right_k)
    return left_clusters + right_clusters

#this function clusters the time points into k clusters
def cluster_sensors(df, k=4):
    #Get the sensor data of 10000 rows and 52 columns
    sensor_cols = [col for col in df.columns if 'sensor_' in col]
    data = df[sensor_cols].values
    indices = np.arange(len(data))


    print(f"\nClustering {data.shape[0]} points with {data.shape[1]} features...")
    clusters = cluster_recursive(data, indices, k)#runs the clustering algorithm
    print(f"Found {len(clusters)} clusters")

    return clusters


def analyze_clusters(clusters, rul_categories):
    print("Task 2: Majority class of each cluster")
    cat_labels = ["Very Low", "Low", "High", "Very High"]#uses labels for converting RUL categories into readable names
    results = []

    #loops thru each cluster to determine the RUL category that it is
    for i, cluster_idx in enumerate(clusters):
        #get all the RUL values in the cluster, then counts the occurences of each RUL category
        cluster_rul = rul_categories[cluster_idx]
        unique, counts = np.unique(cluster_rul, return_counts=True)

        #Now we find the category with the highest amount of occurences
        majority_idx = np.argmax(counts)
        majority_cat = unique[majority_idx]
        percentage = (counts[majority_idx] / len(cluster_rul)) * 100

        results.append({
            'cluster': i,
            'size': len(cluster_rul),
            'majority': cat_labels[majority_cat],
            'percentage': percentage,
            'distribution': dict(zip([cat_labels[c] for c in unique], counts)) #stores the distribution of categories within the cluster
        })

        print(f"\nCluster {i}: {len(cluster_rul)} points")
        print(f"  Majority: {cat_labels[majority_cat]} ({percentage:.1f}%)")
        print(f"  Distribution: {dict(zip([cat_labels[c] for c in unique], counts))}")

    return results

#discussion of how cluster mapping affects interpreting RUL
def discuss_mapping(results):
    print("\nCluster mapping discussion - ")

    #prints a summary of the custer dominant RUL categories
    for r in results:
        print(f"Cluster {r['cluster']}: {r['percentage']:.1f}% {r['majority']}")

    #identifies clusters that have majority of their points belonging to the same category
    pure = [r for r in results if r['percentage'] > 80]
    if pure:
        print(f"\n{pure} cluster(s) show strong alignment with one RUL category")
    else:
        print("\nClusters are mixed - sensor measurements don't perfectly separate RUL states")

#Task 2 Verification
def verify_clustering():
    print("Toy example: Clustering Verification")

    #Creation a test data seed which represents 3 clusters
    np.random.seed(42)
    data = np.vstack([
        np.random.normal(1, 0.3, (5, 2)),  #Cluster 0
        np.random.normal(5, 0.3, (5, 2)),  #Cluster 1
        np.random.normal(9, 0.3, (5, 2))  #Cluster 2
    ])

    #Using fake RUL categories for testing the labeling of clusters
    fake_rul = np.array([0] * 5 + [1] * 5 + [2] * 5)
    print(f"Test data: {data.shape[0]} points in 2D space")

    #Manual clustering for toy example, creating indices for each point
    indices = np.arange(len(data))
    clusters = cluster_recursive(data, indices, 3)#recursive clustering algorithm begins running

    print(f"Clusters found: {len(clusters)}")
    for i, c in enumerate(clusters):
        print(f"  Cluster {i}: {len(c)} points")

    analyze_clusters(clusters, fake_rul)#verifies that the clusters map to RUL categories as they should
    return clusters