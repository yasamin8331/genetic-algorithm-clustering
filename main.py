import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load and preprocess the dataset
data_path = r"Superstore.csv"
dataframe = pd.read_csv(data_path, encoding='Windows-1252')
dataframe = dataframe.ffill().bfill()  # Handle missing data

# Identify numerical columns
num_cols = dataframe.select_dtypes(include=['int64', 'float64']).columns

# Encode categorical columns
cat_cols = dataframe.select_dtypes(exclude=['int64', 'float64']).columns
encoder = LabelEncoder()
for col in cat_cols:
    dataframe[col] = encoder.fit_transform(dataframe[col])

# Normalize numerical data
data_matrix = dataframe[num_cols].values
normalizer = StandardScaler()
data_matrix = normalizer.fit_transform(data_matrix)

# Fitness calculation
def calculate_fitness(centroids, data_matrix):
    dist = np.linalg.norm(data_matrix[:, None] - centroids, axis=2)
    cluster_labels = np.argmin(dist, axis=1)
    score = -np.sum(np.min(dist, axis=1))  # Negative distance sum
    return score, cluster_labels

# Population generation
def create_population(cluster_count, feature_count, pop_size):
    min_vals, max_vals = data_matrix.min(axis=0), data_matrix.max(axis=0)
    return np.random.uniform(min_vals, max_vals, size=(pop_size, cluster_count, feature_count))

# Cross-breeding
def perform_crossover(parent_a, parent_b):
    split_point = np.random.randint(1, parent_a.shape[0])
    offspring_a = np.vstack((parent_a[:split_point], parent_b[split_point:]))
    offspring_b = np.vstack((parent_b[:split_point], parent_a[split_point:]))
    return offspring_a, offspring_b

# Mutation operation
def apply_mutation(offspring, min_vals, max_vals, mutation_chance=0.1):
    if np.random.rand() < mutation_chance:
        row_idx = np.random.randint(offspring.shape[0])
        col_idx = np.random.randint(offspring.shape[1])
        offspring[row_idx, col_idx] += np.random.normal()
        offspring[row_idx, col_idx] = np.clip(offspring[row_idx, col_idx], min_vals[col_idx], max_vals[col_idx])
    return offspring

# Genetic algorithm for clustering
def run_genetic_algorithm(data_matrix, cluster_count, generations=50, pop_size=10, mutation_chance=0.1):
    feature_count = data_matrix.shape[1]
    population = create_population(cluster_count, feature_count, pop_size)
    optimal_solution = None
    highest_fitness = -np.inf
    min_vals, max_vals = data_matrix.min(axis=0), data_matrix.max(axis=0)

    for gen in range(generations):
        fitness_results = []
        for member in population:
            fitness, labels = calculate_fitness(member, data_matrix)
            fitness_results.append((fitness, member, labels))
            if fitness > highest_fitness:
                highest_fitness = fitness
                optimal_solution = (member, labels)

        fitness_results.sort(reverse=True, key=lambda x: x[0])
        next_gen = []

        for idx in range(0, pop_size, 2):
            parent_a, parent_b = fitness_results[idx][1], fitness_results[idx + 1][1]
            child_a, child_b = perform_crossover(parent_a, parent_b)
            next_gen.append(apply_mutation(child_a, min_vals, max_vals, mutation_chance))
            next_gen.append(apply_mutation(child_b, min_vals, max_vals, mutation_chance))

        population = np.array(next_gen)

    return optimal_solution

# Execute genetic algorithm
cluster_count = 3
best_centroids, best_labels = run_genetic_algorithm(data_matrix, cluster_count)

# Compare with K-Means
kmeans_model = KMeans(n_clusters=cluster_count, random_state=42)
kmeans_labels = kmeans_model.fit_predict(data_matrix)

# Calculate silhouette scores
ga_silhouette_score = silhouette_score(data_matrix, best_labels)
kmeans_silhouette_score = silhouette_score(data_matrix, kmeans_labels)
print(f"GA Clustering Silhouette Score: {ga_silhouette_score}")
print(f"K-Means Clustering Silhouette Score: {kmeans_silhouette_score}")

# Visualize results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=best_labels, cmap='viridis', edgecolor='k')
plt.title("Clusters from Genetic Algorithm")
plt.subplot(1, 2, 2)
plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k')
plt.title("Clusters from K-Means")
plt.show()
