import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_wine, load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, SpectralClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import pandas as pd
import seaborn as sns
from typing import List, Tuple, Dict, Any
import psutil
import os
import gc
from memory_profiler import profile
import tracemalloc

# Your OriginScale algorithm (with minor fixes)
class RadialClustering:
    """
    A clustering algorithm using a novel radial distribution initialization
    with memory-efficient batch processing.
    """
    def __init__(self, n_clusters: int = 3, tol: float = 1e-4, max_iter: int = 300, batch_size: int = 1000):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.centroids_ = None
        self.labels_ = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initializes centroids using the radial distribution method."""
        # Memory-efficient distance calculation for large datasets
        n_samples = X.shape[0]
        chunk_size = min(10000, n_samples)
        
        distances_from_origin = np.zeros(n_samples)
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            distances_from_origin[i:end_idx] = np.linalg.norm(X[i:end_idx], axis=1)
        
        avg_distance = np.mean(distances_from_origin)
        
        target_distances = np.linspace(
            start=avg_distance / self.n_clusters,
            stop=avg_distance,
            num=self.n_clusters
        )
        
        initial_centroids = []
        selected_indices = set()
        
        for target_dist in target_distances:
            diffs = np.abs(distances_from_origin - target_dist)
            
            for idx in np.argsort(diffs):
                if idx not in selected_indices:
                    closest_point_idx = idx
                    break
            else:
                closest_point_idx = np.argmin(diffs)
            
            initial_centroids.append(X[closest_point_idx])
            selected_indices.add(closest_point_idx)
        
        return np.array(initial_centroids)

    def _assign_points_batch(self, X: np.ndarray) -> np.ndarray:
        """Memory-efficient point assignment using batch processing."""
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            batch_X = X[i:end_idx]
            
            # Calculate distances for this batch only
            batch_distances = np.zeros((end_idx - i, self.n_clusters))
            for j, centroid in enumerate(self.centroids_):
                batch_distances[:, j] = np.linalg.norm(batch_X - centroid, axis=1)
            
            labels[i:end_idx] = np.argmin(batch_distances, axis=1)
        
        return labels

    def fit(self, X: np.ndarray) -> 'RadialClustering':
        """Fits the algorithm to the data X using memory-efficient batch processing."""
        self.centroids_ = self._initialize_centroids(X)
        
        for _ in range(self.max_iter):
            # Memory-efficient assignment using batching
            self.labels_ = self._assign_points_batch(X)
            
            # Standard centroid update
            new_centroids = np.array([
                X[self.labels_ == i].mean(axis=0) if np.sum(self.labels_ == i) > 0 
                else self.centroids_[i] for i in range(self.n_clusters)
            ])
            
            if np.linalg.norm(new_centroids - self.centroids_) < self.tol:
                break
                
            self.centroids_ = new_centroids
        
        return self

    def fit_predict(self, X: np.ndarray):
        """Fits the algorithm and returns cluster labels."""
        self.fit(X)
        return self.labels_

class MemoryMonitor:
    """Helper class to monitor memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = None
        
    def start_monitoring(self):
        """Start memory monitoring."""
        gc.collect()  # Force garbage collection
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        tracemalloc.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring and return peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = current_memory - self.baseline_memory if self.baseline_memory else 0
        
        # Also get tracemalloc stats
        try:
            current_size, peak_size = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            tracemalloc_peak = peak_size / 1024 / 1024  # Convert to MB
        except:
            tracemalloc_peak = 0
            
        return max(peak_memory, tracemalloc_peak)

class ClusteringBenchmark:
    """Comprehensive benchmarking framework for clustering algorithms."""
    
    def __init__(self):
        self.results = []
        
    def generate_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate various datasets for testing."""
        datasets = {}
        
        # Synthetic datasets
        datasets['blobs_easy'] = make_blobs(n_samples=50000, centers=3, random_state=42)
        datasets['blobs_hard'] = make_blobs(n_samples=50000, centers=3, cluster_std=2.0, random_state=42)
        datasets['blobs_uneven'] = make_blobs(n_samples=50000, centers=3, 
                                            cluster_std=[1.0, 2.5, 0.5], random_state=42)
        datasets['moons'] = make_moons(n_samples=50000, noise=0.1, random_state=42)
        datasets['circles'] = make_circles(n_samples=50000, noise=0.1, factor=0.5, random_state=42)
        datasets['moons_noisy'] = make_moons(n_samples=50000, noise=0.2, random_state=42)
        
        # High-dimensional
        datasets['blobs_high_dim'] = make_blobs(n_samples=50000, centers=3, n_features=20, random_state=42)
        
        # Large-scale
        datasets['blobs_large'] = make_blobs(n_samples=50000, centers=3, random_state=42)
        
        # Extra large for memory testing
        datasets['blobs_xlarge'] = make_blobs(n_samples=50000, centers=3, random_state=42)
        
        # Real-world datasets
        iris = load_iris()
        datasets['iris'] = (iris.data, iris.target)
        
        wine = load_wine()
        datasets['wine'] = (wine.data, wine.target)
        
        digits = load_digits()
        # Use PCA to reduce dimensionality for visualization
        pca = PCA(n_components=10)
        digits_reduced = pca.fit_transform(digits.data)
        datasets['digits'] = (digits_reduced, digits.target)
        
        return datasets
    
    def get_algorithms(self, n_clusters: int) -> Dict[str, Any]:
        """Get dictionary of clustering algorithms to test."""
        algorithms = {
            'KMeans++': KMeans(n_clusters=n_clusters, init='k-means++', random_state=42),
            'KMeans_random': KMeans(n_clusters=n_clusters, init='random', random_state=42),
            'RadialClustering': RadialClustering(n_clusters=n_clusters),
            'MiniBatchKMeans': MiniBatchKMeans(n_clusters=n_clusters, random_state=42),
            'GMM': GaussianMixture(n_components=n_clusters, random_state=42),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=n_clusters),
            'Spectral': SpectralClustering(n_clusters=n_clusters, random_state=42),
        }
        
        return algorithms
    
    def evaluate_algorithm(self, algorithm, X: np.ndarray, y_true: np.ndarray, 
                         alg_name: str, dataset_name: str) -> Dict[str, Any]:
        """Evaluate a single algorithm on a dataset."""
        try:
            # Standardize data for most algorithms
            if alg_name not in ['DBSCAN']:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Initialize memory monitor
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()
            
            # Time the algorithm
            start_time = time.time()
            
            if hasattr(algorithm, 'fit_predict'):
                y_pred = algorithm.fit_predict(X_scaled)
            else:
                algorithm.fit(X_scaled)
                y_pred = algorithm.labels_
            
            fit_time = time.time() - start_time
            
            # Stop memory monitoring
            peak_memory = memory_monitor.stop_monitoring()
            
            # Calculate metrics
            ari = adjusted_rand_score(y_true, y_pred)
            nmi = normalized_mutual_info_score(y_true, y_pred)
            ami = adjusted_mutual_info_score(y_true, y_pred)
            
            # Internal metrics (handle edge cases)
            try:
                silhouette = silhouette_score(X_scaled, y_pred)
            except:
                silhouette = -1
            
            try:
                calinski_harabasz = calinski_harabasz_score(X_scaled, y_pred)
            except:
                calinski_harabasz = -1
            
            try:
                davies_bouldin = davies_bouldin_score(X_scaled, y_pred)
            except:
                davies_bouldin = -1
            
            return {
                'Algorithm': alg_name,
                'Dataset': dataset_name,
                'ARI': ari,
                'NMI': nmi,
                'AMI': ami,
                'Silhouette': silhouette,
                'Calinski_Harabasz': calinski_harabasz,
                'Davies_Bouldin': davies_bouldin,
                'Time': fit_time,
                'Memory_MB': peak_memory,
                'Data_Size': X.shape[0],
                'Data_Dimensions': X.shape[1],
                'Status': 'Success'
            }
            
        except Exception as e:
            return {
                'Algorithm': alg_name,
                'Dataset': dataset_name,
                'ARI': -1,
                'NMI': -1,
                'AMI': -1,
                'Silhouette': -1,
                'Calinski_Harabasz': -1,
                'Davies_Bouldin': -1,
                'Time': -1,
                'Memory_MB': -1,
                'Data_Size': X.shape[0],
                'Data_Dimensions': X.shape[1],
                'Status': f'Error: {str(e)}'
            }
    
    def run_benchmark(self) -> pd.DataFrame:
        """Run the complete benchmark."""
        datasets = self.generate_datasets()
        self.results = []
        
        for dataset_name, (X, y_true) in datasets.items():
            n_clusters = len(np.unique(y_true))
            algorithms = self.get_algorithms(n_clusters)
            
            print(f"Testing on {dataset_name} (n_clusters={n_clusters}, shape={X.shape})...")
            
            for alg_name, algorithm in algorithms.items():
                result = self.evaluate_algorithm(algorithm, X, y_true, alg_name, dataset_name)
                self.results.append(result)
                print(f"  {alg_name}: ARI={result['ARI']:.3f}, Time={result['Time']:.3f}s, Memory={result['Memory_MB']:.2f}MB")
        
        return pd.DataFrame(self.results)
    
    def visualize_performance_results(self, df: pd.DataFrame, save_dir: str = None):
        """Create visualizations of clustering performance results."""
        # Filter out failed runs
        df_clean = df[df['Status'] == 'Success'].copy()
        
        # Create subplots for all metrics
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        
        # External validation metrics (higher is better)
        metrics_higher = [
            ('ARI', 'Adjusted Rand Index (Higher is Better)'),
            ('NMI', 'Normalized Mutual Information (Higher is Better)'),
            ('AMI', 'Adjusted Mutual Information (Higher is Better)'),
            ('Silhouette', 'Silhouette Score (Higher is Better)')
        ]
        
        # Internal validation metrics
        metrics_calinski = [('Calinski_Harabasz', 'Calinski-Harabasz Index (Higher is Better)')]
        metrics_davies = [('Davies_Bouldin', 'Davies-Bouldin Index (Lower is Better)')]
        
        # Plot external validation metrics
        for i, (metric, title) in enumerate(metrics_higher):
            row, col = i // 2, i % 2
            pivot = df_clean.pivot(index='Dataset', columns='Algorithm', values=metric)
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[row, col])
            axes[row, col].set_title(title)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        # Plot Calinski-Harabasz Index
        pivot_ch = df_clean.pivot(index='Dataset', columns='Algorithm', values='Calinski_Harabasz')
        sns.heatmap(pivot_ch, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[2, 0])
        axes[2, 0].set_title('Calinski-Harabasz Index (Higher is Better)')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # Plot Davies-Bouldin Index (lower is better, so invert colormap)
        pivot_db = df_clean.pivot(index='Dataset', columns='Algorithm', values='Davies_Bouldin')
        sns.heatmap(pivot_db, annot=True, fmt='.3f', cmap='RdYlBu', ax=axes[2, 1])
        axes[2, 1].set_title('Davies-Bouldin Index (Lower is Better)')
        axes[2, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'performance_results.png'))
        plt.close(fig)
    
    def visualize_time_memory_analysis(self, df: pd.DataFrame, save_dir: str = None):
        """Create comprehensive visualizations for time and memory analysis, and save plots if save_dir is provided."""
        df_clean = df[df['Status'] == 'Success'].copy()
        # Convert memory to KB for plotting
        df_clean['Memory_KB'] = df_clean['Memory_MB'] * 1024
        
        # Create a 2x3 subplot layout for time and memory analysis
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # 1. Execution Time Heatmap
        pivot_time = df_clean.pivot(index='Dataset', columns='Algorithm', values='Time')
        sns.heatmap(pivot_time, annot=True, fmt='.3f', cmap='RdYlBu', ax=axes[0, 0])
        axes[0, 0].set_title('Execution Time (seconds) - Lower is Better')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Memory Usage Heatmap (in KB)
        pivot_memory = df_clean.pivot(index='Dataset', columns='Algorithm', values='Memory_KB')
        sns.heatmap(pivot_memory, annot=True, fmt='.0f', cmap='RdYlBu', ax=axes[0, 1])
        axes[0, 1].set_title('Memory Usage (KB) - Lower is Better')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Time vs Data Size
        algorithms = df_clean['Algorithm'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        for i, alg in enumerate(algorithms):
            alg_data = df_clean[df_clean['Algorithm'] == alg]
            axes[0, 2].scatter(alg_data['Data_Size'], alg_data['Time'], 
                             label=alg, alpha=0.7, color=colors[i], s=60)
        axes[0, 2].set_xlabel('Data Size (number of samples)')
        axes[0, 2].set_ylabel('Execution Time (seconds)')
        axes[0, 2].set_title('Execution Time vs Data Size')
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 2].set_xscale('log')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Memory vs Data Size (in KB)
        for i, alg in enumerate(algorithms):
            alg_data = df_clean[df_clean['Algorithm'] == alg]
            axes[1, 0].scatter(alg_data['Data_Size'], alg_data['Memory_KB'], 
                             label=alg, alpha=0.7, color=colors[i], s=60)
        axes[1, 0].set_xlabel('Data Size (number of samples)')
        axes[1, 0].set_ylabel('Memory Usage (KB)')
        axes[1, 0].set_title('Memory Usage vs Data Size')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Time vs Dimensions
        for i, alg in enumerate(algorithms):
            alg_data = df_clean[df_clean['Algorithm'] == alg]
            axes[1, 1].scatter(alg_data['Data_Dimensions'], alg_data['Time'], 
                             label=alg, alpha=0.7, color=colors[i], s=60)
        axes[1, 1].set_xlabel('Data Dimensions (number of features)')
        axes[1, 1].set_ylabel('Execution Time (seconds)')
        axes[1, 1].set_title('Execution Time vs Data Dimensions')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Memory vs Time (Efficiency Plot, in KB)
        for i, alg in enumerate(algorithms):
            alg_data = df_clean[df_clean['Algorithm'] == alg]
            axes[1, 2].scatter(alg_data['Time'], alg_data['Memory_KB'], 
                             label=alg, alpha=0.7, color=colors[i], s=60)
        axes[1, 2].set_xlabel('Execution Time (seconds)')
        axes[1, 2].set_ylabel('Memory Usage (KB)')
        axes[1, 2].set_title('Memory vs Time (Lower-left is better)')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 2].set_xscale('log')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'time_memory_analysis.png'))
        plt.close(fig)

    def create_performance_comparison_charts(self, df: pd.DataFrame, save_dir: str = None):
        """Create bar charts comparing average performance metrics."""
        df_clean = df[df['Status'] == 'Success'].copy()
        
        # Calculate average metrics for each algorithm
        avg_metrics = df_clean.groupby('Algorithm').agg({
            'Time': 'mean',
            'Memory_MB': 'mean',
            'ARI': 'mean',
            'Silhouette': 'mean'
        }).round(4)
        
        # Create 2x2 subplot for comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Average Execution Time
        avg_metrics['Time'].plot(kind='bar', ax=axes[0, 0], color='lightcoral')
        axes[0, 0].set_title('Average Execution Time by Algorithm')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Average Memory Usage
        avg_metrics['Memory_MB'].plot(kind='bar', ax=axes[0, 1], color='lightblue')
        axes[0, 1].set_title('Average Memory Usage by Algorithm')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Average ARI Score
        avg_metrics['ARI'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title('Average ARI Score by Algorithm')
        axes[1, 0].set_ylabel('ARI Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Average Silhouette Score
        avg_metrics['Silhouette'].plot(kind='bar', ax=axes[1, 1], color='gold')
        axes[1, 1].set_title('Average Silhouette Score by Algorithm')
        axes[1, 1].set_ylabel('Silhouette Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'performance_comparison.png'))
        plt.close(fig)
        
        return avg_metrics
    
    def create_efficiency_ranking(self, df: pd.DataFrame):
        """Create efficiency ranking based on time and memory usage."""
        df_clean = df[df['Status'] == 'Success'].copy()
        
        # Calculate efficiency metrics
        efficiency_data = []
        algorithms = df_clean['Algorithm'].unique()
        
        for alg in algorithms:
            alg_data = df_clean[df_clean['Algorithm'] == alg]
            
            # Calculate averages
            avg_time = alg_data['Time'].mean()
            avg_memory = alg_data['Memory_MB'].mean()
            avg_ari = alg_data['ARI'].mean()
            avg_silhouette = alg_data['Silhouette'].mean()
            
            # Calculate efficiency scores (higher is better)
            # Time efficiency: inverse of average time
            time_efficiency = 1 / (avg_time + 1e-6)
            # Memory efficiency: inverse of average memory usage
            memory_efficiency = 1 / (avg_memory + 1e-6)
            # Combined efficiency: weighted average
            combined_efficiency = 0.4 * time_efficiency + 0.3 * memory_efficiency + 0.3 * avg_ari
            
            efficiency_data.append({
                'Algorithm': alg,
                'Avg_Time': avg_time,
                'Avg_Memory': avg_memory,
                'Avg_ARI': avg_ari,
                'Avg_Silhouette': avg_silhouette,
                'Time_Efficiency': time_efficiency,
                'Memory_Efficiency': memory_efficiency,
                'Combined_Efficiency': combined_efficiency
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df = efficiency_df.sort_values('Combined_Efficiency', ascending=False)
        
        return efficiency_df
    
    def visualize_results(self, df: pd.DataFrame, save_dir: str = None):
        """Create all visualizations."""
        print("Creating performance visualizations...")
        self.visualize_performance_results(df, save_dir=save_dir)
        
        print("Creating time and memory analysis...")
        self.visualize_time_memory_analysis(df, save_dir=save_dir)
        
        print("Creating performance comparison charts...")
        avg_metrics = self.create_performance_comparison_charts(df, save_dir=save_dir)
        
        print("Creating radar chart...")
        self._create_radar_chart(df[df['Status'] == 'Success'], save_dir=save_dir)
        
        return avg_metrics
    
    def _create_radar_chart(self, df: pd.DataFrame, save_dir: str = None):
        """Create radar chart for algorithm comparison."""
        from math import pi
        
        # Calculate normalized scores for each algorithm
        algorithms = df['Algorithm'].unique()
        
        # Metrics for radar chart (normalize all to 0-1 scale where 1 is best)
        metrics = ['ARI', 'NMI', 'AMI', 'Silhouette', 'Calinski_Harabasz']
        
        # Calculate mean scores and normalize
        radar_data = {}
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]
            scores = []
            for metric in metrics:
                values = alg_data[metric].values
                values = values[values != -1]  # Remove error values
                if len(values) > 0:
                    mean_score = np.mean(values)
                    # Normalize to 0-1 scale relative to all algorithms
                    all_values = df[df[metric] != -1][metric].values
                    if len(all_values) > 0:
                        normalized = (mean_score - np.min(all_values)) / (np.max(all_values) - np.min(all_values))
                        scores.append(normalized)
                    else:
                        scores.append(0)
                else:
                    scores.append(0)
            radar_data[alg] = scores
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of metrics
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each algorithm
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        for i, (alg, scores) in enumerate(radar_data.items()):
            scores += scores[:1]  # Complete the circle
            ax.plot(angles, scores, 'o-', linewidth=2, label=alg, color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Algorithm Performance Comparison\n(Normalized Scores)', size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'radar_chart.png'))
        plt.close(fig)
    
    def summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for each algorithm."""
        df_clean = df[df['Status'] == 'Success'].copy()
        
        summary = df_clean.groupby('Algorithm').agg({
            'ARI': ['mean', 'std', 'min', 'max'],
            'NMI': ['mean', 'std', 'min', 'max'],
            'AMI': ['mean', 'std', 'min', 'max'],
            'Silhouette': ['mean', 'std', 'min', 'max'],
            'Calinski_Harabasz': ['mean', 'std', 'min', 'max'],
            'Davies_Bouldin': ['mean', 'std', 'min', 'max'],
            'Time': ['mean', 'std', 'min', 'max'],
            'Memory_MB': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        return summary
    
    def create_performance_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a comprehensive ranking of algorithms."""
        df_clean = df[df['Status'] == 'Success'].copy()
        
        # Calculate average normalized scores
        algorithms = df_clean['Algorithm'].unique()
        ranking_data = []
        
        # Metrics where higher is better
        higher_better = ['ARI', 'NMI', 'AMI', 'Silhouette', 'Calinski_Harabasz']
        # Metrics where lower is better  
        lower_better = ['Davies_Bouldin', 'Time', 'Memory_MB']
        
        for alg in algorithms:
            alg_data = df_clean[df_clean['Algorithm'] == alg]
            scores = {}
            
            # Process higher-is-better metrics
            for metric in higher_better:
                values = alg_data[metric].values
                values = values[values != -1]
                if len(values) > 0:
                    scores[f'{metric}_avg'] = np.mean(values)
                else:
                    scores[f'{metric}_avg'] = 0
            
            # Process lower-is-better metrics (invert for ranking)
            for metric in lower_better:
                values = alg_data[metric].values
                values = values[values != -1]
                if len(values) > 0:
                    scores[f'{metric}_avg'] = np.mean(values)
                else:
                    scores[f'{metric}_avg'] = float('inf')
            
            scores['Algorithm'] = alg
            ranking_data.append(scores)
        
        ranking_df = pd.DataFrame(ranking_data)
        return ranking_df

# Usage example
if __name__ == "__main__":
    # Run the benchmark
    benchmark = ClusteringBenchmark()
    results_df = benchmark.run_benchmark()
    
    # Display results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    # Show summary statistics
    summary = benchmark.summary_statistics(results_df)
    print("\nSUMMARY STATISTICS:")
    print(summary)
    
    # Show efficiency ranking
    efficiency_ranking = benchmark.create_efficiency_ranking(results_df)
    print("\nEFFICIENCY RANKING:")
    print(efficiency_ranking.round(4))
    
    # Show performance ranking
    ranking = benchmark.create_performance_ranking(results_df)
    print("\nPERFORMANCE RANKING:")
    print(ranking.round(4))
    
    # Show detailed results
    print("\nDETAILED RESULTS:")
    successful_results = results_df[results_df['Status'] == 'Success']
    print(successful_results[['Algorithm', 'Dataset', 'ARI', 'NMI', 'AMI', 'Silhouette', 
                             'Calinski_Harabasz', 'Davies_Bouldin', 'Time', 'Memory_MB']].to_string(index=False))
    
    # Visualize results and save plots
    plot_dir = "clustering_results_20250719_200123/plots"
    os.makedirs(plot_dir, exist_ok=True)
    avg_metrics = benchmark.visualize_results(results_df, save_dir=plot_dir)
    
    # Print rankings
    df_clean = results_df[results_df['Status'] == 'Success'].copy()
    
    print(f"\nALGORITHM RANKING (by average ARI):")
    avg_ari = df_clean.groupby('Algorithm')['ARI'].mean().sort_values(ascending=False)
    for i, (alg, score) in enumerate(avg_ari.items(), 1):
        print(f"{i}. {alg}: {score:.4f}")
    
    print(f"\nALGORITHM RANKING (by average Silhouette Score):")
    avg_sil = df_clean.groupby('Algorithm')['Silhouette'].mean().sort_values(ascending=False)
    for i, (alg, score) in enumerate(avg_sil.items(), 1):
        print(f"{i}. {alg}: {score:.4f}")
    
    print(f"\nALGORITHM RANKING (by average Execution Time - lower is better):")
    avg_time = df_clean.groupby('Algorithm')['Time'].mean().sort_values(ascending=True)
    for i, (alg, score) in enumerate(avg_time.items(), 1):
        print(f"{i}. {alg}: {score:.4f}s")
    
    print(f"\nALGORITHM RANKING (by average Memory Usage - lower is better):")
    avg_memory = df_clean.groupby('Algorithm')['Memory_MB'].mean().sort_values(ascending=True)
    for i, (alg, score) in enumerate(avg_memory.items(), 1):
        print(f"{i}. {alg}: {score:.2f}MB")
    
    print(f"\nALGORITHM RANKING (by average Davies-Bouldin Index - lower is better):")
    avg_db = df_clean.groupby('Algorithm')['Davies_Bouldin'].mean().sort_values(ascending=True)
    for i, (alg, score) in enumerate(avg_db.items(), 1):
        print(f"{i}. {alg}: {score:.4f}")
    
    # Create comprehensive analysis report
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("="*60)
    
    # Time and Memory Analysis
    print("\nTIME AND MEMORY ANALYSIS:")
    time_memory_analysis = df_clean.groupby('Algorithm').agg({
        'Time': ['mean', 'std', 'min', 'max'],
        'Memory_MB': ['mean', 'std', 'min', 'max'],
        'Data_Size': 'mean'
    }).round(4)
    print(time_memory_analysis)
    
    # Scalability Analysis
    print("\nSCALABILITY ANALYSIS:")
    scalability_data = []
    for alg in df_clean['Algorithm'].unique():
        alg_data = df_clean[df_clean['Algorithm'] == alg]
        
        # Calculate correlation between data size and time/memory
        try:
            time_correlation = np.corrcoef(alg_data['Data_Size'], alg_data['Time'])[0, 1]
            memory_correlation = np.corrcoef(alg_data['Data_Size'], alg_data['Memory_MB'])[0, 1]
        except:
            time_correlation = 0
            memory_correlation = 0
        
        scalability_data.append({
            'Algorithm': alg,
            'Time_Data_Size_Correlation': time_correlation,
            'Memory_Data_Size_Correlation': memory_correlation,
            'Avg_Time_Per_1K_Samples': alg_data['Time'].mean() / (alg_data['Data_Size'].mean() / 1000),
            'Avg_Memory_Per_1K_Samples': alg_data['Memory_MB'].mean() / (alg_data['Data_Size'].mean() / 1000)
        })
    
    scalability_df = pd.DataFrame(scalability_data)
    print(scalability_df.round(4))
    
    # Best Algorithm by Category
    print("\nBEST ALGORITHM BY CATEGORY:")
    categories = {
        'Best Overall Performance (ARI)': avg_ari.index[0],
        'Best Clustering Quality (Silhouette)': avg_sil.index[0],
        'Fastest Execution': avg_time.index[0],
        'Most Memory Efficient': avg_memory.index[0],
        'Best Trade-off (Efficiency)': efficiency_ranking.iloc[0]['Algorithm']
    }
    
    for category, best_alg in categories.items():
        print(f"{category}: {best_alg}")
    
    # Dataset-specific recommendations
    print("\nDATASET-SPECIFIC RECOMMENDATIONS:")
    dataset_recommendations = {}
    for dataset in df_clean['Dataset'].unique():
        dataset_data = df_clean[df_clean['Dataset'] == dataset]
        best_ari_alg = dataset_data.loc[dataset_data['ARI'].idxmax(), 'Algorithm']
        best_time_alg = dataset_data.loc[dataset_data['Time'].idxmin(), 'Algorithm']
        best_memory_alg = dataset_data.loc[dataset_data['Memory_MB'].idxmin(), 'Algorithm']
        
        dataset_recommendations[dataset] = {
            'Best_Performance': best_ari_alg,
            'Fastest': best_time_alg,
            'Most_Memory_Efficient': best_memory_alg
        }
    
    for dataset, recommendations in dataset_recommendations.items():
        print(f"\n{dataset}:")
        print(f"  Best Performance: {recommendations['Best_Performance']}")
        print(f"  Fastest: {recommendations['Fastest']}")
        print(f"  Most Memory Efficient: {recommendations['Most_Memory_Efficient']}")
    
    # Export KB-based memory results
    results_df['Memory_KB'] = results_df['Memory_MB'] * 1024
    results_df.to_csv('clustering_benchmark_results.csv', index=False)
    efficiency_ranking.to_csv('clustering_efficiency_ranking.csv', index=False)
    scalability_df.to_csv('clustering_scalability_analysis.csv', index=False)
    results_df.to_csv(os.path.join(plot_dir, 'clustering_benchmark_results.csv'), index=False)
    # Write detailed readings and plot references to radial-clustering.md
    with open('radial-clustering.md', 'w') as f:
        f.write('# Radial Clustering Benchmark Report\n\n')
        f.write('## Time and Memory Efficiency (in KB)\n')
        f.write('![Time and Memory Analysis](clustering_results_20250719_200123/plots/time_memory_analysis.png)\n\n')
        f.write('### Detailed Results Table\n')
        f.write('\n')
        f.write(results_df[['Algorithm', 'Dataset', 'Time', 'Memory_KB']].to_markdown(index=False))
        f.write('\n\n')
        f.write('## Performance Results\n')
        f.write('![Performance Results](clustering_results_20250719_200123/plots/performance_results.png)\n\n')
        f.write('## Performance Comparison\n')
        f.write('![Performance Comparison](clustering_results_20250719_200123/plots/performance_comparison.png)\n\n')
        f.write('## Radar Chart\n')
        f.write('![Radar Chart](clustering_results_20250719_200123/plots/radar_chart.png)\n\n')
        f.write('---\n')
        f.write('### For full CSVs, see clustering_benchmark_results.csv, clustering_efficiency_ranking.csv, clustering_scalability_analysis.csv\n')