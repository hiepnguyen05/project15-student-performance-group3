"""
Module Phân Cụm
Thực hiện phân tích phân cụm trên dữ liệu sinh viên
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple


class ClusteringAnalyzer:
    """Thực hiện phân tích phân cụm trên dữ liệu hiệu suất học tập"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """Khởi tạo ClusteringAnalyzer với cấu hình"""
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.random_seed = self.config['random_seed']
        self.scaler = StandardScaler()
    
    def prepare_data(self, df: pd.DataFrame, features: List[str] = None) -> np.ndarray:
        """
        Chuẩn bị và chuẩn hóa dữ liệu cho phân cụm
        
        Args:
            df: DataFrame đầu vào
            features: Danh sách đặc trưng cần sử dụng
        """
        if features is None:
            # Chỉ sử dụng đặc trưng số
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            # Loại trừ cột target và điểm
            exclude = ['pass', 'G1', 'G2', 'G3']
            features = [f for f in features if f not in exclude]
        
        X = df[features].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Đã chuẩn bị {X_scaled.shape[1]} đặc trưng cho phân cụm")
        return X_scaled, features
    
    def kmeans_clustering(self, X: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """
        Thực hiện phân cụm K-Means
        
        Args:
            X: Ma trận đặc trưng
            n_clusters: Số cụm
        """
        kmeans = KMeans(n_clusters=n_clusters, 
                       random_state=self.random_seed,
                       n_init=10)
        labels = kmeans.fit_predict(X)
        
        return labels, kmeans
    
    def hierarchical_clustering(self, X: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """
        Thực hiện phân cụm phân cấp
        
        Args:
            X: Ma trận đặc trưng
            n_clusters: Số cụm
        """
        linkage = self.config['clustering']['hierarchical']['linkage']
        
        hc = AgglomerativeClustering(n_clusters=n_clusters, 
                                     linkage=linkage)
        labels = hc.fit_predict(X)
        
        return labels, hc
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Đánh giá chất lượng phân cụm
        
        Args:
            X: Ma trận đặc trưng
            labels: Nhãn cụm
        """
        # Lọc các điểm nhiễu (-1) cho các độ đo
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        if len(set(labels_filtered)) < 2:
            return {
                'silhouette': 0.0,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0.0
            }
        
        metrics = {
            'silhouette': silhouette_score(X_filtered, labels_filtered),
            'davies_bouldin': davies_bouldin_score(X_filtered, labels_filtered),
            'calinski_harabasz': calinski_harabasz_score(X_filtered, labels_filtered)
        }
        
        return metrics
    
    def find_optimal_k(self, X: np.ndarray, k_range: List[int] = None) -> pd.DataFrame:
        """
        Tìm số cụm tối ưu sử dụng phương pháp elbow
        
        Args:
            X: Ma trận đặc trưng
            k_range: Phạm vi giá trị k cần kiểm tra
        """
        if k_range is None:
            k_range = self.config['clustering']['kmeans']['n_clusters_range']
        
        results = []
        
        for k in k_range:
            labels, model = self.kmeans_clustering(X, n_clusters=k)
            metrics = self.evaluate_clustering(X, labels)
            
            results.append({
                'k': k,  # Đổi từ 'n_clusters' thành 'k' để dễ dùng
                'inertia': model.inertia_,
                'silhouette': metrics['silhouette'],
                'davies_bouldin': metrics['davies_bouldin'],
                'calinski_harabasz': metrics['calinski_harabasz']
            })
        
        return pd.DataFrame(results)
    
    def profile_clusters(self, df: pd.DataFrame, labels: np.ndarray, 
                        features: List[str]) -> pd.DataFrame:
        """
        Phân tích cụm bằng cách tính toán thống kê cho mỗi cụm
        
        Args:
            df: DataFrame gốc
            labels: Nhãn cụm
            features: Đặc trưng được sử dụng cho phân cụm
        """
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        # Tính toán thống kê cho mỗi cụm
        profiles = []
        
        for cluster_id in sorted(df_with_clusters['cluster'].unique()):
            if cluster_id == -1:  # Bỏ qua điểm nhiễu
                continue
            
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            profile = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100
            }
            
            # Thêm giá trị trung bình cho các đặc trưng chính
            for feature in features[:10]:  # Giới hạn 10 đặc trưng hàng đầu
                if feature in cluster_data.columns:
                    profile[f'{feature}_mean'] = cluster_data[feature].mean()
            
            # Thêm tỷ lệ đỗ nếu có
            if 'pass' in cluster_data.columns:
                profile['pass_rate'] = cluster_data['pass'].mean() * 100
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def compare_methods(self, X: np.ndarray, n_clusters: int = 3) -> pd.DataFrame:
        """
        So sánh các phương pháp phân cụm khác nhau
        
        Args:
            X: Ma trận đặc trưng
            n_clusters: Số cụm
        """
        results = []
        
        # K-Means
        labels_km, _ = self.kmeans_clustering(X, n_clusters)
        metrics_km = self.evaluate_clustering(X, labels_km)
        results.append({
            'method': 'K-Means',
            'n_clusters': n_clusters,
            **metrics_km
        })
        
        # Phân cấp
        labels_hc, _ = self.hierarchical_clustering(X, n_clusters)
        metrics_hc = self.evaluate_clustering(X, labels_hc)
        results.append({
            'method': 'Hierarchical',
            'n_clusters': n_clusters,
            **metrics_hc
        })
        
        return pd.DataFrame(results)
    
    def plot_clusters_pca(self, X_scaled: np.ndarray, labels: np.ndarray, title: str = "K-Means Clustering (PCA 2D)"):
        """
        Vẽ biểu đồ phân tán 2D các cụm sử dụng thuật toán PCA (Principal Component Analysis)
        
        Args:
            X_scaled: Ma trận đặc trưng đã chuẩn hóa
            labels: Nhãn cụm phân bổ cho các điểm dữ liệu
            title: Tiêu đề biểu đồ
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        
        # Giảm chiều xuống 2 thành phần chính
        pca = PCA(n_components=2, random_state=self.random_seed)
        X_pca = pca.fit_transform(X_scaled)
        
        # Lấy tỷ lệ phương sai giải thích được
        explained_variance = pca.explained_variance_ratio_ * 100
        
        plt.figure(figsize=(10, 8))
        
        # Tạo scatter plot
        palette = sns.color_palette("viridis", len(np.unique(labels)))
        sns.scatterplot(
            x=X_pca[:, 0], 
            y=X_pca[:, 1],
            hue=labels,
            palette=palette,
            s=80,
            alpha=0.7,
            edgecolor='k'
        )
        
        plt.title(title, fontsize=15, pad=15)
        plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1f}%)", fontsize=12)
        plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1f}%)", fontsize=12)
        
        # Di chuyển legend ra ngoài biểu đồ
        plt.legend(title='Cụm (Cluster)', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_cluster_profiles(self, profile: pd.DataFrame, features_to_plot: List[str] = None):
        """
        Trực quan hóa profile của các cụm bằng Bar chart
        
        Args:
            profile: DataFrame chứa cấu hình cụm từ phương thức profile_clusters()
            features_to_plot: Danh sách các đặc trưng cần biểu diễn (nếu None, sẽ tự chọn các cột _mean)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Nếu không cung cấp các đặc trưng cần plot, lấy các cột chứa '_mean' và 'pass_rate'
        if features_to_plot is None:
            features_to_plot = [col for col in profile.columns if col.endswith('_mean') or col == 'pass_rate'][:6]
            
        n_features = len(features_to_plot)
        clusters = profile['cluster'].values
        n_clusters = len(clusters)
        
        # Tạo lưới subplot
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        palette = sns.color_palette("viridis", n_clusters)
        
        for i, feature in enumerate(features_to_plot):
            if feature in profile.columns:
                sns.barplot(
                    x='cluster', 
                    y=feature, 
                    data=profile, 
                    ax=axes[i],
                    palette=palette
                )
                
                # Format tiêu đề đặc trưng cho đẹp
                clean_name = feature.replace('_mean', '').replace('_', ' ').title()
                axes[i].set_title(f'Mean {clean_name}', fontsize=12)
                axes[i].set_xlabel('Cụm (Cluster)')
                axes[i].set_ylabel('Giá trị trung bình')
                
                # Thêm nhãn giá trị lên trên các cột
                for container in axes[i].containers:
                    axes[i].bar_label(container, fmt='%.2f', padding=3)
                    
        # Ẩn các subplot không dùng đến
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout(pad=3.0)
        plt.show()


    def get_cluster_recommendations(self, profile: pd.DataFrame) -> List[str]:
        """
        Tạo khuyến nghị cho mỗi cụm
        
        Args:
            profile: DataFrame phân tích cụm
        """
        recommendations = []
        
        for _, cluster in profile.iterrows():
            cluster_id = cluster['cluster']
            pass_rate = cluster.get('pass_rate', 0)
            
            if pass_rate < 50:
                rec = f"Cụm {cluster_id}: Nhóm rủi ro cao (tỷ lệ đỗ: {pass_rate:.1f}%). Cần hỗ trợ tích cực."
            elif pass_rate < 75:
                rec = f"Cụm {cluster_id}: Nhóm rủi ro trung bình (tỷ lệ đỗ: {pass_rate:.1f}%). Cần can thiệp có mục tiêu."
            else:
                rec = f"Cụm {cluster_id}: Nhóm rủi ro thấp (tỷ lệ đỗ: {pass_rate:.1f}%). Duy trì hỗ trợ hiện tại."
            
            recommendations.append(rec)
        
        return recommendations
