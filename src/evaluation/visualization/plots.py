"""
Module Trực Quan Hóa
Tạo biểu đồ và đồ thị để phân tích
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


class PlotGenerator:
    """Tạo trực quan hóa cho dự án khai phá dữ liệu"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """Khởi tạo PlotGenerator với cấu hình"""
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Đảm bảo đường dẫn output đúng
        project_root = Path(__file__).parent.parent.parent.parent
        self.figures_dir = project_root / self.config['output']['figures_dir']
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize = tuple(self.config['visualization']['figure_size'])
        self.dpi = self.config['visualization']['dpi']
        
        # Thiết lập style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
    
    def save_figure(self, filename: str):
        """Lưu hình hiện tại"""
        filepath = self.figures_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"Hình đã lưu vào {filepath}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None,
                             filename: str = 'confusion_matrix.png'):
        """Vẽ ma trận nhầm lẫn"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels or ['Trượt', 'Đỗ'],
                   yticklabels=labels or ['Trượt', 'Đỗ'])
        plt.title('Ma Trận Nhầm Lẫn')
        plt.ylabel('Nhãn Thực')
        plt.xlabel('Nhãn Dự Đoán')
        
        self.save_figure(filename)
    
    def plot_roc_curve(self, y_true, y_pred_proba, 
                      filename: str = 'roc_curve.png'):
        """Vẽ đường cong ROC"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'Đường cong ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Ngẫu nhiên')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tỷ Lệ Dương Tính Giả')
        plt.ylabel('Tỷ Lệ Dương Tính Thực')
        plt.title('Đường Cong ROC (Receiver Operating Characteristic)')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        self.save_figure(filename)
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba,
                                   filename: str = 'pr_curve.png'):
        """Vẽ đường cong Precision-Recall"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_pred_proba)
        
        plt.figure()
        plt.plot(recall, precision, label=f'Đường cong PR (AP = {ap:.3f})')
        plt.xlabel('Recall (Độ Nhạy)')
        plt.ylabel('Precision (Độ Chính Xác)')
        plt.title('Đường Cong Precision-Recall')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        self.save_figure(filename)
    
    def plot_model_comparison(self, results_df: pd.DataFrame,
                            metric: str = 'f1',
                            filename: str = 'model_comparison.png'):
        """Vẽ biểu đồ cột so sánh mô hình"""
        plt.figure(figsize=(12, 6))
        
        results_sorted = results_df.sort_values(metric, ascending=True)
        
        plt.barh(results_sorted.index, results_sorted[metric])
        plt.xlabel(metric.upper())
        plt.ylabel('Mô Hình')
        plt.title(f'So Sánh Mô Hình Theo {metric.upper()}')
        plt.grid(axis='x', alpha=0.3)
        
        self.save_figure(filename)
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 20,
                               filename: str = 'feature_importance.png'):
        """Vẽ tầm quan trọng đặc trưng"""
        plt.figure(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Tầm Quan Trọng')
        plt.ylabel('Đặc Trưng')
        plt.title(f'Top {top_n} Đặc Trưng Quan Trọng Nhất')
        plt.gca().invert_yaxis()
        
        self.save_figure(filename)
    
    def plot_clustering_elbow(self, elbow_data: pd.DataFrame,
                             filename: str = 'clustering_elbow.png'):
        """Vẽ đường cong elbow cho phân cụm"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Biểu đồ Inertia
        axes[0].plot(elbow_data['k'], elbow_data['inertia'], 'bo-')
        axes[0].set_xlabel('Số Cụm')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Phương Pháp Elbow - Inertia')
        axes[0].grid(True)
        
        # Biểu đồ Silhouette
        axes[1].plot(elbow_data['k'], elbow_data['silhouette'], 'ro-')
        axes[1].set_xlabel('Số Cụm')
        axes[1].set_ylabel('Điểm Silhouette')
        axes[1].set_title('Phương Pháp Elbow - Điểm Silhouette')
        axes[1].grid(True)
        
        self.save_figure(filename)
    
    def plot_cluster_distribution(self, df: pd.DataFrame, cluster_col: str = 'cluster',
                                 filename: str = 'cluster_distribution.png'):
        """Vẽ phân bố kích thước cụm"""
        plt.figure()
        
        cluster_counts = df[cluster_col].value_counts().sort_index()
        
        plt.bar(cluster_counts.index, cluster_counts.values)
        plt.xlabel('Cụm')
        plt.ylabel('Số Sinh Viên')
        plt.title('Phân Bố Kích Thước Cụm')
        plt.grid(axis='y', alpha=0.3)
        
        self.save_figure(filename)
    
    def plot_learning_curve(self, learning_curve_df: pd.DataFrame,
                           filename: str = 'learning_curve.png'):
        """Vẽ đường cong học tập cho học bán giám sát"""
        plt.figure(figsize=(12, 6))
        
        x = learning_curve_df['labeled_pct']
        
        plt.plot(x, learning_curve_df['supervised_f1'], 'o-', label='Chỉ Có Giám Sát')
        plt.plot(x, learning_curve_df['self_training_f1'], 's-', label='Self-Training')
        plt.plot(x, learning_curve_df['label_spreading_f1'], '^-', label='Label Spreading')
        
        plt.xlabel('Phần Trăm Dữ Liệu Có Nhãn (%)')
        plt.ylabel('Điểm F1')
        plt.title('Đường Cong Học Tập: Bán Giám Sát vs Có Giám Sát')
        plt.legend()
        plt.grid(True)
        
        self.save_figure(filename)
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, features: list = None,
                                filename: str = 'correlation_heatmap.png'):
        """Vẽ bản đồ nhiệt tương quan"""
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr = df[features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Bản Đồ Nhiệt Tương Quan Đặc Trưng')
        
        self.save_figure(filename)
    
    def plot_distribution(self, df: pd.DataFrame, column: str,
                         filename: str = None):
        """Vẽ phân bố của một cột"""
        if filename is None:
            filename = f'distribution_{column}.png'
        
        plt.figure()
        
        if df[column].dtype in ['int64', 'float64']:
            plt.hist(df[column], bins=30, edgecolor='black')
            plt.xlabel(column)
            plt.ylabel('Tần Suất')
        else:
            df[column].value_counts().plot(kind='bar')
            plt.xlabel(column)
            plt.ylabel('Số Lượng')
            plt.xticks(rotation=45)
        
        plt.title(f'Phân Bố của {column}')
        plt.grid(axis='y', alpha=0.3)
        
        self.save_figure(filename)
    
    def plot_target_distribution(self, df: pd.DataFrame, target_col: str = 'pass',
                                filename: str = 'target_distribution.png'):
        """Vẽ phân bố biến mục tiêu"""
        plt.figure()
        
        counts = df[target_col].value_counts()
        labels = ['Trượt', 'Đỗ'] if target_col == 'pass' else counts.index
        
        plt.pie(counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'Phân Bố Mục Tiêu: {target_col}')
        
        self.save_figure(filename)
