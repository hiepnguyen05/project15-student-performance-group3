"""
Script Pipeline Chính
Chạy toàn bộ pipeline khai phá dữ liệu
"""

import sys
from pathlib import Path

# Thêm src vào path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import AssociationRuleMiner
from src.mining.clustering import ClusteringAnalyzer
from src.models.supervised import SupervisedModels
from src.models.semi_supervised import SemiSupervisedModels
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.report import ReportGenerator
from src.evaluation.visualization.plots import PlotGenerator

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def main():
    """Chạy pipeline hoàn chỉnh"""
    
    print("="*60)
    print("PIPELINE KHAI PHÁ DỮ LIỆU HIỆU SUẤT HỌC TẬP")
    print("="*60)
    
    # 1. Tải Dữ Liệu
    print("\n[1/8] Đang tải dữ liệu...")
    loader = DataLoader()
    # Sử dụng cả 2 bộ dữ liệu (Toán và Tiếng Bồ Đào Nha) bằng cách nối tiếp
    df_combined = loader.load_combined_data(merge=False)
    print(f"Tổng số mẫu từ cả 2 môn: {len(df_combined)}")
    print(f"Phân bố theo môn học:")
    print(df_combined['course'].value_counts())
    
    # 2. Làm Sạch và Tiền Xử Lý
    print("\n[2/8] Đang làm sạch và tiền xử lý...")
    cleaner = DataCleaner()
    df_clean = cleaner.preprocess_pipeline(df_combined, fit=True)
    
    # 3. Thiết Kế Đặc Trưng
    print("\n[3/8] Đang xây dựng đặc trưng...")
    builder = FeatureBuilder()
    df_features = builder.build_all_features(df_clean, drop_leakage=False)
    
    # Lưu dữ liệu đã xử lý
    df_features.to_csv('data/processed/student_processed.csv', index=False)
    print("Dữ liệu đã xử lý được lưu vào data/processed/student_processed.csv")
    
    # 4. Khai Phá Luật Kết Hợp
    print("\n[4/8] Đang khai phá luật kết hợp...")
    miner = AssociationRuleMiner()
    all_rules, pass_rules = miner.mine_and_analyze(df_features, target_consequent='pass')
    
    # 5. Phân Tích Phân Cụm
    print("\n[5/8] Đang thực hiện phân tích phân cụm...")
    clusterer = ClusteringAnalyzer()
    X_scaled, features = clusterer.prepare_data(df_features)
    
    # Tìm K tối ưu
    elbow_data = clusterer.find_optimal_k(X_scaled)
    
    # Thực hiện phân cụm với K tối ưu (ví dụ: 4)
    labels, model = clusterer.kmeans_clustering(X_scaled, n_clusters=4)
    profile = clusterer.profile_clusters(df_features, labels, features)
    
    # 6. Học Có Giám Sát
    print("\n[6/8] Đang huấn luyện các mô hình có giám sát...")
    supervised = SupervisedModels()
    
    # Chuẩn bị dữ liệu cho mô hình hóa (loại bỏ đặc trưng rò rỉ)
    df_modeling = builder.drop_leakage_features(df_features)
    X_train, X_test, y_train, y_test = supervised.split_data(df_modeling)
    
    # Xử lý mất cân bằng
    X_train_balanced, y_train_balanced = supervised.handle_imbalance(X_train, y_train)
    
    # Huấn luyện mô hình
    models = supervised.train_all_models(X_train_balanced, y_train_balanced, 
                                        tune_hyperparams=False)
    
    # 7. Đánh Giá Mô Hình
    print("\n[7/8] Đang đánh giá mô hình...")
    metrics_calc = MetricsCalculator()
    
    results = {}
    for model_name in models.keys():
        y_pred = supervised.predict(model_name, X_test)
        y_pred_proba = supervised.predict_proba(model_name, X_test)
        
        metrics = metrics_calc.calculate_classification_metrics(
            y_test, y_pred, y_pred_proba
        )
        results[model_name] = metrics
        
        print(metrics_calc.get_metrics_summary(y_test, y_pred, y_pred_proba, model_name))
    
    # 8. Tạo Báo Cáo
    print("\n[8/8] Đang tạo báo cáo...")
    reporter = ReportGenerator()
    plotter = PlotGenerator()
    
    # So sánh mô hình
    comparison_df = reporter.generate_model_comparison_table(results)
    plotter.plot_model_comparison(comparison_df)
    
    # Phân tích mô hình tốt nhất
    best_model_name = comparison_df.index[0]
    y_pred_best = supervised.predict(best_model_name, X_test)
    y_pred_proba_best = supervised.predict_proba(best_model_name, X_test)
    
    plotter.plot_confusion_matrix(y_test, y_pred_best)
    plotter.plot_roc_curve(y_test, y_pred_proba_best[:, 1])
    plotter.plot_precision_recall_curve(y_test, y_pred_proba_best[:, 1])
    
    # Biểu đồ phân cụm
    plotter.plot_clustering_elbow(elbow_data)
    
    # Báo cáo tổng kết
    project_info = {
        'Bộ dữ liệu': 'UCI Student Performance (Toán + Tiếng Bồ Đào Nha)',
        'Tổng số mẫu': len(df_features),
        'Số đặc trưng': len(df_features.columns),
        'Tỷ lệ đỗ': f"{df_features['pass'].mean()*100:.2f}%"
    }
    
    best_model_info = {
        'Mô hình tốt nhất': best_model_name,
        'F1 Score': f"{results[best_model_name]['f1']:.4f}",
        'Accuracy': f"{results[best_model_name]['accuracy']:.4f}",
        'ROC-AUC': f"{results[best_model_name]['roc_auc']:.4f}"
    }
    
    reporter.generate_summary_report(project_info, best_model_info)
    
    print("\n" + "="*60)
    print("HOÀN THÀNH PIPELINE THÀNH CÔNG!")
    print("="*60)
    print(f"\nKết quả được lưu tại:")
    print(f"  - Bảng: outputs/tables/")
    print(f"  - Hình ảnh: outputs/figures/")
    print(f"  - Mô hình: outputs/models/")


if __name__ == "__main__":
    main()
