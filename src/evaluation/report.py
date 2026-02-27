"""
Module Tạo Báo Cáo
Tạo báo cáo toàn diện và bảng kết quả
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import yaml


class ReportGenerator:
    """Tạo báo cáo và lưu kết quả"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """Khởi tạo ReportGenerator với cấu hình"""
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Đảm bảo đường dẫn output đúng
        project_root = Path(__file__).parent.parent.parent
        self.tables_dir = project_root / self.config['output']['tables_dir']
        self.tables_dir.mkdir(parents=True, exist_ok=True)
    
    def save_table(self, df: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Lưu DataFrame dưới dạng bảng
        
        Args:
            df: DataFrame cần lưu
            filename: Tên file đầu ra
            format: Định dạng đầu ra (csv, excel, latex)
        """
        filepath = self.tables_dir / filename
        
        if format == 'csv':
            df.to_csv(filepath, index=True)
        elif format == 'excel':
            df.to_excel(filepath, index=True)
        elif format == 'latex':
            with open(filepath, 'w') as f:
                f.write(df.to_latex(index=True))
        
        print(f"Bảng đã lưu vào {filepath}")
    
    def generate_model_comparison_table(self, results: Dict[str, Dict],
                                       filename: str = 'model_comparison.csv'):
        """
        Tạo bảng so sánh mô hình
        
        Args:
            results: Dictionary {tên_mô_hình: dict_độ_đo}
            filename: Tên file đầu ra
        """
        df = pd.DataFrame(results).T
        df = df.round(4)
        df = df.sort_values('f1', ascending=False)
        
        # Thêm cột xếp hạng
        df['rank'] = range(1, len(df) + 1)
        
        self.save_table(df, filename)
        return df
    
    def generate_clustering_report(self, profiles: pd.DataFrame,
                                  metrics: pd.DataFrame,
                                  filename: str = 'clustering_report.csv'):
        """
        Tạo báo cáo phân tích phân cụm
        
        Args:
            profiles: DataFrame phân tích cụm
            metrics: DataFrame độ đo phân cụm
            filename: Tên file đầu ra
        """
        # Lưu phân tích
        self.save_table(profiles, f'cluster_profiles_{filename}')
        
        # Lưu độ đo
        self.save_table(metrics, f'cluster_metrics_{filename}')
        
        print("Báo cáo phân cụm đã được tạo")
    
    def generate_association_rules_report(self, rules: pd.DataFrame,
                                         top_n: int = 20,
                                         filename: str = 'association_rules.csv'):
        """
        Tạo báo cáo luật kết hợp
        
        Args:
            rules: DataFrame luật kết hợp
            top_n: Số lượng luật hàng đầu cần lưu
            filename: Tên file đầu ra
        """
        top_rules = rules.head(top_n)
        self.save_table(top_rules, filename)
        
        print(f"Báo cáo luật kết hợp đã được tạo ({top_n} luật)")
    
    def generate_semi_supervised_report(self, comparison_df: pd.DataFrame,
                                       filename: str = 'semi_supervised_comparison.csv'):
        """
        Tạo báo cáo học bán giám sát
        
        Args:
            comparison_df: DataFrame kết quả so sánh
            filename: Tên file đầu ra
        """
        # Lưu bảng gốc
        self.save_table(comparison_df, filename)
        
        # Tính toán cải thiện (nếu có đầy đủ cột cần thiết)
        df_improve = comparison_df.copy()
        
        if {'self_training_f1', 'supervised_f1'}.issubset(df_improve.columns):
            df_improve['self_training_improvement'] = (
                df_improve['self_training_f1'] - df_improve['supervised_f1']
            )
        
        if {'label_spreading_f1', 'supervised_f1'}.issubset(df_improve.columns):
            df_improve['label_spreading_improvement'] = (
                df_improve['label_spreading_f1'] - df_improve['supervised_f1']
            )
        
        self.save_table(df_improve, f'semi_supervised_improvement_{filename}')
        
        print("Báo cáo học bán giám sát đã được tạo")
    
    def generate_feature_importance_report(self, model, feature_names: List[str],
                                          top_n: int = 20,
                                          filename: str = 'feature_importance.csv'):
        """
        Tạo báo cáo tầm quan trọng đặc trưng
        
        Args:
            model: Mô hình đã huấn luyện có thuộc tính feature_importances_
            feature_names: Danh sách tên đặc trưng
            top_n: Số lượng đặc trưng hàng đầu cần lưu
            filename: Tên file đầu ra
        """
        if not hasattr(model, 'feature_importances_'):
            print("Mô hình không có thuộc tính feature_importances_")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.head(top_n)
        
        self.save_table(importance_df, filename)
        
        print(f"Báo cáo tầm quan trọng đặc trưng đã được tạo (top {top_n} đặc trưng)")
        return importance_df
    
    def generate_error_analysis_report(self, error_analysis: Dict,
                                      filename: str = 'error_analysis.csv'):
        """
        Tạo báo cáo phân tích lỗi
        
        Args:
            error_analysis: Dictionary phân tích lỗi
            filename: Tên file đầu ra
        """
        df = pd.DataFrame([error_analysis])
        self.save_table(df, filename)
        
        print("Báo cáo phân tích lỗi đã được tạo")
    
    def generate_summary_report(self, project_info: Dict,
                               best_model_info: Dict,
                               filename: str = 'project_summary.txt'):
        """
        Tạo báo cáo tóm tắt dự án
        
        Args:
            project_info: Dictionary thông tin dự án
            best_model_info: Dictionary thông tin mô hình tốt nhất
            filename: Tên file đầu ra
        """
        filepath = self.tables_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("DỰ ÁN KHAI PHÁ DỮ LIỆU HIỆU SUẤT HỌC TẬP - BÁO CÁO TÓNG TẮT\n")
            f.write("="*60 + "\n\n")
            
            f.write("THÔNG TIN DỰ ÁN\n")
            f.write("-"*60 + "\n")
            for key, value in project_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("THÔNG TIN MÔ HÌNH TỐT NHẤT\n")
            f.write("-"*60 + "\n")
            for key, value in best_model_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"Báo cáo tóm tắt đã lưu vào {filepath}")
    
    def generate_insights_report(self, insights: List[str],
                                filename: str = 'actionable_insights.txt'):
        """
        Tạo báo cáo insight có thể hành động
        
        Args:
            insights: Danh sách các chuỗi insight
            filename: Tên file đầu ra
        """
        filepath = self.tables_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CÁC INSIGHT CÓ THỂ HÀNH ĐỘNG\n")
            f.write("="*60 + "\n\n")
            
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n\n")
        
        print(f"Báo cáo insight đã lưu vào {filepath}")
