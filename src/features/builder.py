"""
Module Xây Dựng Đặc Trưng
Tạo các đặc trưng mới từ dữ liệu hiện có
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import List


class FeatureBuilder:
    """Xây dựng và thiết kế đặc trưng cho dự đoán hiệu suất học tập"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """Khởi tạo FeatureBuilder với cấu hình"""
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            # Thử tìm từ thư mục gốc dự án
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def create_parent_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo đặc trưng trình độ học vấn trung bình của bố mẹ"""
        df = df.copy()
        df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
        df['parent_edu_max'] = df[['Medu', 'Fedu']].max(axis=1)
        df['parent_edu_min'] = df[['Medu', 'Fedu']].min(axis=1)
        return df
    
    def create_alcohol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo đặc trưng tiêu thụ rượu"""
        df = df.copy()
        df['alcohol_total'] = df['Dalc'] + df['Walc']
        df['alcohol_avg'] = (df['Dalc'] + df['Walc']) / 2
        df['alcohol_weekend_ratio'] = df['Walc'] / (df['Dalc'] + 1)  # Avoid division by zero
        df['high_alcohol'] = (df['alcohol_total'] > 6).astype(int)
        return df
    
    def create_support_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo đặc trưng hỗ trợ học tập"""
        df = df.copy()
        
        # Chuyển yes/no thành 1/0 nếu chưa được chuyển
        support_cols = ['schoolsup', 'famsup', 'paid']
        for col in support_cols:
            if df[col].dtype == 'object':
                df[col] = (df[col] == 'yes').astype(int)
        
        df['support_total'] = df['schoolsup'] + df['famsup'] + df['paid']
        df['has_any_support'] = (df['support_total'] > 0).astype(int)
        return df
    
    def create_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo đặc trưng hành vi xã hội"""
        df = df.copy()
        df['social_score'] = df['goout'] + df['freetime']
        df['social_high'] = (df['goout'] >= 4).astype(int)
        df['has_romantic'] = (df['romantic'] == 'yes').astype(int) if df['romantic'].dtype == 'object' else df['romantic']
        return df
    
    def create_study_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo đặc trưng liên quan đến học tập"""
        df = df.copy()
        df['study_time_low'] = (df['studytime'] <= 2).astype(int)
        df['has_failures'] = (df['failures'] > 0).astype(int)
        df['high_absences'] = (df['absences'] > df['absences'].median()).astype(int)
        df['risk_score'] = df['failures'] * 2 + df['high_absences'] + df['study_time_low']
        return df
    
    def create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo đặc trưng liên quan đến gia đình"""
        df = df.copy()
        df['family_quality'] = df['famrel']
        df['parents_together'] = (df['Pstatus'] == 'T').astype(int) if df['Pstatus'].dtype == 'object' else df['Pstatus']
        df['large_family'] = (df['famsize'] == 'GT3').astype(int) if df['famsize'].dtype == 'object' else df['famsize']
        return df
    
    def create_behavioral_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo phiên bản phân nhóm của đặc trưng hành vi cho luật kết hợp
        """
        df = df.copy()
        
        # Phân nhóm thời gian học
        df['studytime_bin'] = pd.cut(df['studytime'], bins=[0, 2, 3, 5], 
                                     labels=['hoc_it', 'hoc_trung_binh', 'hoc_nhieu'])
        
        # Phân nhóm vắng mặt
        df['absences_bin'] = pd.cut(df['absences'], bins=[-1, 0, 5, 15, 100], 
                                    labels=['khong_vang', 'vang_it', 'vang_trung_binh', 'vang_nhieu'])
        
        # Phân nhóm đi chơi
        df['goout_bin'] = pd.cut(df['goout'], bins=[0, 2, 3, 5], 
                                labels=['di_choi_it', 'di_choi_trung_binh', 'di_choi_nhieu'])
        
        # Phân nhóm rượu
        if 'alcohol_total' in df.columns:
            df['alcohol_bin'] = pd.cut(df['alcohol_total'], bins=[0, 2, 4, 10], 
                                      labels=['ruou_it', 'ruou_trung_binh', 'ruou_nhieu'])
        
        # Phân nhóm trượt môn
        df['failures_bin'] = pd.cut(df['failures'], bins=[-1, 0, 1, 5], 
                                   labels=['khong_truot', 'truot_mot_lan', 'truot_nhieu_lan'])
        
        # Phân nhóm điểm (cho G3)
        df['grade_bin'] = pd.cut(df['G3'], bins=[-1, 10, 14, 20], 
                                labels=['truot', 'do', 'gioi'])
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo đặc trưng tương tác"""
        df = df.copy()
        
        # Thời gian học × hỗ trợ
        if 'support_total' in df.columns:
            df['study_support_interaction'] = df['studytime'] * df['support_total']
        
        # Học vấn bố mẹ × thời gian học
        if 'parent_edu_avg' in df.columns:
            df['edu_study_interaction'] = df['parent_edu_avg'] * df['studytime']
        
        # Trượt môn × vắng mặt
        df['failure_absence_interaction'] = df['failures'] * df['absences']
        
        return df
    
    def drop_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ đặc trưng có thể gây rò rỉ dữ liệu (G1, G2)
        """
        df = df.copy()
        leakage_cols = ['G1', 'G2']
        
        cols_to_drop = [col for col in leakage_cols if col in df.columns]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Đã loại bỏ đặc trưng rò rỉ: {cols_to_drop}")
        
        return df
    
    def build_all_features(self, df: pd.DataFrame, drop_leakage: bool = False) -> pd.DataFrame:
        """
        Xây dựng tất cả đặc trưng thiết kế
        
        Args:
            df: DataFrame đầu vào
            drop_leakage: Có loại bỏ đặc trưng G1, G2 không
        """
        print("Đang xây dựng đặc trưng thiết kế...")
        
        df = self.create_parent_education(df)
        df = self.create_alcohol_features(df)
        df = self.create_support_features(df)
        df = self.create_social_features(df)
        df = self.create_study_features(df)
        df = self.create_family_features(df)
        df = self.create_behavioral_bins(df)
        df = self.create_interaction_features(df)
        
        if drop_leakage:
            df = self.drop_leakage_features(df)
        
        print(f"Hoàn thành thiết kế đặc trưng. Tổng số đặc trưng: {len(df.columns)}")
        return df
    
    def get_feature_groups(self) -> dict:
        """Trả về dictionary các nhóm đặc trưng để phân tích"""
        return {
            'demographic': ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus'],
            'family': ['Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'famrel', 
                      'parent_edu_avg', 'parent_edu_max', 'parents_together'],
            'academic': ['studytime', 'failures', 'schoolsup', 'paid', 'higher',
                        'study_time_low', 'has_failures', 'risk_score'],
            'social': ['goout', 'romantic', 'freetime', 'activities', 
                      'social_score', 'social_high'],
            'health': ['health', 'absences', 'Dalc', 'Walc', 'alcohol_total', 
                      'high_alcohol', 'high_absences'],
            'support': ['schoolsup', 'famsup', 'paid', 'support_total', 'has_any_support']
        }
