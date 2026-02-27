"""
Module Làm Sạch Dữ Liệu
Xử lý làm sạch, tiền xử lý và mã hóa dữ liệu
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import Tuple, List


class DataCleaner:
    """Làm sạch và tiền xử lý dữ liệu hiệu suất học tập"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """Khởi tạo DataCleaner với cấu hình"""
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            # Thử tìm từ thư mục gốc dự án
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.pass_threshold = self.config['preprocessing']['pass_threshold']
        self.missing_strategy = self.config['preprocessing']['missing_strategy']
        self.scaling_method = self.config['preprocessing']['scaling_method']
        
        self.scaler = None
        self.label_encoders = {}
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo biến mục tiêu nhị phân (đỗ/trượt) từ G3
        
        Args:
            df: DataFrame đầu vào
            
        Returns:
            DataFrame với cột 'pass' được thêm vào
        """
        df = df.copy()
        df['pass'] = (df['G3'] >= self.pass_threshold).astype(int)
        
        pass_count = df['pass'].sum()
        fail_count = len(df) - pass_count
        
        print(f"Phân bố mục tiêu: Đỗ={pass_count} ({pass_count/len(df)*100:.1f}%), "
              f"Trượt={fail_count} ({fail_count/len(df)*100:.1f}%)")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý giá trị thiếu dựa trên chiến lược"""
        df = df.copy()
        
        if df.isnull().sum().sum() == 0:
            print("Không tìm thấy giá trị thiếu")
            return df
        
        print(f"Giá trị thiếu trước xử lý: {df.isnull().sum().sum()}")
        
        if self.missing_strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif self.missing_strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif self.missing_strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif self.missing_strategy == 'drop':
            df = df.dropna()
        
        print(f"Giá trị thiếu sau xử lý: {df.isnull().sum().sum()}")
        return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Phát hiện ngoại lệ sử dụng phương pháp IQR
        
        Args:
            df: DataFrame đầu vào
            columns: Các cột cần kiểm tra ngoại lệ
            
        Returns:
            DataFrame chứa thông tin ngoại lệ
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_info = []
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_info.append({
                    'column': col,
                    'n_outliers': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
        
        return pd.DataFrame(outlier_info)
    
    def encode_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mã hóa đặc trưng nhị phân (yes/no) thành 0/1"""
        df = df.copy()
        
        binary_map = {'yes': 1, 'no': 0, 'F': 0, 'M': 1, 
                     'U': 1, 'R': 0, 'LE3': 0, 'GT3': 1,
                     'T': 1, 'A': 0, 'GP': 0, 'MS': 1}
        
        binary_features = self.config['preprocessing']['encoding']['binary_features']
        
        for col in binary_features:
            if col in df.columns:
                df[col] = df[col].map(binary_map)
        
        print(f"Đã mã hóa {len(binary_features)} đặc trưng nhị phân")
        return df
    
    def encode_nominal_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Mã hóa đặc trưng danh nghĩa sử dụng Label Encoding
        
        Args:
            df: DataFrame đầu vào
            fit: Nếu True, fit encoder mới; nếu False, dùng encoder hiện có
        """
        df = df.copy()
        nominal_features = self.config['preprocessing']['encoding']['nominal_features']
        
        for col in nominal_features:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        print(f"Đã mã hóa {len(nominal_features)} đặc trưng danh nghĩa")
        return df
    
    def scale_features(self, df: pd.DataFrame, columns: List[str] = None, 
                      fit: bool = True) -> pd.DataFrame:
        """
        Chuẩn hóa đặc trưng số
        
        Args:
            df: DataFrame đầu vào
            columns: Các cột cần chuẩn hóa
            fit: Nếu True, fit scaler mới; nếu False, dùng scaler hiện có
        """
        df = df.copy()
        
        if columns is None:
            # Chuẩn hóa tất cả cột số trừ target và điểm
            exclude_cols = ['pass', 'G1', 'G2', 'G3']
            columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col not in exclude_cols]
        
        if fit:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                self.scaler = RobustScaler()
            
            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            if self.scaler is not None:
                df[columns] = self.scaler.transform(df[columns])
        
        print(f"Đã chuẩn hóa {len(columns)} đặc trưng sử dụng {self.scaling_method}")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loại bỏ các dòng trùng lặp"""
        df = df.copy()
        n_before = len(df)
        df = df.drop_duplicates()
        n_after = len(df)
        
        print(f"Đã loại bỏ {n_before - n_after} dòng trùng lặp")
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Pipeline tiền xử lý hoàn chỉnh
        
        Args:
            df: DataFrame đầu vào
            fit: Nếu True, fit các transformer; nếu False, dùng transformer hiện có
        """
        print("Bắt đầu pipeline tiền xử lý...")
        
        # 1. Tạo biến mục tiêu
        df = self.create_target(df)
        
        # 2. Loại bỏ trùng lặp
        df = self.remove_duplicates(df)
        
        # 3. Xử lý giá trị thiếu
        df = self.handle_missing_values(df)
        
        # 4. Mã hóa đặc trưng
        df = self.encode_binary_features(df)
        df = self.encode_nominal_features(df, fit=fit)
        
        # 5. Chuẩn hóa đặc trưng (tùy chọn, có thể làm sau)
        # df = self.scale_features(df, fit=fit)
        
        print("Hoàn thành pipeline tiền xử lý")
        return df
