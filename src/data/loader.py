"""
Module Tải Dữ Liệu
Xử lý việc tải và kiểm tra dữ liệu hiệu suất học tập của sinh viên
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Tải và kiểm tra dữ liệu hiệu suất học tập của sinh viên"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Khởi tạo DataLoader với cấu hình
        
        Args:
            config_path: Đường dẫn đến file cấu hình YAML
        """
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            # Thử tìm từ thư mục gốc dự án
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Đảm bảo đường dẫn data đúng
        project_root = Path(__file__).parent.parent.parent
        self.raw_dir = project_root / self.config['data']['raw_dir']
        self.math_file = self.config['data']['math_file']
        self.portuguese_file = self.config['data']['portuguese_file']
    
    def load_math_data(self) -> pd.DataFrame:
        """Tải dữ liệu môn Toán"""
        filepath = self.raw_dir / self.math_file
        df = pd.read_csv(filepath, sep=';')
        df['course'] = 'math'
        print(f"Loaded {len(df)} records from {self.math_file}")
        return df
    
    def load_portuguese_data(self) -> pd.DataFrame:
        """Tải dữ liệu môn Tiếng Bồ Đào Nha"""
        filepath = self.raw_dir / self.portuguese_file
        df = pd.read_csv(filepath, sep=';')
        df['course'] = 'portuguese'
        print(f"Loaded {len(df)} records from {self.portuguese_file}")
        return df
    
    def load_combined_data(self, merge: bool = False) -> pd.DataFrame:
        """
        Tải cả hai bộ dữ liệu
        
        Args:
            merge: Nếu True, gộp datasets; nếu False, nối tiếp
            
        Returns:
            DataFrame đã kết hợp
        """
        df_math = self.load_math_data()
        df_por = self.load_portuguese_data()
        
        if merge:
            # Gộp theo thuộc tính sinh viên (loại trừ điểm số)
            merge_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                         'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                         'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
                         'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
                         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
            
            df_combined = pd.merge(df_math, df_por, on=merge_cols, 
                                  how='outer', suffixes=('_math', '_por'))
            print(f"Dữ liệu đã gộp: {len(df_combined)} sinh viên duy nhất")
        else:
            # Nối tiếp đơn giản
            df_combined = pd.concat([df_math, df_por], ignore_index=True)
            print(f"Dữ liệu đã nối: {len(df_combined)} bản ghi tổng cộng")
        
        return df_combined
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Kiểm tra cấu trúc dữ liệu
        
        Args:
            df: DataFrame cần kiểm tra
            
        Returns:
            True nếu cấu trúc hợp lệ
        """
        required_cols = ['school', 'sex', 'age', 'G1', 'G2', 'G3']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Cảnh báo: Thiếu các cột: {missing_cols}")
            return False
        
        print("Kiểm tra cấu trúc thành công")
        return True
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Lấy thông tin cơ bản về dữ liệu
        
        Args:
            df: DataFrame cần phân tích
            
        Returns:
            Dictionary chứa thông tin dữ liệu
        """
        info = {
            'n_records': len(df),
            'n_features': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info
