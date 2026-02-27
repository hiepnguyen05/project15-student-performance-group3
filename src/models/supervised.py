"""
Module Học Có Giám Sát
Huấn luyện và đánh giá các mô hình phân lớp
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from typing import Dict, Tuple, List
import joblib


class SupervisedModels:
    """Huấn luyện và đánh giá các mô hình phân lớp có giám sát"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """Khởi tạo SupervisedModels với cấu hình"""
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.random_seed = self.config['random_seed']
        self.test_size = self.config['split']['test_size']
        self.cv_folds = self.config['evaluation']['cv_folds']
        
        self.models = {}
        self.best_models = {}
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'pass',
                   exclude_cols: List[str] = None) -> Tuple:
        """
        Chia dữ liệu thành tập train và test
        
        Args:
            df: DataFrame đầu vào
            target_col: Tên cột mục tiêu
            exclude_cols: Các cột cần loại trừ khỏi đặc trưng
        """
        if exclude_cols is None:
            exclude_cols = ['pass', 'G3', 'grade_bin']
        
        # Tách đặc trưng và mục tiêu
        X = df.drop(columns=exclude_cols, errors='ignore')
        y = df[target_col]
        
        # Loại bỏ các cột không phải số
        X = X.select_dtypes(include=[np.number])
        
        # Chia dữ liệu
        stratify = y if self.config['split']['stratify'] else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=stratify
        )
        
        print(f"Tập train: {len(X_train)} mẫu")
        print(f"Tập test: {len(X_test)} mẫu")
        print(f"Đặc trưng: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
        """
        Xử lý mất cân bằng lớp sử dụng SMOTE
        
        Args:
            X_train: Đặc trưng huấn luyện
            y_train: Nhãn huấn luyện
        """
        if not self.config['evaluation']['handle_imbalance']:
            return X_train, y_train
        
        print("Áp dụng SMOTE để xử lý mất cân bằng lớp...")
        print(f"Trước SMOTE: {y_train.value_counts().to_dict()}")
        
        smote = SMOTE(random_state=self.random_seed)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Sau SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def get_model(self, model_name: str):
        """Lấy instance mô hình theo tên"""
        models_dict = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_seed
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_seed
            )
        }
        
        return models_dict.get(model_name)
    
    def get_param_grid(self, model_name: str) -> Dict:
        """Lấy lưới siêu tham số cho mô hình"""
        param_grids = {
            'logistic_regression': {
                'C': self.config['classification']['logistic_regression']['C']
            },
            'decision_tree': {
                'max_depth': self.config['classification']['decision_tree']['max_depth'],
                'min_samples_split': self.config['classification']['decision_tree']['min_samples_split']
            },
            'random_forest': {
                'n_estimators': self.config['classification']['random_forest']['n_estimators'],
                'max_depth': self.config['classification']['random_forest']['max_depth']
            }
        }
        
        return param_grids.get(model_name, {})
    
    def train_model(self, model_name: str, X_train, y_train, 
                   tune_hyperparams: bool = True):
        """
        Huấn luyện một mô hình đơn
        
        Args:
            model_name: Tên mô hình
            X_train: Đặc trưng huấn luyện
            y_train: Nhãn huấn luyện
            tune_hyperparams: Có điều chỉnh siêu tham số không
        """
        print(f"\nĐang huấn luyện {model_name}...")
        
        model = self.get_model(model_name)
        
        if tune_hyperparams:
            param_grid = self.get_param_grid(model_name)
            
            if param_grid:
                grid_search = GridSearchCV(
                    model, param_grid,
                    cv=self.cv_folds,
                    scoring='f1',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                print(f"Tham số tốt nhất: {grid_search.best_params_}")
                print(f"F1 score CV tốt nhất: {grid_search.best_score_:.4f}")
            else:
                model.fit(X_train, y_train)
                best_model = model
        else:
            model.fit(X_train, y_train)
            best_model = model
        
        self.best_models[model_name] = best_model
        return best_model
    
    def train_all_models(self, X_train, y_train, tune_hyperparams: bool = True) -> Dict:
        """
        Huấn luyện tất cả mô hình đã cấu hình
        
        Args:
            X_train: Đặc trưng huấn luyện
            y_train: Nhãn huấn luyện
            tune_hyperparams: Có điều chỉnh siêu tham số không
        """
        model_names = self.config['classification']['models']
        
        for model_name in model_names:
            try:
                self.train_model(model_name, X_train, y_train, tune_hyperparams)
            except Exception as e:
                print(f"Lỗi khi huấn luyện {model_name}: {e}")
        
        print(f"\nĐã huấn luyện thành công {len(self.best_models)} mô hình")
        return self.best_models
    
    def predict(self, model_name: str, X) -> np.ndarray:
        """Dự đoán sử dụng mô hình đã huấn luyện"""
        if model_name not in self.best_models:
            raise ValueError(f"Mô hình {model_name} chưa được huấn luyện")
        
        return self.best_models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X) -> np.ndarray:
        """Lấy xác suất dự đoán"""
        if model_name not in self.best_models:
            raise ValueError(f"Mô hình {model_name} chưa được huấn luyện")
        
        return self.best_models[model_name].predict_proba(X)
    
    def save_model(self, model_name: str, filepath: str):
        """Lưu mô hình đã huấn luyện vào đĩa"""
        if model_name not in self.best_models:
            raise ValueError(f"Mô hình {model_name} chưa được huấn luyện")
        
        joblib.dump(self.best_models[model_name], filepath)
        print(f"Mô hình đã lưu vào {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Tải mô hình đã huấn luyện từ đĩa"""
        self.best_models[model_name] = joblib.load(filepath)
        print(f"Mô hình đã tải từ {filepath}")
