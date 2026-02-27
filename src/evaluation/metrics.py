"""
Module Tính Toán Độ Đo
Tính toán và báo cáo các độ đo đánh giá
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from typing import Dict, Tuple


class MetricsCalculator:
    """Tính toán các độ đo đánh giá cho mô hình phân lớp"""
    
    def __init__(self):
        """Khởi tạo MetricsCalculator"""
        pass
    
    def calculate_classification_metrics(self, y_true, y_pred, 
                                        y_pred_proba=None) -> Dict[str, float]:
        """
        Tính toán các độ đo phân lớp toàn diện
        
        Args:
            y_true: Nhãn thực
            y_pred: Nhãn dự đoán
            y_pred_proba: Xác suất dự đoán (cho độ đo AUC)
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Thêm độ đo AUC nếu có xác suất
        if y_pred_proba is not None:
            try:
                # Đối với phân lớp nhị phân, sử dụng xác suất của lớp dương
                if y_pred_proba.ndim == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                print(f"Cảnh báo: Không thể tính độ đo AUC: {e}")
                metrics['roc_auc'] = np.nan
                metrics['pr_auc'] = np.nan
        
        return metrics
    
    def get_confusion_matrix(self, y_true, y_pred) -> np.ndarray:
        """Tính ma trận nhầm lẫn"""
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true, y_pred, 
                                 target_names=None) -> str:
        """Lấy báo cáo phân lớp chi tiết"""
        return classification_report(y_true, y_pred, 
                                    target_names=target_names,
                                    zero_division=0)
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        So sánh nhiều mô hình
        
        Args:
            results: Dictionary {tên_mô_hình: dict_độ_đo}
        """
        df = pd.DataFrame(results).T
        df = df.round(4)
        df = df.sort_values('f1', ascending=False)
        
        return df
    
    def calculate_error_analysis(self, y_true, y_pred, X=None, 
                                feature_names=None) -> Dict:
        """
        Perform error analysis
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Feature matrix (optional)
            feature_names: Feature names (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate error types
        tn, fp, fn, tp = cm.ravel()
        
        analysis = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_errors': int(fp + fn),
            'error_rate': (fp + fn) / len(y_true),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        analysis['n_misclassified'] = int(np.sum(misclassified_mask))
        
        if X is not None and feature_names is not None:
            # Analyze features of misclassified samples
            X_misclassified = X[misclassified_mask]
            analysis['misclassified_feature_means'] = X_misclassified.mean(axis=0).tolist()
        
        return analysis
    
    def calculate_cost_sensitive_metrics(self, y_true, y_pred, 
                                        fp_cost: float = 1.0,
                                        fn_cost: float = 2.0) -> Dict:
        """
        Calculate cost-sensitive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            fp_cost: Cost of false positive
            fn_cost: Cost of false negative
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (fp * fp_cost) + (fn * fn_cost)
        
        return {
            'total_cost': total_cost,
            'fp_cost': fp * fp_cost,
            'fn_cost': fn * fn_cost,
            'avg_cost_per_sample': total_cost / len(y_true)
        }
    
    def get_metrics_summary(self, y_true, y_pred, y_pred_proba=None,
                          model_name: str = "Model") -> str:
        """
        Lấy tóm tắt độ đo đã định dạng
        
        Args:
            y_true: Nhãn thực
            y_pred: Nhãn dự đoán
            y_pred_proba: Xác suất dự đoán
            model_name: Tên mô hình
        """
        metrics = self.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        
        summary = f"\n{'='*50}\n"
        summary += f"Độ Đo Hiệu Suất {model_name}\n"
        summary += f"{'='*50}\n"
        
        for metric_name, value in metrics.items():
            if not np.isnan(value):
                summary += f"{metric_name.upper():15s}: {value:.4f}\n"
        
        summary += f"{'='*50}\n"
        
        return summary
