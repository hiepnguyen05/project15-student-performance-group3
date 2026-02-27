"""
Module Học Bán Giám Sát
Xử lý học tập với dữ liệu nhãn hạn chế
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Dict


class SemiSupervisedModels:
    """Huấn luyện và đánh giá các mô hình học bán giám sát"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """Khởi tạo SemiSupervisedModels với cấu hình"""
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.random_seed = self.config['random_seed']
        self.models = {}
    
    def create_labeled_unlabeled_split(self, X_train, y_train, 
                                      labeled_percentage: float = 10) -> Tuple:
        """
        Chia dữ liệu thành labeled và unlabeled
        
        Args:
            X_train: Đặc trưng huấn luyện
            y_train: Nhãn huấn luyện
            labeled_percentage: Phần trăm dữ liệu giữ nhãn
            
        Returns:
            X_labeled, X_unlabeled, y_labeled, y_unlabeled
        """
        n_labeled = int(len(y_train) * labeled_percentage / 100)
        
        # Chọn ngẫu nhiên các chỉ số có nhãn
        np.random.seed(self.random_seed)
        labeled_indices = np.random.choice(len(y_train), n_labeled, replace=False)
        unlabeled_indices = np.array([i for i in range(len(y_train)) if i not in labeled_indices])
        
        # Chia dữ liệu
        X_labeled = X_train.iloc[labeled_indices]
        X_unlabeled = X_train.iloc[unlabeled_indices]
        y_labeled = y_train.iloc[labeled_indices]
        y_unlabeled = y_train.iloc[unlabeled_indices]
        
        print(f"Chia dữ liệu: {len(X_labeled)} có nhãn ({labeled_percentage}%), "
              f"{len(X_unlabeled)} không nhãn ({100-labeled_percentage}%)")
        
        return X_labeled, X_unlabeled, y_labeled, y_unlabeled
    
    def simulate_limited_labels(self, X_train, y_train, 
                               labeled_percentage: float = 10) -> Tuple:
        """
        Mô phỏng kịch bản với dữ liệu nhãn hạn chế
        
        Args:
            X_train: Đặc trưng huấn luyện
            y_train: Nhãn huấn luyện
            labeled_percentage: Phần trăm dữ liệu giữ nhãn
            
        Returns:
            X_train, y_train_partial (với -1 cho unlabeled)
        """
        n_labeled = int(len(y_train) * labeled_percentage / 100)
        
        # Chọn ngẫu nhiên các chỉ số có nhãn
        np.random.seed(self.random_seed)
        labeled_indices = np.random.choice(len(y_train), n_labeled, replace=False)
        
        # Tạo mục tiêu có nhãn một phần
        y_train_partial = np.full(len(y_train), -1)
        y_train_partial[labeled_indices] = y_train.iloc[labeled_indices].values
        
        print(f"Mô phỏng {labeled_percentage}% dữ liệu có nhãn: "
              f"{n_labeled} có nhãn, {len(y_train) - n_labeled} không nhãn")
        
        return X_train, y_train_partial, labeled_indices
    
    def train_self_training(self, X_train, y_train_partial, 
                           base_classifier=None) -> SelfTrainingClassifier:
        """
        Train self-training classifier
        
        Args:
            X_train: Training features
            y_train_partial: Partially labeled target (-1 for unlabeled)
            base_classifier: Base classifier to use
        """
        if base_classifier is None:
            base_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_seed
            )
        
        threshold = self.config['semi_supervised']['self_training']['threshold']
        max_iter = self.config['semi_supervised']['self_training']['max_iter']
        
        print(f"Training Self-Training classifier (threshold={threshold})...")
        
        self_training = SelfTrainingClassifier(
            base_classifier,
            threshold=threshold,
            max_iter=max_iter,
            verbose=False
        )
        
        self_training.fit(X_train, y_train_partial)
        
        # Count how many samples were labeled
        n_labeled = np.sum(self_training.transduction_ != -1)
        print(f"Self-training labeled {n_labeled}/{len(y_train_partial)} samples")
        
        return self_training
    

    
    def compare_with_supervised(self, X_train, y_train, X_test, y_test,
                               labeled_percentages: List[float] = None) -> pd.DataFrame:
        """
        Compare semi-supervised vs supervised learning across different label percentages
        
        Args:
            X_train: Training features
            y_train: Full training labels
            X_test: Test features
            y_test: Test labels
            labeled_percentages: List of label percentages to test
        """
        if labeled_percentages is None:
            labeled_percentages = self.config['semi_supervised']['labeled_percentages']
        
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.semi_supervised import LabelSpreading
        import pandas as pd
        
        results = []
        
        for pct in labeled_percentages:
            print(f"\n--- Testing with {pct}% labeled data ---")
            
            # Ensure numpy array for semi-supervised estimators
            X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            
            # Simulate limited labels (y_partial has -1 for unlabeled)
            _, y_partial, labeled_idx = self.simulate_limited_labels(
                X_train_arr, y_train, pct
            )
            
            # Supervised-only baseline (train only on labeled subset)
            X_labeled = X_train_arr[labeled_idx]
            y_labeled = y_train.iloc[labeled_idx]
            
            supervised_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_seed
            )
            supervised_model.fit(X_labeled, y_labeled)
            
            y_pred_supervised = supervised_model.predict(X_test)
            acc_supervised = accuracy_score(y_test, y_pred_supervised)
            f1_supervised = f1_score(y_test, y_pred_supervised)
            
            # Semi-supervised: Self-Training
            self_training = self.train_self_training(X_train_arr, y_partial)
            y_pred_self = self_training.predict(X_test)
            acc_self = accuracy_score(y_test, y_pred_self)
            f1_self = f1_score(y_test, y_pred_self)
            
            # Semi-supervised: Label Spreading
            label_spread = LabelSpreading(kernel='knn', alpha=0.2)
            label_spread.fit(X_train_arr, y_partial)
            y_pred_ls = label_spread.predict(X_test)
            acc_ls = accuracy_score(y_test, y_pred_ls)
            f1_ls = f1_score(y_test, y_pred_ls)
            
            results.append({
                'labeled_pct': pct,
                'n_labeled': len(labeled_idx),
                'supervised_accuracy': acc_supervised,
                'supervised_f1': f1_supervised,
                'self_training_accuracy': acc_self,
                'self_training_f1': f1_self,
                'label_spreading_accuracy': acc_ls,
                'label_spreading_f1': f1_ls
            })
        
        return pd.DataFrame(results)
    
    def analyze_pseudo_labels(self, self_training_model, X_train, y_train_partial,
                             y_train_true) -> Dict:
        """
        Analyze quality of pseudo-labels generated by self-training
        
        Args:
            self_training_model: Trained self-training model
            X_train: Training features
            y_train_partial: Partially labeled target
            y_train_true: True labels
        """
        # Get transduction (pseudo-labels)
        transduction = self_training_model.transduction_
        
        # Find unlabeled samples
        unlabeled_mask = y_train_partial == -1
        
        # Compare pseudo-labels with true labels
        pseudo_labels = transduction[unlabeled_mask]
        true_labels = y_train_true.iloc[unlabeled_mask].values
        
        # Calculate accuracy of pseudo-labels
        correct = np.sum(pseudo_labels == true_labels)
        total = len(pseudo_labels)
        accuracy = correct / total if total > 0 else 0
        
        analysis = {
            'n_pseudo_labeled': total,
            'n_correct': correct,
            'n_incorrect': total - correct,
            'pseudo_label_accuracy': accuracy,
            'error_rate': 1 - accuracy
        }
        
        print(f"Pseudo-label analysis: {correct}/{total} correct ({accuracy*100:.2f}%)")
        
        return analysis
    
    def get_learning_curve_data(self, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        """
        Tạo dữ liệu đường cong học tập để trực quan hóa
        
        Args:
            X_train: Đặc trưng huấn luyện
            y_train: Nhãn huấn luyện
            X_test: Đặc trưng kiểm tra
            y_test: Nhãn kiểm tra
        """
        return self.compare_with_supervised(X_train, y_train, X_test, y_test)

    def supervised_baseline(self, X_labeled, y_labeled, X_test, y_test):
        """
        Huấn luyện mô hình supervised-only với dữ liệu có nhãn hạn chế
        
        Args:
            X_labeled: Đặc trưng có nhãn
            y_labeled: Nhãn
            X_test: Đặc trưng test
            y_test: Nhãn test
            
        Returns:
            y_pred, metrics
        """
        from sklearn.ensemble import RandomForestClassifier
        from src.evaluation.metrics import MetricsCalculator
        
        # Huấn luyện mô hình
        model = RandomForestClassifier(random_state=self.random_seed, n_estimators=100)
        model.fit(X_labeled, y_labeled)
        
        # Dự đoán
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Tính metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return y_pred, metrics
    
    def self_training_classifier(self, X_labeled, y_labeled, X_unlabeled, X_test, y_test):
        """
        Huấn luyện mô hình self-training
        
        Args:
            X_labeled: Đặc trưng có nhãn
            y_labeled: Nhãn
            X_unlabeled: Đặc trưng không nhãn
            X_test: Đặc trưng test
            y_test: Nhãn test
            
        Returns:
            y_pred, metrics
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.semi_supervised import SelfTrainingClassifier
        from src.evaluation.metrics import MetricsCalculator
        
        # Kết hợp labeled và unlabeled
        import pandas as pd
        X_train = pd.concat([X_labeled, X_unlabeled], ignore_index=True)
        y_train = pd.concat([y_labeled, pd.Series([-1] * len(X_unlabeled))], ignore_index=True)
        
        # Huấn luyện self-training
        base_classifier = RandomForestClassifier(random_state=self.random_seed, n_estimators=100)
        self_training = SelfTrainingClassifier(base_classifier, threshold=0.75, max_iter=10)
        self_training.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = self_training.predict(X_test)
        y_pred_proba = self_training.predict_proba(X_test)
        
        # Tính metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return y_pred, metrics
    

    
    def label_spreading_classifier(self, X_labeled, y_labeled, X_unlabeled, X_test, y_test):
        """
        Huấn luyện mô hình Label Spreading
        
        Args:
            X_labeled: Đặc trưng có nhãn
            y_labeled: Nhãn
            X_unlabeled: Đặc trưng không nhãn
            X_test: Đặc trưng test
            y_test: Nhãn test
            
        Returns:
            y_pred, metrics
        """
        from sklearn.semi_supervised import LabelSpreading
        from src.evaluation.metrics import MetricsCalculator
        import pandas as pd
        
        # Kết hợp labeled và unlabeled
        X_train = pd.concat([X_labeled, X_unlabeled], ignore_index=True)
        y_train = pd.concat([y_labeled, pd.Series([-1] * len(X_unlabeled))], ignore_index=True)
        
        # Huấn luyện Label Spreading
        label_spread = LabelSpreading(kernel='knn', alpha=0.2)
        label_spread.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = label_spread.predict(X_test)
        try:
            y_pred_proba = label_spread.predict_proba(X_test)
        except Exception:
            y_pred_proba = None
            
        # Tính metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return y_pred, metrics

    def label_spreading_classifier(self, X_labeled, y_labeled, X_unlabeled, X_test, y_test):
        """
        Huấn luyện mô hình Label Spreading
        
        Args:
            X_labeled: Đặc trưng có nhãn
            y_labeled: Nhãn
            X_unlabeled: Đặc trưng không nhãn
            X_test: Đặc trưng test
            y_test: Nhãn test
            
        Returns:
            y_pred, metrics
        """
        from sklearn.semi_supervised import LabelSpreading
        from src.evaluation.metrics import MetricsCalculator
        import pandas as pd
        
        # Kết hợp labeled và unlabeled
        X_train = pd.concat([X_labeled, X_unlabeled], ignore_index=True)
        y_train = pd.concat([y_labeled, pd.Series([-1] * len(X_unlabeled))], ignore_index=True)
        
        # Huấn luyện Label Spreading
        label_spread = LabelSpreading(kernel='knn', alpha=0.2)
        label_spread.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = label_spread.predict(X_test)
        try:
            y_pred_proba = label_spread.predict_proba(X_test)
        except Exception:
            y_pred_proba = None
            
        # Tính metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return y_pred, metrics


    def analyze_pseudo_labels(self, X_labeled, y_labeled, X_unlabeled, y_unlabeled):
        """
        Phân tích chất lượng pseudo-labels
        
        Args:
            X_labeled: Đặc trưng có nhãn
            y_labeled: Nhãn
            X_unlabeled: Đặc trưng không nhãn
            y_unlabeled: Nhãn thật (để đánh giá)
            
        Returns:
            Dictionary chứa phân tích
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.semi_supervised import SelfTrainingClassifier
        import pandas as pd
        
        # Kết hợp labeled và unlabeled
        X_train = pd.concat([X_labeled, X_unlabeled], ignore_index=True)
        y_train = pd.concat([y_labeled, pd.Series([-1] * len(X_unlabeled))], ignore_index=True)
        
        # Huấn luyện self-training
        base_classifier = RandomForestClassifier(random_state=self.random_seed, n_estimators=100)
        self_training = SelfTrainingClassifier(base_classifier, threshold=0.75, max_iter=10)
        self_training.fit(X_train, y_train)
        
        # Lấy pseudo-labels
        pseudo_labels = self_training.transduction_[len(y_labeled):]
        
        # So sánh với nhãn thật
        correct = (pseudo_labels == y_unlabeled.values).sum()
        total = len(pseudo_labels)
        
        analysis = {
            'total_pseudo': total,
            'correct_pseudo': correct,
            'incorrect_pseudo': total - correct,
            'pseudo_accuracy': correct / total if total > 0 else 0
        }
        
        return analysis
