"""
Module Khai Phá Luật Kết Hợp
Khám phá các mẫu và luật trong hành vi học tập của sinh viên
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import List, Tuple


class AssociationRuleMiner:
    """Khai phá luật kết hợp từ dữ liệu hiệu suất học tập"""
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """Khởi tạo AssociationRuleMiner với cấu hình"""
        # Tìm đường dẫn đúng đến file config
        if not Path(config_path).exists():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.min_support = self.config['association']['min_support']
        self.min_confidence = self.config['association']['min_confidence']
        self.min_lift = self.config['association']['min_lift']
    
    def prepare_transactions(self, df: pd.DataFrame, 
                           features: List[str] = None) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu cho khai phá luật kết hợp
        Chuyển đổi đặc trưng sang định dạng nhị phân
        
        Args:
            df: DataFrame đầu vào
            features: Danh sách đặc trưng cần bao gồm
        """
        df_trans = df.copy()
        
        if features is None:
            # Sử dụng các đặc trưng đã phân nhóm
            features = [col for col in df_trans.columns if '_bin' in col or col == 'pass']
        
        # Chuyển đổi các nhóm phân loại thành cột nhị phân
        df_binary = pd.DataFrame()
        
        for col in features:
            if col in df_trans.columns:
                if df_trans[col].dtype == 'object' or df_trans[col].dtype.name == 'category':
                    # Mã hóa one-hot cho đặc trưng phân loại
                    dummies = pd.get_dummies(df_trans[col], prefix=col)
                    df_binary = pd.concat([df_binary, dummies], axis=1)
                else:
                    # Giữ nguyên đặc trưng nhị phân
                    df_binary[col] = df_trans[col]
        
        return df_binary
    
    def mine_frequent_itemsets(self, df_binary: pd.DataFrame) -> pd.DataFrame:
        """
        Khai phá tập phổ biến sử dụng thuật toán Apriori
        
        Args:
            df_binary: DataFrame giao dịch nhị phân
            
        Returns:
            DataFrame các tập phổ biến
        """
        print(f"Đang khai phá tập phổ biến với min_support={self.min_support}...")
        
        frequent_itemsets = apriori(df_binary, 
                                   min_support=self.min_support, 
                                   use_colnames=True)
        
        print(f"Tìm thấy {len(frequent_itemsets)} tập phổ biến")
        return frequent_itemsets
    
    def generate_rules(self, frequent_itemsets: pd.DataFrame, 
                      metric: str = "confidence") -> pd.DataFrame:
        """
        Tạo luật kết hợp từ tập phổ biến
        
        Args:
            frequent_itemsets: DataFrame các tập phổ biến
            metric: Độ đo sử dụng để tạo luật
            
        Returns:
            DataFrame các luật kết hợp
        """
        print(f"Đang tạo luật với min_confidence={self.min_confidence}...")
        
        rules = association_rules(frequent_itemsets, 
                                 metric=metric,
                                 min_threshold=self.min_confidence)
        
        # Lọc theo lift
        rules = rules[rules['lift'] >= self.min_lift]
        
        # Sắp xếp theo lift
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"Đã tạo {len(rules)} luật")
        return rules
    
    def filter_rules_by_consequent(self, rules: pd.DataFrame, 
                                  consequent: str) -> pd.DataFrame:
        """
        Lọc luật theo kết quả (ví dụ: 'pass' hoặc 'fail')
        
        Args:
            rules: DataFrame các luật kết hợp
            consequent: Kết quả mục tiêu cần lọc
        """
        filtered_rules = rules[rules['consequents'].apply(
            lambda x: consequent in str(x)
        )]
        
        print(f"Tìm thấy {len(filtered_rules)} luật với kết quả '{consequent}'")
        return filtered_rules
    
    def interpret_rules(self, rules: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Diễn giải và định dạng các luật hàng đầu để báo cáo
        
        Args:
            rules: DataFrame các luật kết hợp
            top_n: Số lượng luật hàng đầu cần trả về
        """
        top_rules = rules.head(top_n).copy()
        
        # Chuyển frozensets thành chuỗi dễ đọc
        top_rules['antecedents_str'] = top_rules['antecedents'].apply(
            lambda x: ', '.join(list(x))
        )
        top_rules['consequents_str'] = top_rules['consequents'].apply(
            lambda x: ', '.join(list(x))
        )
        
        # Chọn các cột liên quan
        result = top_rules[['antecedents_str', 'consequents_str', 
                          'support', 'confidence', 'lift']]
        
        result.columns = ['Tiền đề (NẾU)', 'Kết quả (THÌ)', 
                         'Độ hỗ trợ', 'Độ tin cậy', 'Lift']
        
        return result
    
    def mine_and_analyze(self, df: pd.DataFrame, 
                        target_consequent: str = 'pass') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pipeline hoàn chỉnh: chuẩn bị dữ liệu, khai phá luật và phân tích
        
        Args:
            df: DataFrame đầu vào
            target_consequent: Kết quả mục tiêu cần phân tích
            
        Returns:
            Tuple gồm (tất_cả_luật, luật_mục_tiêu)
        """
        print("Bắt đầu pipeline khai phá luật kết hợp...")
        
        # Chuẩn bị giao dịch
        df_binary = self.prepare_transactions(df)
        print(f"Đã chuẩn bị {len(df_binary.columns)} đặc trưng nhị phân")
        
        # Khai phá tập phổ biến
        frequent_itemsets = self.mine_frequent_itemsets(df_binary)
        
        # Tạo luật
        all_rules = self.generate_rules(frequent_itemsets)
        
        # Lọc luật theo mục tiêu
        target_rules = self.filter_rules_by_consequent(all_rules, target_consequent)
        
        print("Hoàn thành khai phá luật kết hợp")
        return all_rules, target_rules
    
    def get_actionable_insights(self, rules: pd.DataFrame, top_n: int = 5) -> List[str]:
        """
        Trích xuất các insight có thể hành động từ luật
        
        Args:
            rules: DataFrame các luật kết hợp
            top_n: Số lượng insight cần tạo
        """
        insights = []
        
        top_rules = rules.head(top_n)
        
        for idx, rule in top_rules.iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            confidence = rule['confidence']
            lift = rule['lift']
            
            insight = (f"Sinh viên có {antecedents} có {confidence*100:.1f}% "
                      f"xác suất {consequents} (lift: {lift:.2f})")
            insights.append(insight)
        
        return insights
