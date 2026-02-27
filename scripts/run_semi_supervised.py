"""
Script Pipeline Bán Giám Sát
So sánh supervised-only vs semi-supervised (Self-Training, Label Spreading)
"""

import sys
from pathlib import Path

# Thêm src vào path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.models.supervised import SupervisedModels
from src.models.semi_supervised import SemiSupervisedModels
from src.evaluation.report import ReportGenerator


def main():
    """Chạy nhánh thực nghiệm bán giám sát"""
    print("=" * 60)
    print("PIPELINE BÁN GIÁM SÁT - THIẾU NHÃN")
    print("=" * 60)

    # 1. Tải dữ liệu
    print("\n[1/5] Đang tải dữ liệu...")
    loader = DataLoader()
    df_combined = loader.load_combined_data(merge=False)

    # 2. Tiền xử lý
    print("\n[2/5] Đang làm sạch và tiền xử lý...")
    cleaner = DataCleaner()
    df_clean = cleaner.preprocess_pipeline(df_combined, fit=True)

    # 3. Xây dựng đặc trưng (loại đặc trưng rò rỉ để đúng với kịch bản thực tế)
    print("\n[3/5] Đang xây dựng đặc trưng...")
    builder = FeatureBuilder()
    df_features = builder.build_all_features(df_clean, drop_leakage=False)
    df_modeling = builder.drop_leakage_features(df_features)

    # 4. Chuẩn bị dữ liệu train/test từ SupervisedModels (giữ đồng bộ với pipeline chính)
    print("\n[4/5] Đang chia tập train/test...")
    supervised = SupervisedModels()
    X_train, X_test, y_train, y_test = supervised.split_data(df_modeling)

    # 5. So sánh supervised-only vs semi-supervised
    print("\n[5/5] Đang so sánh supervised vs semi-supervised...")
    semi_supervised = SemiSupervisedModels()
    comparison_df = semi_supervised.compare_with_supervised(X_train, y_train, X_test, y_test)

    reporter = ReportGenerator()
    reporter.generate_semi_supervised_report(comparison_df)

    print("\n" + "=" * 60)
    print("HOÀN THÀNH PIPELINE BÁN GIÁM SÁT!")
    print("=" * 60)
    print("Bảng kết quả được lưu trong thư mục outputs/tables/")


if __name__ == "__main__":
    main()

