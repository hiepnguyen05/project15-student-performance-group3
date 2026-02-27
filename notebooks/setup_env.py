"""
Setup môi trường cho Jupyter Notebooks
Import file này ở đầu mỗi notebook: from setup_env import *
"""

import sys
import os
from pathlib import Path

# Tự động chuyển về thư mục gốc
current_dir = Path.cwd()
if current_dir.name == 'notebooks':
    project_root = current_dir.parent
    os.chdir(project_root)
    print(f"✓ Đã chuyển về thư mục gốc: {project_root}")
else:
    project_root = current_dir
    print(f"✓ Đang ở thư mục gốc: {project_root}")

# Thêm vào Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import các thư viện thường dùng
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')

print("✓ Môi trường đã được thiết lập!")
print(f"✓ Python path: {sys.path[0]}")
