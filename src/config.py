"""
Configuration constants for the oil type discrimination project.
"""

# Feature columns in the dataset
FEATURE_COLUMNS = [f"en{i}" for i in range(1, 16)]

# Default data path
DATA_PATH = "data/raw/data-923.csv"
