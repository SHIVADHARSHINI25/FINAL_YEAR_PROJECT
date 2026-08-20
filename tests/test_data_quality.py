import pytest
import pandas as pd
import numpy as np
from agents.schemas import Evidence
from agents.data_quality import DataQualityAgent

@pytest.fixture
def agent():
    return DataQualityAgent()

@pytest.fixture
def ref_schema():
    return {
        "columns": {
            "age": "int64",
            "income": "float64",
            "city": "object",
            "label": "int64"
        },
        "ranges": {
            "age": {"min": 0, "max": 120},
            "income": {"min": 0.0}
        },
        "categories": {
            "city": ["New York", "London", "Tokyo"]
        },
        "stats": {
            "income": {"mean": 50000.0, "std": 15000.0}
        },
        "label_column": "label"
    }

def test_empty_dataset(agent):
    df = pd.DataFrame()
    evidences = agent.assess(df)
    assert len(evidences) == 1
    assert "Dataset is empty" in evidences[0].finding
    assert evidences[0].supporting_data["data_quality_score"] == 0.0

def test_healthy_dataset(agent, ref_schema):
    # Generates a healthy dataset
    data = {
        "age": [25, 30, 45, 50, 60, 28, 35, 42, 55, 40],
        "income": [45000.0, 48000.0, 52000.0, 55000.0, 62000.0, 43000.0, 49000.0, 51000.0, 58000.0, 50000.0],
        "city": ["New York", "London", "Tokyo", "London", "New York", "Tokyo", "London", "New York", "Tokyo", "London"],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    evidences = agent.assess(df, ref_schema)
    assert len(evidences) == 1
    assert "Dataset is healthy" in evidences[0].finding
    assert evidences[0].supporting_data["data_quality_score"] == 1.0

def test_missing_values(agent, ref_schema):
    # Create dataset with >5% missing value in age (e.g. 2 out of 10 are NaN)
    data = {
        "age": [25, np.nan, 45, np.nan, 60, 28, 35, 42, 55, 40],
        "income": [45000.0, 48000.0, 52000.0, 55000.0, 62000.0, 43000.0, 49000.0, 51000.0, 58000.0, 50000.0],
        "city": ["New York", "London", "Tokyo", "London", "New York", "Tokyo", "London", "New York", "Tokyo", "London"],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    evidences = agent.assess(df, ref_schema)
    # Check that missing value alert was fired
    missing_ev = next(e for e in evidences if e.supporting_data.get("check_type") == "missing_values")
    assert missing_ev.supporting_data["column"] == "age"
    assert missing_ev.supporting_data["missing_ratio"] == 0.2
    assert "impute missing values in column 'age' using median" in missing_ev.recommended_action.lower()
    assert missing_ev.supporting_data["data_quality_score"] < 1.0

def test_schema_drift_and_violations(agent, ref_schema):
    # age has out of bounds (-5), city has invalid category ("Paris")
    data = {
        "age": [-5, 30, 45, 50, 60, 28, 35, 42, 55, 40],
        "income": [45000.0, 48000.0, 52000.0, 55000.0, 62000.0, 43000.0, 49000.0, 51000.0, 58000.0, 50000.0],
        "city": ["Paris", "London", "Tokyo", "London", "New York", "Tokyo", "London", "New York", "Tokyo", "London"],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    evidences = agent.assess(df, ref_schema)
    
    range_ev = next(e for e in evidences if e.supporting_data.get("check_type") == "value_range_violation")
    assert range_ev.supporting_data["column"] == "age"
    assert range_ev.supporting_data["out_of_bounds_count"] == 1
    
    cat_ev = next(e for e in evidences if e.supporting_data.get("check_type") == "invalid_categories")
    assert cat_ev.supporting_data["column"] == "city"
    assert "Paris" in cat_ev.supporting_data["found_unregistered"]

def test_outliers_and_distribution_shift(agent, ref_schema):
    # Generates a dataset where income has an outlier and sample mean is shifted
    # Expected mean is 50000, std is 15000. Let's make all incomes extremely high (e.g. 100000)
    # We need n > 30 for distribution shift check
    np.random.seed(42)
    age = np.random.randint(20, 60, size=40)
    income = np.random.normal(120000.0, 5000.0, size=40) # Highly shifted mean (120k vs 50k)
    # Let's inject three extreme outliers as well (e.g. > 5% outlier ratio for 40 elements)
    income[0] = 500000.0
    income[1] = 450000.0
    income[2] = 480000.0
    city = ["London"] * 40
    label = [0, 1] * 20
    
    df = pd.DataFrame({"age": age, "income": income, "city": city, "label": label})
    evidences = agent.assess(df, ref_schema)
    
    types = [e.supporting_data.get("check_type") for e in evidences]
    assert "distribution_drift" in types
    assert "outliers" in types
    
    drift_ev = next(e for e in evidences if e.supporting_data.get("check_type") == "distribution_drift")
    assert drift_ev.supporting_data["column"] == "income"
    assert drift_ev.supporting_data["z_score"] > 3.0

def test_duplicates(agent, ref_schema):
    data = {
        "age": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        "income": [50000.0] * 10,
        "city": ["London"] * 10,
        "label": [0] * 10
    }
    df = pd.DataFrame(data)
    evidences = agent.assess(df, ref_schema)
    dup_ev = next(e for e in evidences if e.supporting_data.get("check_type") == "duplicate_records")
    assert dup_ev.supporting_data["duplicate_count"] == 9

def test_label_imbalance_and_noise(agent, ref_schema):
    # Imbalance: only 1 label is 1, remaining 9 are 0
    # Noise: age=30, income=50000, city=London has two entries with different labels (0 and 1)
    data = {
        "age": [30, 30, 45, 50, 60, 28, 35, 42, 55, 40],
        "income": [50000.0, 50000.0, 52000.0, 55000.0, 62000.0, 43000.0, 49000.0, 51000.0, 58000.0, 50000.0],
        "city": ["London", "London", "Tokyo", "London", "New York", "Tokyo", "London", "New York", "Tokyo", "London"],
        "label": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    evidences = agent.assess(df, ref_schema)
    
    types = [e.supporting_data.get("check_type") for e in evidences]
    assert "label_class_imbalance" in types
    assert "label_noise" in types
    
    imb_ev = next(e for e in evidences if e.supporting_data.get("check_type") == "label_class_imbalance")
    assert imb_ev.supporting_data["minority_class_ratio"] == 0.10
    
    noise_ev = next(e for e in evidences if e.supporting_data.get("check_type") == "label_noise")
    assert noise_ev.supporting_data["inconsistent_rows_count"] > 0

def test_zero_iqr_outliers(agent, ref_schema):
    # A column where more than 75% of values are identical (e.g. 0), so IQR = 0.
    # One value is extremely high (e.g. 100.0), which should be caught by Z-score fallback.
    data = {
        "age": [30] * 15,
        "income": [50000.0] * 15,
        "city": ["London"] * 15,
        "label": [0] * 15
    }
    df = pd.DataFrame(data)
    # Inject sparse column where 100.0 is > 3 standard deviations away from the mean
    df["sparse_feature"] = [0.0] * 14 + [100.0]
    evidences = agent.assess(df, ref_schema)
    
    outlier_evs = [e for e in evidences if e.supporting_data.get("check_type") == "outliers" and e.supporting_data.get("column") == "sparse_feature"]
    assert len(outlier_evs) == 1
    assert outlier_evs[0].supporting_data["outlier_count"] == 1
