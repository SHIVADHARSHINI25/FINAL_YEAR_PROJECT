import numpy as np
import pandas as pd
from datetime import datetime
from agents.base import BaseAgent
from agents.schemas import Evidence

class DataQualityAgent(BaseAgent):
    """
    Analyzes datasets for missing values, schema conformity, distribution shift,
    outliers, duplicate records, and label consistency. Calculates a global quality score.
    """
    def __init__(self, name: str = "DataQualityAgent"):
        super().__init__(name)

    def assess(
        self,
        dataset: pd.DataFrame,
        reference_schema: dict | None = None
    ) -> list[Evidence]:
        """
        Run data quality checks and return Evidence for each issue found.
        """
        if dataset.empty:
            return [
                Evidence(
                    agent_name=self.name,
                    finding="Dataset is empty.",
                    confidence_score=1.0,
                    supporting_data={"data_quality_score": 0.0, "reason": "empty_dataset"},
                    recommended_action="Verify upstream data pipeline output or fetch process.",
                    predicted_impact="Pipeline cannot continue training or evaluating.",
                    estimated_cost="Low compute, ~5 min."
                )
            ]

        evidences = []
        total_rows = len(dataset)
        quality_score = 1.0

        # Extract options from reference_schema
        ref_cols = reference_schema.get("columns", {}) if reference_schema else {}
        ref_ranges = reference_schema.get("ranges", {}) if reference_schema else {}
        ref_categories = reference_schema.get("categories", {}) if reference_schema else {}
        ref_stats = reference_schema.get("stats", {}) if reference_schema else {}
        label_col = reference_schema.get("label_column") if reference_schema else None

        # 1. Check Missing Values
        missing_counts = dataset.isnull().sum()
        for col in dataset.columns:
            missing_count = int(missing_counts[col])
            if missing_count > 0:
                missing_ratio = missing_count / total_rows
                if missing_ratio > 0.05:
                    severity = "critical" if missing_ratio > 0.15 else "warning"
                    # Deduct score
                    quality_score -= 0.25 if severity == "critical" else 0.10

                    # Concrete recommendation
                    if pd.api.types.is_numeric_dtype(dataset[col]):
                        rec = f"Impute missing values in column '{col}' using median ({dataset[col].median()})."
                    else:
                        rec = f"Impute missing values in column '{col}' using mode ('{dataset[col].mode().iloc[0] if not dataset[col].mode().empty else 'Unknown'}')."

                    evidences.append(
                        Evidence(
                            agent_name=self.name,
                            finding=f"High missing values in column '{col}': {missing_ratio:.1%} ({missing_count}/{total_rows} rows).",
                            confidence_score=0.95,
                            supporting_data={
                                "check_type": "missing_values",
                                "column": col,
                                "missing_ratio": missing_ratio,
                                "missing_count": missing_count,
                                "severity": severity
                            },
                            recommended_action=rec,
                            predicted_impact="Downstream models will encounter NaN errors or experience prediction bias.",
                            estimated_cost="Low compute, ~2 min run."
                        )
                    )

        # 2. Check Schema Conformity (Columns, Types, Range, Categories)
        if ref_cols:
            for col, expected_type in ref_cols.items():
                if col not in dataset.columns:
                    quality_score -= 0.25
                    evidences.append(
                        Evidence(
                            agent_name=self.name,
                            finding=f"Schema Drift: Missing expected column '{col}'.",
                            confidence_score=1.0,
                            supporting_data={
                                "check_type": "schema_drift",
                                "column": col,
                                "issue": "missing_column",
                                "expected_type": expected_type
                            },
                            recommended_action=f"Restore column '{col}' by verifying upstream schema transformation or feature store source.",
                            predicted_impact="Downstream features cannot be constructed; model execution will fail.",
                            estimated_cost="Manual debugging required."
                        )
                    )
                else:
                    actual_type = str(dataset[col].dtype)
                    # Broad type comparison (e.g. float64 vs float)
                    def clean_type(t):
                        t = t.lower()
                        if "int" in t: return "integer"
                        if "float" in t or "double" in t: return "float"
                        if "str" in t or "object" in t or "string" in t: return "string"
                        if "bool" in t: return "boolean"
                        return t
                    is_type_compatible = clean_type(expected_type) == clean_type(actual_type)
                        
                    if not is_type_compatible:
                        quality_score -= 0.15
                        evidences.append(
                            Evidence(
                                agent_name=self.name,
                                finding=f"Schema Drift: Column '{col}' type mismatch. Expected '{expected_type}', found '{actual_type}'.",
                                confidence_score=0.95,
                                supporting_data={
                                    "check_type": "schema_drift",
                                    "column": col,
                                    "issue": "type_mismatch",
                                    "expected_type": expected_type,
                                    "actual_type": actual_type
                                },
                                recommended_action=f"Cast column '{col}' to type '{expected_type}' explicitly during pre-processing.",
                                predicted_impact="Operations depending on correct type alignment will throw RuntimeErrors.",
                                estimated_cost="Low compute, ~2 min script edit."
                            )
                        )

            # Range checks for numeric values
            for col, range_limits in ref_ranges.items():
                if col in dataset.columns and pd.api.types.is_numeric_dtype(dataset[col]):
                    min_val = range_limits.get("min")
                    max_val = range_limits.get("max")
                    
                    out_of_bounds_mask = pd.Series(False, index=dataset.index)
                    if min_val is not None:
                        out_of_bounds_mask |= (dataset[col] < min_val)
                    if max_val is not None:
                        out_of_bounds_mask |= (dataset[col] > max_val)
                        
                    out_of_bounds_count = int(out_of_bounds_mask.sum())
                    if out_of_bounds_count > 0:
                        out_ratio = out_of_bounds_count / total_rows
                        quality_score -= 0.05 * out_ratio
                        evidences.append(
                            Evidence(
                                agent_name=self.name,
                                finding=f"Value range violation in column '{col}': {out_of_bounds_count} rows ({out_ratio:.1%}) out of expected range [{min_val or '-inf'}, {max_val or 'inf'}].",
                                confidence_score=0.90,
                                supporting_data={
                                    "check_type": "value_range_violation",
                                    "column": col,
                                    "out_of_bounds_count": out_of_bounds_count,
                                    "out_of_bounds_ratio": out_ratio
                                },
                                recommended_action=f"Clip or drop values in '{col}' to fit boundaries: [{min_val or '-inf'}, {max_val or 'inf'}].",
                                predicted_impact="Out of bounds inputs may trigger unpredictable gradients or scaling errors.",
                                estimated_cost="Low compute, ~2 min."
                            )
                        )

            # Categorical checks
            for col, allowed_categories in ref_categories.items():
                if col in dataset.columns:
                    invalid_mask = ~dataset[col].isin(allowed_categories) & dataset[col].notnull()
                    invalid_count = int(invalid_mask.sum())
                    if invalid_count > 0:
                        invalid_ratio = invalid_count / total_rows
                        quality_score -= 0.10 * invalid_ratio
                        evidences.append(
                            Evidence(
                                agent_name=self.name,
                                finding=f"Invalid categories detected in column '{col}': {invalid_count} rows ({invalid_ratio:.1%}) contain novel categories not in reference list.",
                                confidence_score=0.90,
                                supporting_data={
                                    "check_type": "invalid_categories",
                                    "column": col,
                                    "invalid_count": invalid_count,
                                    "invalid_ratio": invalid_ratio,
                                    "allowed": allowed_categories,
                                    "found_unregistered": list(dataset.loc[invalid_mask, col].unique())[:5]
                                },
                                recommended_action=f"Filter out or group unregistered categories in column '{col}' into an 'Other' category.",
                                predicted_impact="One-hot encoders or model embeddings will throw lookup keys exceptions.",
                                estimated_cost="Low compute, ~3 min."
                            )
                        )

        # 3. Check Outliers & Distribution Shift
        for col in dataset.columns:
            if pd.api.types.is_numeric_dtype(dataset[col]) and total_rows > 5:
                # 3a. Outliers using IQR
                col_data = dataset[col].dropna()
                if len(col_data) > 5:
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    else:
                        # Fallback to Z-score if IQR is 0 (e.g. constant/sparse columns)
                        std_dev = col_data.std()
                        if std_dev > 0:
                            mean_val = col_data.mean()
                            outlier_mask = ((col_data - mean_val).abs() / std_dev) > 3.0
                            lower_bound = mean_val - 3 * std_dev
                            upper_bound = mean_val + 3 * std_dev
                        else:
                            outlier_mask = pd.Series(False, index=col_data.index)
                            lower_bound = q1
                            upper_bound = q3
                    outlier_count = int(outlier_mask.sum())
                    outlier_ratio = outlier_count / total_rows
                    
                    if outlier_ratio > 0.05: # >5% outliers is unusual
                        quality_score -= 0.05
                        evidences.append(
                            Evidence(
                                agent_name=self.name,
                                finding=f"Outliers detected in column '{col}': {outlier_count} rows ({outlier_ratio:.1%}) deviate from IQR limits [{lower_bound:.2f}, {upper_bound:.2f}].",
                                confidence_score=0.85,
                                supporting_data={
                                    "check_type": "outliers",
                                    "column": col,
                                    "outlier_count": outlier_count,
                                    "outlier_ratio": outlier_ratio,
                                    "iqr_limits": [lower_bound, upper_bound]
                                },
                                recommended_action=f"Apply log transformation or cap outliers in column '{col}' to threshold bounds.",
                                predicted_impact="Extreme values may distort model weights during optimization.",
                                estimated_cost="Low compute, ~2 min."
                            )
                        )

                # 3b. Distribution Shift (via statistics)
                if ref_stats and col in ref_stats:
                    mean_ref = ref_stats[col].get("mean")
                    std_ref = ref_stats[col].get("std")
                    
                    if mean_ref is not None and std_ref is not None and std_ref > 0 and len(col_data) > 30:
                        mean_cur = col_data.mean()
                        # Calculate Z-score of the sample mean: Z = (mean_cur - mean_ref) / (std_ref / sqrt(n))
                        se = std_ref / np.sqrt(len(col_data))
                        z_stat = (mean_cur - mean_ref) / se
                        
                        if abs(z_stat) > 3.0: # Highly statistically significant shift
                            quality_score -= 0.15
                            evidences.append(
                                Evidence(
                                    agent_name=self.name,
                                    finding=f"Statistical Distribution Drift detected in column '{col}': Current mean {mean_cur:.2f} vs expected {mean_ref:.2f} (Z-Score: {z_stat:.2f}).",
                                    confidence_score=0.90,
                                    supporting_data={
                                        "check_type": "distribution_drift",
                                        "column": col,
                                        "z_score": z_stat,
                                        "reference_mean": mean_ref,
                                        "current_mean": mean_cur
                                    },
                                    recommended_action=f"Retrain the model on updated dataset containing the new distribution for column '{col}'.",
                                    predicted_impact="Model predictions will degrade due to covariate shift.",
                                    estimated_cost="High compute (requires retraining)."
                                )
                            )

        # 4. Check Duplicate Records
        duplicate_count = int(dataset.duplicated().sum())
        if duplicate_count > 0:
            dup_ratio = duplicate_count / total_rows
            if dup_ratio > 0.01: # Flag if >1% duplicates
                quality_score -= 0.10 * dup_ratio
                evidences.append(
                    Evidence(
                        agent_name=self.name,
                        finding=f"Duplicate records detected: {duplicate_count} rows ({dup_ratio:.1%}) are identical copies.",
                        confidence_score=0.95,
                        supporting_data={
                            "check_type": "duplicate_records",
                            "duplicate_count": duplicate_count,
                            "duplicate_ratio": dup_ratio
                        },
                        recommended_action="Drop duplicate rows from the dataframe before training.",
                        predicted_impact="Duplicates may lead to artificial overfitting and skewed cross-validation scores.",
                        estimated_cost="Low compute, ~2 min run."
                    )
                )

        # 5. Check Label Quality Issues (imbalance, label noise)
        if label_col and label_col in dataset.columns:
            # 5a. Class Imbalance (only for classification/categorical targets)
            label_series = dataset[label_col].dropna()
            if not label_series.empty and (pd.api.types.is_integer_dtype(label_series) or pd.api.types.is_object_dtype(label_series) or pd.api.types.is_categorical_dtype(label_series)):
                value_counts = label_series.value_counts()
                if len(value_counts) >= 2:
                    min_class_ratio = value_counts.min() / len(label_series)
                    if min_class_ratio <= 0.10: # Flag if minority class is <= 10%
                        quality_score -= 0.10
                        evidences.append(
                            Evidence(
                                agent_name=self.name,
                                finding=f"Label Class Imbalance: Minority class representation is only {min_class_ratio:.1%} in target column '{label_col}'.",
                                confidence_score=0.90,
                                supporting_data={
                                    "check_type": "label_class_imbalance",
                                    "column": label_col,
                                    "minority_class_ratio": min_class_ratio,
                                    "class_distributions": value_counts.to_dict()
                                },
                                recommended_action=f"Apply minority class oversampling (e.g. SMOTE) or configure class_weight='balanced' in model configuration.",
                                predicted_impact="Model will favor the majority class, leading to high accuracy but poor recall/F1 on minority labels.",
                                estimated_cost="Medium compute during training."
                            )
                        )

            # 5b. Label Noise (identical features but different labels)
            feature_cols = [c for c in dataset.columns if c != label_col]
            if feature_cols:
                # Group by feature columns and look for varying labels
                # Note: to be fast and handle large dataframes, we count unique values
                grouped = dataset.groupby(feature_cols)[label_col].nunique()
                noisy_groups = grouped[grouped > 1]
                noisy_count = int(noisy_groups.sum()) # Total rows/groups with conflicting labels
                
                if noisy_count > 0:
                    noise_ratio = noisy_count / total_rows
                    if noise_ratio > 0.005: # Flag if >0.5% inconsistent labels
                        quality_score -= 0.15 * noise_ratio
                        evidences.append(
                            Evidence(
                                agent_name=self.name,
                                finding=f"Label Noise / Inconsistency: Found {noisy_count} rows ({noise_ratio:.1%}) with identical features but different labels.",
                                confidence_score=0.85,
                                supporting_data={
                                    "check_type": "label_noise",
                                    "inconsistent_rows_count": noisy_count,
                                    "noise_ratio": noise_ratio
                                },
                                recommended_action=f"Audit or remove rows with inconsistent label entries for matching features.",
                                predicted_impact="Model convergence will be slowed down; performance ceiling is limited due to label ambiguity.",
                                estimated_cost="Medium manual audit effort."
                            )
                        )

        # Clamp quality score to 0.0 - 1.0 range
        quality_score = max(0.0, min(1.0, float(quality_score)))

        # Append global score telemetry as a general information Evidence (or tag supporting_data of all items)
        for ev in evidences:
            ev.supporting_data["data_quality_score"] = quality_score

        # If no issues found, return a healthy status Evidence
        if not evidences:
            evidences.append(
                Evidence(
                    agent_name=self.name,
                    finding="Dataset is healthy. No data quality issues detected.",
                    confidence_score=1.0,
                    supporting_data={
                        "data_quality_score": 1.0,
                        "check_status": "passed"
                    },
                    recommended_action="None. Proceed with training.",
                    predicted_impact="None. Pipeline should execute correctly.",
                    estimated_cost="None."
                )
            )

        return evidences
