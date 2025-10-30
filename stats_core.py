import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chisquare, t as t_dist, norm

def calculate_sample_size(baseline_mean: float, baseline_std_dev: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Calculate the required sample size per variation for a continuous metric.
    
    :param baseline_mean: The mean of the control group's metric.
    :param baseline_std_dev: The standard deviation of the control group's metric.
    :param mde: Minimum Detectable Effect (absolute). The smallest lift we want to detect.
    :param alpha: Significance level (Type I error rate).
    :param power: Statistical power (1 - Type II error rate).
    :return: Required sample size per variation.
    """
    from statsmodels.stats.power import TTestIndPower
    
    # Cohen's d: Standardized effect size
    effect_size = mde / baseline_std_dev
    
    analysis = TTestIndPower()
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0, # Assuming 50/50 split
        alternative='two-sided'
    )
    
    # solve_power returns n for *one* group
    return int(np.ceil(sample_size))

def run_analysis(df: pd.DataFrame, user_col: str, metric_col: str, covariate_col: str, variation_col: str = 'variation') -> dict:
    """
    Runs a full A/B test analysis with SRM check and CUPED variance reduction.
    (For Continuous Metrics)
    
    :param df: DataFrame with one row per user. Must contain:
               - user_col (e.g., 'user_id')
               - variation_col (e.g., 'A', 'B')
               - metric_col (e.g., 'revenue')
               - covariate_col (e.g., 'revenue_pre_experiment')
    :param user_col: The name of the unique user identifier column. (Used for validation)
    :param metric_col: The name of the metric column to analyze.
    :param covariate_col: The name of the pre-experiment covariate column.
    :param variation_col: The name of the variation assignment column.
    :return: A dictionary of results.
    """
    
    # --- 1. SRM Check ---
    # Check for Sample Ratio Mismatch
    observed_counts = df[variation_col].value_counts().sort_index()
    control_count = observed_counts.get('A', 0)
    treatment_count = observed_counts.get('B', 0)
    total_count = control_count + treatment_count
    
    if total_count == 0:
        return {"error": "No data found."}
        
    expected_counts = [total_count * 0.5, total_count * 0.5]
    observed_counts_list = [control_count, treatment_count]
    
    srm_test = chisquare(f_obs=observed_counts_list, f_exp=expected_counts)
    srm_p_value = srm_test.pvalue
    srm_passed = srm_p_value >= 0.01 # Common threshold
    
    if not srm_passed:
        return {
            "error": "SRM Check Failed. Experiment is invalid.",
            "srm_p_value": srm_p_value,
            "observed_split": {"A": int(control_count), "B": int(treatment_count)}
        }
        
    # --- 2. CUPED Implementation ---
    # Y_cuped = Y - theta * (X - mu_X)
    
    # Calculate mu_X (mean of covariate across all users)
    mu_x = df[covariate_col].mean()
    
    # Calculate theta = cov(Y, X) / var(X)
    # Using data from control group is a common, robust practice
    control_df = df[df[variation_col] == 'A']
    cov_matrix = np.cov(control_df[metric_col], control_df[covariate_col])
    
    # Check for zero variance in covariate
    if cov_matrix[1, 1] == 0:
        theta = 0 # Covariate has no variance, cannot be used
    else:
        theta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # Apply CUPED adjustment to all users
    metric_cuped = f"{metric_col}_cuped"
    df[metric_cuped] = df[metric_col] - theta * (df[covariate_col] - mu_x)
    
    # --- 3. T-test and Metrics Calculation ---
    control = df[df[variation_col] == 'A'][metric_cuped]
    treatment = df[df[variation_col] == 'B'][metric_cuped]
    
    control_mean = control.mean()
    treatment_mean = treatment.mean()
    
    # Run t-test on the variance-reduced CUPED metric
    t_stat, p_value = ttest_ind(treatment, control, equal_var=False) # Welch's T-test
    
    lift_absolute = treatment_mean - control_mean
    lift_percent = (lift_absolute / control_mean) * 100
    
    # --- 4. Confidence Interval Calculation ---
    # Based on the t-test results
    n1, n2 = len(control), len(treatment)
    var1, var2 = control.var(ddof=1), treatment.var(ddof=1)
    
    # Standard Error of the difference
    se_diff = np.sqrt(var1/n1 + var2/n2)
    
    # Degrees of freedom for Welch's T-test
    dof = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    # 95% Confidence Interval
    alpha = 0.05
    margin_of_error = t_dist.ppf(1 - alpha/2, df=dof) * se_diff
    
    ci_low = lift_absolute - margin_of_error
    ci_high = lift_absolute + margin_of_error
    
    # --- 5. Variance Reduction Check ---
    var_original = df[metric_col].var()
    var_cuped = df[metric_cuped].var()
    variance_reduction_percent = (1 - var_cuped / var_original) * 100

    # --- 6. Format for JSON-safe output ---
    # Convert all numpy-specific types to standard Python types
    return {
        "srm_check": {
            "passed": bool(srm_passed),
            "p_value": float(srm_p_value),
            "split": {"A": int(control_count), "B": int(treatment_count)}
        },
        "cuped_adjustment": {
            "theta": float(theta),
            "variance_reduction_percent": float(variance_reduction_percent)
        },
        "results": {
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
            "lift_absolute": float(lift_absolute),
            "lift_percent": float(lift_percent),
            "p_value": float(p_value),
            "is_stat_sig": bool(p_value < 0.05),
            "confidence_interval_absolute": [float(ci_low), float(ci_high)]
        }
    }

def run_analysis_proportions(df: pd.DataFrame, user_col: str, numerator_col: str, denominator_col: str, variation_col: str = 'variation') -> dict:
    """
    Runs an A/B test analysis for proportion metrics (e.g., conversion rate).
    This performs aggregation at the user level, then a Z-test.
    
    :param df: DataFrame with one row per *event* (e.g., page view). Must contain:
               - user_col (e.g., 'user_id')
               - variation_col (e.g., 'A', 'B')
               - numerator_col (e.g., 'converted' - 1 for conversion, 0 otherwise)
               - denominator_col (e.g., 'session_id' - for aggregation, or a col of 1s)
    :param user_col: The name of the unique user identifier column.
    :param numerator_col: The name of the column representing a successful event (e.g., 'converted').
    :param denominator_col: The name of the column representing an exposure (e.g., 'session').
    :param variation_col: The name of the variation assignment column.
    :return: A dictionary of results.
    """

    # --- 1. Aggregate data to user-level ---
    # This is crucial. We need one 'n' (denominator) and one 'x' (numerator) per user.
    # We group by user *and* variation to keep the assignment
    user_data = df.groupby([user_col, variation_col]).agg(
        X=(numerator_col, 'sum'),  # Total conversions (numerator)
        N=(denominator_col, 'count') # Total sessions (denominator)
    ).reset_index()

    # To be robust, we only count users with at least one exposure
    user_data = user_data[user_data['N'] > 0]
    
    # --- 2. SRM Check (on users) ---
    observed_counts = user_data[variation_col].value_counts().sort_index()
    control_count = observed_counts.get('A', 0)
    treatment_count = observed_counts.get('B', 0)
    total_count = control_count + treatment_count

    if total_count == 0:
        return {"error": "No data found."}

    expected_counts = [total_count * 0.5, total_count * 0.5]
    srm_test = chisquare(f_obs=[control_count, treatment_count], f_exp=expected_counts)
    srm_p_value = srm_test.pvalue
    srm_passed = srm_p_value >= 0.01

    if not srm_passed:
        return {
            "error": "SRM Check Failed (on Users). Experiment is invalid.",
            "srm_p_value": float(srm_p_value),
            "observed_split": {"A": int(control_count), "B": int(treatment_count)}
        }

    # --- 3. Calculate Pooled Z-Test Stats ---
    control_users = user_data[user_data[variation_col] == 'A']
    treatment_users = user_data[user_data[variation_col] == 'B']

    # n = number of users (not sessions)
    n_A = len(control_users)
    n_B = len(treatment_users)

    # x = total conversions (sum of numerators)
    x_A = control_users['X'].sum()
    x_B = treatment_users['X'].sum()

    # N = total sessions (sum of denominators)
    N_A = control_users['N'].sum()
    N_B = treatment_users['N'].sum()

    if N_A == 0 or N_B == 0:
        return {"error": "No denominator data for one or both variations."}

    # p = conversion rate (p_A, p_B)
    p_A = x_A / N_A
    p_B = x_B / N_B

    # p_pooled = pooled conversion rate
    p_pooled = (x_A + x_B) / (N_A + N_B)

    # --- 4. Run the Z-Test ---
    # Z = (p_B - p_A) / sqrt( p_pooled * (1 - p_pooled) * (1/N_A + 1/N_B) )
    numerator = p_B - p_A
    denominator_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/N_A + 1/N_B))

    if denominator_pooled == 0:
         return {"error": "Z-test denominator is zero. Cannot calculate."}
         
    z_score = numerator / denominator_pooled
    
    # p-value from z-score (two-tailed)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    # --- 5. Confidence Interval Calculation ---
    # CI = (p_B - p_A) +/- Z_crit * sqrt( p_A*(1-p_A)/N_A + p_B*(1-p_B)/N_B )
    alpha = 0.05
    z_crit = norm.ppf(1 - alpha / 2)
    
    se_diff = np.sqrt((p_A * (1 - p_A) / N_A) + (p_B * (1 - p_B) / N_B))
    margin_of_error = z_crit * se_diff
    
    lift_absolute = p_B - p_A
    ci_low = lift_absolute - margin_of_error
    ci_high = lift_absolute + margin_of_error
    
    lift_percent = (lift_absolute / p_A) * 100 if p_A != 0 else 0

    # --- 6. Format for JSON-safe output ---
    return {
        "srm_check": {
            "passed": bool(srm_passed),
            "p_value": float(srm_p_value),
            "split": {"A": int(control_count), "B": int(treatment_count)}
        },
        "cuped_adjustment": { # Not applicable for this test
            "theta": None,
            "variance_reduction_percent": 0.0
        },
        "results": {
            "control_mean": float(p_A), # "mean" is conversion rate here
            "treatment_mean": float(p_B),
            "lift_absolute": float(lift_absolute),
            "lift_percent": float(lift_percent),
            "p_value": float(p_value),
            "is_stat_sig": bool(p_value < 0.05),
            "confidence_interval_absolute": [float(ci_low), float(ci_high)]
        }
    }

def run_analysis_sequential(
    df: pd.DataFrame, 
    user_col: str, 
    metric_col: str, 
    covariate_col: str, 
    total_peeks: int,
    variation_col: str = 'variation'
) -> dict:
    """
    Runs a Group Sequential Analysis for a continuous metric, using CUPED.
    This is for "peeking" at a test.
    
    It uses the Bonferroni correction method, which is simple and robust.
    The p-value will be compared against a corrected alpha.
    
    :param df: DataFrame, same as run_analysis.
    :param user_col: User ID column.
    :param metric_col: Metric column (e.g., 'revenue').
    :param covariate_col: Covariate column (e.g., 'revenue_pre').
    :param total_peeks: The TOTAL number of times you plan to check this test (e.g., 4).
    :param variation_col: Variation column.
    :return: A dictionary of results, with 'is_stat_sig' adjusted.
    """
    
    if total_peeks <= 0:
        return {"error": "Total peeks must be at least 1."}

    # --- 1. Run the standard CUPED T-test ---
    # We get all the same calculations for free.
    results = run_analysis(
        df=df,
        user_col=user_col,
        metric_col=metric_col,
        covariate_col=covariate_col,
        variation_col=variation_col
    )
    
    # If the analysis failed (e.g., SRM), just return the error
    if "error" in results:
        return results

    # --- 2. Apply Bonferroni Correction ---
    standard_alpha = 0.05
    adjusted_alpha = standard_alpha / total_peeks
    
    # Get the p-value from the T-test
    p_value = results['results']['p_value']
    
    # --- 3. Overwrite the significance logic ---
    # Re-judge the 'is_stat_sig' field based on the *new* adjusted alpha
    is_stat_sig_adjusted = p_value < adjusted_alpha
    
    results['results']['is_stat_sig'] = bool(is_stat_sig_adjusted)
    
    # Add info about this sequential test to the output
    results['sequential_test_info'] = {
        'method': 'Group Sequential (Bonferroni)',
        'total_peeks': int(total_peeks),
        'standard_alpha': standard_alpha,
        'adjusted_alpha': float(adjusted_alpha),
        'note': f"P-value ({p_value:.6f}) was compared against the adjusted alpha ({adjusted_alpha:.6f})."
    }
    
    return results