import numpy as np
import pandas as pd
import os

def generate_mock_data(num_users=20000, lift_effect=2.5, prop_lift=0.02, continuous_filename="mock_continuous.csv", prop_filename="mock_proportions.csv"):
    """
    Generates CSVs with mock data for all test types.
    """
    print(f"Generating mock data with {num_users} users...")
    
    # --- Base User Data ---
    data = {
        'user_id': [f"user_{i}" for i in range(num_users)],
        'variation': np.random.choice(['A', 'B'], num_users, p=[0.5, 0.5]),
    }
    df = pd.DataFrame(data)

    # --- Continuous Metric (CUPED) Data ---
    df['revenue_pre'] = np.random.normal(loc=50, scale=10, size=num_users).clip(0)
    noise = np.random.normal(loc=0, scale=3, size=num_users)
    df['revenue'] = (df['revenue_pre'] * 0.8 + noise).clip(0) # Strong correlation
    
    # Apply the lift effect
    df.loc[df['variation'] == 'B', 'revenue'] += lift_effect
    
    # --- Proportion Metric Data ---
    session_data = {
        'user_id': np.random.choice(df['user_id'], size=num_users * 5), # 5 sessions per user on avg
        'session_id': [f"session_{i}" for i in range(num_users * 5)],
    }
    sessions_df = pd.DataFrame(session_data)
    sessions_df = sessions_df.merge(df[['user_id', 'variation']], on='user_id', how='left')
    
    baseline_cr = 0.10 # 10%
    
    sessions_df['converted'] = 0
    
    control_mask = sessions_df['variation'] == 'A'
    control_sessions = sessions_df[control_mask]
    control_converted = control_sessions.sample(frac=baseline_cr)
    sessions_df.loc[control_converted.index, 'converted'] = 1
    
    treatment_mask = sessions_df['variation'] == 'B'
    treatment_sessions = sessions_df[treatment_mask]
    treatment_converted = treatment_sessions.sample(frac=baseline_cr + prop_lift)
    sessions_df.loc[treatment_converted.index, 'converted'] = 1
    
    # --- Save Files ---
    user_df = df
    user_df.to_csv(continuous_filename, index=False)
    print(f"Saved continuous (CUPED) data to {os.path.abspath(continuous_filename)}")

    sessions_df.to_csv(prop_filename, index=False)
    print(f"Saved proportion (Z-test) data to {os.path.abspath(prop_filename)}")

if __name__ == "__main__":
    generate_mock_data()
    print("Mock data generation complete.")