import json
import llm_agents # Import our new library
import stats_core # We'll need this for the mock results

def test_designer_agent():
    print("--- Running Test: Designer Agent ---")
    
    idea = "Change the main 'Buy Now' button on the checkout page from blue to bright green."
    print(f"Test Idea: {idea}\n")
    
    spec = llm_agents.get_experiment_spec(idea)
    
    print("Agent Response (JSON):")
    print(json.dumps(spec, indent=2))
    
    # Test the structure of the response
    assert "hypothesis_if_then" in spec
    assert "primary_metric" in spec
    assert "secondary_metrics" in spec
    assert "audience" in spec
    assert "power_calc_inputs" in spec
    assert "metric_baseline" in spec["power_calc_inputs"]
    assert "error" not in spec # Check that no error occurred
    
    print("\n✅ Designer Agent test passed.\n")

def test_reporter_agent():
    print("--- Running Test: Reporter Agent ---")
    
    # Get some mock results from our stats core
    # We don't need to generate data, just create a mock results dict
    mock_results = {
        "srm_check": {
            "passed": True,
            "p_value": 0.59,
            "split": {"A": 10010, "B": 9990}
        },
        "cuped_adjustment": {
            "theta": 0.82,
            "variance_reduction_percent": 58.78
        },
        "results": {
            "control_mean": 40.15,
            "treatment_mean": 42.85,
            "lift_absolute": 2.7,
            "lift_percent": 6.96,
            "p_value": 0.0000001,
            "is_stat_sig": True,
            "confidence_interval_absolute": [2.50, 3.05]
        }
    }
    
    print("Test Data (JSON):")
    print(json.dumps(mock_results, indent=2))
    
    # Get the summary
    summary = llm_agents.get_executive_summary(json.dumps(mock_results))
    
    print("\nAgent Response (Text):")
    print(summary)
    
    # Test the response
    assert "Error" not in summary
    assert "success" in summary.lower() or "launch" in summary.lower()
    assert "6.96%" in summary or "7.0%" in summary
    
    print("\n✅ Reporter Agent test passed.")

if __name__ == "__main__":
    print("Running A/B LLM Agent Tests...\n")
    test_designer_agent()
    test_reporter_agent()
    print("\nAll agent tests complete. Milestone 2 is functional.")
