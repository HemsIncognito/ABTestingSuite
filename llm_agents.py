import os
import google.generativeai as genai
import json
import re # We'll still use this for finding the JSON

# --- Load API Key ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # This error will show in the Docker logs if the .env file fails
    raise ValueError("FATAL ERROR: No GEMINI_API_KEY found. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# --- LLM Agent 1: The Experiment Designer ---
def get_experiment_spec(product_idea: str) -> dict:
    """
    Calls the Gemini API to get a structured experiment spec.
    
    This version avoids 'system_instruction' and manually builds the prompt,
    just like the working 'get_executive_summary' function.
    """
    
    SCHEMA = """
    {
      "hypothesis_if_then": "If we [change], then [expected user behavior] because [reason].",
      "primary_metric": "The main metric to measure the 'expected behavior'.",
      "secondary_metrics": [
        "A list of guardrail metrics (e.g., Page Load Time)",
        "A list of secondary success metrics (e.g., Revenue)"
      ],
      "audience": "The specific user segment for this experiment.",
      "power_calc_inputs": {
        "metric_baseline": "The baseline value of the primary metric (e.g., '0.15' for 15% CR).",
        "mde_percent": "The relative minimum detectable effect (e.g., '2.0' for a 2% lift)."
      }
    }
    """
    
    system_prompt = f"""
    You are an expert Senior Product Analyst. A Product Manager has a raw idea.
    Your job is to design a single, rigorous A/B experiment spec.
    
    Respond ONLY with a JSON object. Do not include markdown (```json) or any
    other text before or after the JSON.
    
    The JSON must follow this exact schema:
    {SCHEMA}
    """

    # --- THIS IS THE FIX ---
    # We manually combine the system prompt and the user idea
    # into a single prompt, just like the other working function.
    
    full_prompt = system_prompt + f"\n\nUser's Idea:\n{product_idea}"

    try:
        # --- FIX 1 ---
        # We initialize the model *without* the system_instruction
        # --- FIX 2 ---
        # We use the stable 'gemini-pro' model
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro"
        )
        
        # We pass the single, combined prompt
        response = model.generate_content(full_prompt)
        
        # --- Robust Parsing Logic ---
        text = response.text
        
        # 1. Try to find the JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if not match:
            # LLM failed to return a JSON block at all
            raise ValueError(f"LLM response contained no JSON. Raw response: {text}")
        
        json_string = match.group(0)
        
        # 2. Try to parse the found block
        try:
            json_dict = json.loads(json_string)
            return json_dict # <-- SUCCESS! Return the dictionary
        except json.JSONDecodeError as e:
            # The block we found wasn't valid JSON
            raise ValueError(f"LLM returned invalid JSON. Parse error: {e}. Raw text: {json_string}")

    except Exception as e:
        # Catch-all for API errors or other issues
        raise ValueError(f"Error during Gemini API call: {str(e)}")


# --- LLM Agent 2: The Executive Reporter ---
def get_executive_summary(results_json_dict: dict) -> str:
    """
    Calls the Gemini API to summarize results.
    Takes a DICTIONARY of results.
    (This function is known to work)
    """
    
    system_prompt = """
    You are a Senior Analyst writing a brief executive summary for leadership.
    You will be given a JSON blob of A/B test results.
    
    Write a 3-bullet-point summary in Markdown:
    - **Outcome:** What was the result? (e.g., "The treatment was stat-sig positive.")
    - **Key Metrics:** State the lift, p-value, and CI. Be precise.
    - **Recommendation:** "LAUNCH", "DO NOT LAUNCH", or "INVESTIGATE".
    
    Be concise, data-driven, and confident. Do not add any extra text.
    """

    # We must stringify the dict to put it in the prompt
    prompt = system_prompt + f"\n\nHere is the results JSON:\n{json.dumps(results_json_dict, indent=2)}"

    try:
        # We use 'gemini-flash' model here
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash"
        )
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"