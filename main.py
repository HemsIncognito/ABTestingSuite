from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
import json
from pydantic import BaseModel
from typing import Optional

# Import our custom libraries
import stats_core
import llm_agents

app = FastAPI(title="A/B Testing Suite API")

# --- Helper Model for Pydantic Validation ---
class PowerCalcInput(BaseModel):
    baseline_mean: float
    mde_absolute: float # Renamed from 'mde' for clarity
    std_dev: float

class ExperimentIdea(BaseModel):
    product_idea: str

class ResultsJson(BaseModel):
    results: dict


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "A/B Testing API is running!"}

@app.post("/calculate-power")
async def calculate_power(data: PowerCalcInput):
    """
    Endpoint for the Power Calculator.
    """
    try:
        sample_size = stats_core.calculate_sample_size(
            baseline_mean=data.baseline_mean,
            baseline_std_dev=data.std_dev,
            mde=data.mde_absolute
        )
        return {"required_sample_size_per_variation": sample_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/design-experiment")
async def design_experiment(idea: ExperimentIdea):
    """
    Endpoint for the LLM Experiment Designer.
    This is now much simpler and more robust.
    """
    try:
        # This function now returns a DICT or raises a ValueError
        spec_dict = llm_agents.get_experiment_spec(idea.product_idea)
        
        # FastAPI will automatically serialize the dict to a JSON response
        return spec_dict # <-- NO MORE json.loads()
        
    except ValueError as ve:
        # Catch parsing errors from our agent
        raise HTTPException(status_code=500, detail=f"LLM Parsing Error: {str(ve)}")
    except Exception as e:
        # Catch all other errors
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {str(e)}")

@app.post("/generate-summary")
async def generate_summary(data: ResultsJson):
    """
    Endpoint for the LLM Reporter Agent.
    """
    try:
        # Pydantic model already parsed the dict, pass it directly
        summary_text = llm_agents.get_executive_summary(data.results)
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {str(e)}")

@app.post("/analyze-results")
async def analyze_results(
    file: UploadFile = File(...),
    analysis_type: str = Form(...),
    user_col: str = Form(...),
    variation_col: str = Form(...),
    metric_col: Optional[str] = Form(None),
    covariate_col: Optional[str] = Form(None),
    numerator_col: Optional[str] = Form(None),
    denominator_col: Optional[str] = Form(None),
    total_peeks: Optional[int] = Form(None)
):
    """
    THE MAIN "SMART" ENDPOINT.
    """
    try:
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    try:
        results = {}
        
        # --- Route to the correct analysis function ---
        
        if analysis_type == "Continuous (CUPED)":
            if not metric_col or not covariate_col:
                raise ValueError("Missing 'metric_col' or 'covariate_col' for CUPED.")
            results = stats_core.run_analysis(
                df=df,
                user_col=user_col,
                metric_col=metric_col,
                covariate_col=covariate_col,
                variation_col=variation_col
            )
        
        elif analysis_type == "Proportion (Z-Test)":
            if not numerator_col or not denominator_col:
                raise ValueError("Missing 'numerator_col' or 'denominator_col' for Z-Test.")
            results = stats_core.run_analysis_proportions(
                df=df,
                user_col=user_col,
                numerator_col=numerator_col,
                denominator_col=denominator_col,
                variation_col=variation_col
            )
        
        elif analysis_type == "Sequential (Bonferroni)":
            if not metric_col or not covariate_col or not total_peeks:
                raise ValueError("Missing 'metric_col', 'covariate_col', or 'total_peeks' for Sequential.")
            results = stats_core.run_analysis_sequential(
                df=df,
                user_col=user_col,
                metric_col=metric_col,
                covariate_col=covariate_col,
                total_peeks=total_peeks,
                variation_col=variation_col
            )
            
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")
        
        # Check if the stats function returned an error
        if "error" in results:
            raise ValueError(results["error"])
            
        # Return the results as JSON
        # The stats_core functions already return JSON-safe dicts
        return JSONResponse(content=results)

    except Exception as e:
        # This will catch errors from pandas or the stats_core functions
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")