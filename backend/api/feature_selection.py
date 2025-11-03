from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pathlib import Path
import time
import pandas as pd
import shutil
import requests
import sys
from typing import Optional
from utils.data import prepare_data_for_example
from utils.comparison import get_model_factory, run_comparison_method
from utils.plotting import plot_results, plot_comparisons
from utils.ga_optimized import run_ga as run_ga_optimized
from utils.ga_original import run_ga as run_ga_original
from ..config import logger, UPLOAD_DIR, OUTPUT_BASE

router = APIRouter()

from typing import List

@router.post("/run")
async def run_feature_selection(
    file: UploadFile = File(...),  # Make file required
    target_column: str = Form(default=None),
    problem_type: str = Form(default="regression"),
    pop_size: int = Form(default=40),
    generations: int = Form(default=12),
    mutation_rate: float = Form(default=0.02),
    crossover_rate: float = Form(default=0.8),
    cv: int = Form(default=3),
    model_type: str = Form(default="linear"),
    ga_version: str = Form(default="optimized"),
    mode: str = Form(default="all"),
    methods: str = Form(default="[]")  # Accept as string to parse later
):
    """Run genetic algorithm feature selection with comparison methods."""
    logger.info(f"Starting feature selection: problem_type={problem_type}, pop_size={pop_size}, generations={generations}, ga_version={ga_version}")

    # ===== INPUT VALIDATION =====
    if not file:
        logger.warning("No file provided")
        raise HTTPException(status_code=400, detail="Please provide a CSV file")

    if not (10 <= pop_size <= 200):
        raise HTTPException(status_code=400, detail="Population size must be between 10 and 200")
    if not (5 <= generations <= 100):
        raise HTTPException(status_code=400, detail="Generations must be between 5 and 100")
    if not (0.01 <= mutation_rate <= 0.1):
        raise HTTPException(status_code=400, detail="Mutation rate must be between 0.01 and 0.1")
    if not (0.5 <= crossover_rate <= 1.0):
        raise HTTPException(status_code=400, detail="Crossover rate must be between 0.5 and 1.0")
    if not (2 <= cv <= 10):
        raise HTTPException(status_code=400, detail="CV folds must be between 2 and 10")
    if problem_type not in ["regression", "classification"]:
        raise HTTPException(status_code=400, detail="Problem type must be 'regression' or 'classification'")
    if file and not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    temp_path = None
    try:
        # ===== FILE HANDLING =====
        logger.info(f"Uploading file: {file.filename}")
        temp_path = UPLOAD_DIR / file.filename
        file_content = await file.read()
        if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")
        
        # Ensure upload directory exists
        UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
        
        # Write file with proper error handling
        try:
            with temp_path.open("wb") as f:
                f.write(file_content)
        except Exception as e:
            logger.error(f"Failed to write file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

        # ===== CSV VALIDATION =====
        try:
            df_check = pd.read_csv(temp_path, nrows=5)
            if df_check.empty:
                raise ValueError("CSV file is empty")
            logger.info(f"CSV validated: {df_check.shape[1]} columns, {len(pd.read_csv(temp_path))} rows")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

        # ===== DATA PREPARATION =====
        try:
            X, y = prepare_data_for_example(
                str(temp_path),
                group_top=10,
                drop_numeric_features=False,
                target=target_column.strip() if target_column else None
            )
            logger.info(f"Data prepared: X shape={X.shape}, y shape={y.shape}")
            if X.shape[0] < cv:
                raise ValueError(f"Dataset has {X.shape[0]} rows, need at least {cv} for {cv}-fold CV")
            if X.shape[1] < 2:
                raise ValueError(f"Dataset has only {X.shape[1]} feature(s). Need at least 2 features.")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="CSV file not found")
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Target column not found: {str(e)}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Data preparation error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")

        # ===== SELECT GA IMPLEMENTATION =====
        if ga_version == "optimized":
            run_ga = run_ga_optimized
        else:
            run_ga = run_ga_original

        # ===== RUN GENETIC ALGORITHM =====
        is_classification = (problem_type.strip().lower() == "classification")
        scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
        model_factory = get_model_factory(model_type, is_classification=is_classification)
        logger.info("Starting Genetic Algorithm...")
        t0 = time.perf_counter()
        try:
            best_genome, best_score, history = run_ga(
                X, y, model_factory,
                pop_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                cv=cv,
                seed=42,
                verbose=False,
                scoring=scoring
            )
            ga_time = time.perf_counter() - t0
            ga_selected = [col for bit, col in zip(best_genome, X.columns) if bit]
            logger.info(f"GA completed in {ga_time:.2f}s. Selected {len(ga_selected)} features. Best score: {best_score:.4f}")
        except Exception as e:
            logger.error(f"GA execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Genetic Algorithm failed: {str(e)}")

        # ===== RUN COMPARISON METHODS =====
        all_methods = ["SelectKBest", "LassoCV", "RFE", "VarianceThreshold", "MutualInfo_topK", "RandomForest_topK"]
        # Parse methods from string
        try:
            methods_list = eval(methods) if methods else []
            if not isinstance(methods_list, list):
                methods_list = []
        except:
            methods_list = []
        methods_to_run = all_methods if mode == "all" else [m for m in methods_list if m in all_methods]
        k = max(1, len(ga_selected))
        logger.info(f"Running {len(methods_to_run)} comparison methods...")
        results = {"GA": {"selected": ga_selected, "mse": best_score if not is_classification else -best_score, "time": ga_time}}
        for method in methods_to_run:
            try:
                res = run_comparison_method(method, X, y, k, model_factory, cv, is_classification=is_classification)
                results[method] = res
                logger.info(f"{method}: {len(res['selected'])} features, score={res['mse']:.4f}, time={res['time']:.2f}s")
            except Exception as e:
                logger.error(f"Method {method} failed: {str(e)}")
                results[method] = {"selected": [], "mse": None, "time": 0.0, "error": str(e)}

        # ===== GENERATE PLOTS =====
        ds_name = temp_path.stem
        out_dir = OUTPUT_BASE / ds_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(exist_ok=True)
        parsed = {
            'gen_mse': history,
            'best_mse': best_score if not is_classification else -best_score,
            'selected': ga_selected,
            'ga_time': ga_time,
            'comparisons': {
                m: {'selected': r['selected'], 'cv_mse': r['mse'], 'time_s': r['time']}
                for m, r in results.items() if m != 'GA'
            },
            'is_classification': is_classification
        }
        try:
            logger.info("Generating plots...")
            plot_results(parsed, out_dir, ds_name=ds_name)
            plot_comparisons(parsed, out_dir, ds_name=ds_name)
            logger.info("Plots generated successfully")
        except Exception as e:
            logger.error(f"Plot generation failed: {str(e)}")

        metric_suffix = "accuracy" if is_classification else "mse"
        plots = [
            f"/outputs/{ds_name}/ga_{metric_suffix}_per_gen_{ds_name}.png",
            f"/outputs/{ds_name}/comparison_{metric_suffix}_{ds_name}.png",
            f"/outputs/{ds_name}/comparison_jaccard_{ds_name}.png",
            f"/outputs/{ds_name}/comparison_time_vs_{metric_suffix}_{ds_name}.png",
            f"/outputs/{ds_name}/comparison_counts_{ds_name}.png"
        ]

        logger.info("Feature selection completed successfully")
        return JSONResponse({
            "dataset": ds_name,
            "model_type": model_type,
            "problem_type": problem_type,
            "results": results,
            "plots": plots,
            "metadata": {
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "cv_folds": cv,
                "pop_size": pop_size,
                "generations": generations,
                "total_time": sum(r['time'] for r in results.values())
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please check your data and try again.")
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {str(e)}")