# backend/main.py
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import requests
import time

from utils.data import prepare_data_for_example
from utils.ga import run_ga
from utils.comparison import get_model_factory, run_comparison_method
from utils.plotting import plot_results, plot_comparisons
from fastapi.middleware.cors import CORSMiddleware


project_root = Path(__file__).parent.parent  # ← V1_BIA601/
sys.path.insert(0, str(project_root))

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_BASE = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_BASE.mkdir(exist_ok=True)

app = FastAPI(title="Genetic Feature Selection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← يسمح بجميع الطلبات (للتطوير)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_BASE), name="outputs")

# UPLOAD_DIR = Path("../uploads")
# OUTPUT_BASE = Path("../outputs")
# UPLOAD_DIR.mkdir(exist_ok=True)
# OUTPUT_BASE.mkdir(exist_ok=True)

@app.post("/run")
async def run_feature_selection(
    file: UploadFile = File(None),
    url: str = Form(None),
    pop_size: int = Form(40),
    generations: int = Form(12),
    mutation_rate: float = Form(0.02),
    crossover_rate: float = Form(0.8),
    cv: int = Form(3),
    model_type: str = Form("linear"),
    mode: str = Form("all"),
    methods: list = Form([])
):
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide file or URL")

    temp_path = None
    try:
        if file:
            temp_path = UPLOAD_DIR / file.filename
            with temp_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        elif url:
            resp = requests.get(url)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download URL")
            temp_path = UPLOAD_DIR / "downloaded.csv"
            temp_path.write_bytes(resp.content)

        X, y = prepare_data_for_example(str(temp_path), group_top=10, drop_numeric_features=False)
        model_factory = get_model_factory(model_type)

        t0 = time.perf_counter()
        best_genome, best_mse, history = run_ga(
            X, y, model_factory,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            cv=cv,
            seed=42,
            verbose=False
        )
        ga_time = time.perf_counter() - t0
        ga_selected = [col for bit, col in zip(best_genome, X.columns) if bit]

        all_methods = ["SelectKBest", "LassoCV", "RFE", "VarianceThreshold", "MutualInfo_topK", "RandomForest_topK"]
        methods_to_run = all_methods if mode == "all" else [m for m in methods if m in all_methods]

        k = max(1, len(ga_selected))
        results = {"GA": {"selected": ga_selected, "mse": best_mse, "time": ga_time}}
        for method in methods_to_run:
            res = run_comparison_method(method, X, y, k, model_factory, cv)
            results[method] = res

        ds_name = temp_path.stem
        out_dir = OUTPUT_BASE / ds_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(exist_ok=True)

        parsed = {
            'gen_mse': history,
            'best_mse': best_mse,
            'selected': ga_selected,
            'ga_time': ga_time,
            'comparisons': {
                m: {'selected': r['selected'], 'cv_mse': r['mse'], 'time_s': r['time']}
                for m, r in results.items() if m != 'GA'
            }
        }

        plot_results(parsed, out_dir, ds_name=ds_name)
        plot_comparisons(parsed, out_dir, ds_name=ds_name)

        plots = [f"/outputs/{ds_name}/{img}" for img in [
            f"ga_mse_per_gen_{ds_name}.png",
            f"comparison_mse_{ds_name}.png",
            f"comparison_jaccard_{ds_name}.png",
            f"comparison_time_vs_mse_{ds_name}.png",
            f"comparison_counts_{ds_name}.png"
        ]]

        return JSONResponse({
            "dataset": ds_name,
            "model_type": model_type,
            "results": results,
            "plots": plots
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()