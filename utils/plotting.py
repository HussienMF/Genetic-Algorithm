import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def jaccard(a, b):
    A = set(a)
    B = set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)

def plot_results(parsed, outdir: Path, ds_name: str = None):
    title_suffix = f" - {ds_name}" if ds_name else ''
    is_classification = parsed.get('is_classification', False)
    metric_label = "Accuracy" if is_classification else "MSE"
    metric_key = "Best Accuracy" if is_classification else "Best MSE"

    if parsed.get('gen_mse'):
        try:
            plt.figure(figsize=(8, 4))
            y_vals = parsed['gen_mse']
            # إذا كان تصنيفًا، فالقيم موجبة (دقة)، أما إذا انحدار فموجبة أيضًا (MSE)
            sns.lineplot(x=range(1, len(y_vals) + 1), y=y_vals, marker='o')
            plt.xlabel('Generation')
            plt.ylabel(f'Best {metric_label}')
            plt.title(f'GA Best {metric_label} per Generation' + title_suffix)
            plt.tight_layout()
            p = outdir / f'ga_{metric_label.lower()}_per_gen{("_" + ds_name) if ds_name else ""}.png'
            plt.savefig(p)
            plt.close()
        except Exception as e:
            p = outdir / f'ga_{metric_label.lower()}_per_gen_placeholder.png'
            plt.figure(figsize=(6, 2))
            plt.text(0.5, 0.5, f'Plot error: {e}', ha='center')
            plt.axis('off')
            plt.savefig(p)
            plt.close()

def plot_comparisons(parsed, outdir: Path, ds_name: str = None):
    comps = parsed.get('comparisons', {})
    is_classification = parsed.get('is_classification', False)
    metric_label = "Accuracy" if is_classification else "MSE"
    xlabel_metric = f'CV {metric_label} ({("higher" if is_classification else "lower")} is better)'

    title_suffix = f" - {ds_name}" if ds_name else ''

    methods = []
    metrics = []  # will hold MSE or Accuracy
    counts = []
    times = []

    # Add GA
    methods.append('GA')
    metrics.append(parsed.get('best_mse'))  # already converted in main.py
    counts.append(len(parsed.get('selected', [])))
    times.append(parsed.get('ga_time', 0.0))

    # Add other methods
    order = ['SelectKBest', 'VarianceThreshold', 'MutualInfo_topK', 'LassoCV', 'RandomForest_topK', 'RFE']
    for name in order:
        if name in comps:
            v = comps[name]
            methods.append(name)
            metrics.append(v.get('cv_mse'))
            counts.append(len(v.get('selected', []) or []))
            times.append(v.get('time_s', 0.0))

    # Plot Metric (MSE or Accuracy)
    try:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=metrics, y=methods, palette="viridis")
        plt.xlabel(xlabel_metric)
        plt.title(f'Comparison: CV {metric_label} by method' + title_suffix)
        plt.tight_layout()
        p = outdir / f'comparison_{metric_label.lower()}{("_" + ds_name) if ds_name else ""}.png'
        plt.savefig(p)
        plt.close()
    except Exception as e:
        print(f"Plot error (metric): {e}")

    # Plot counts
    try:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts, y=methods, palette="Blues_d")
        plt.xlabel('Number of selected features')
        plt.title('Comparison: selected feature counts' + title_suffix)
        plt.tight_layout()
        p2 = outdir / f'comparison_counts{("_" + ds_name) if ds_name else ""}.png'
        plt.savefig(p2)
        plt.close()
    except Exception as e:
        print(f"Plot error (counts): {e}")

    # Plot Jaccard similarity
    ga_sel = parsed.get('selected', [])
    jac_vals = []
    names = []
    for name in order:
        if name in comps:
            sel = comps[name].get('selected', []) or []
            jac = jaccard(ga_sel, sel)
            jac_vals.append(jac)
            names.append(name)
    if jac_vals:
        try:
            plt.figure(figsize=(8, 4))
            sns.barplot(x=jac_vals, y=names, palette="Greens_d")
            plt.xlabel('Jaccard similarity with GA')
            plt.xlim(0, 1)
            plt.title('Overlap between GA and other methods' + title_suffix)
            plt.tight_layout()
            p3 = outdir / f'comparison_jaccard{("_" + ds_name) if ds_name else ""}.png'
            plt.savefig(p3)
            plt.close()
        except Exception as e:
            print(f"Plot error (Jaccard): {e}")

    # Plot Time vs Metric
    if any(t > 0 for t in times):
        try:
            plt.figure(figsize=(8, 5))
            plt.scatter(times, metrics, s=100, color='red')
            for i, method in enumerate(methods):
                plt.text(times[i], metrics[i], method, ha='center', va='bottom', fontsize=9)
            plt.xlabel('Time (seconds)')
            plt.ylabel(f'CV {metric_label}')
            plt.title(f'Time vs {metric_label} by method' + title_suffix)
            plt.tight_layout()
            p4 = outdir / f'comparison_time_vs_{metric_label.lower()}{("_" + ds_name) if ds_name else ""}.png'
            plt.savefig(p4)
            plt.close()
        except Exception as e:
            print(f"Plot error (Time vs Metric): {e}")