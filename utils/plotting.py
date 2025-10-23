# utils/plotting.py
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
    if parsed.get('gen_mse'):
        try:
            plt.figure(figsize=(8,4))
            sns.lineplot(x=range(1, len(parsed['gen_mse'])+1), y=parsed['gen_mse'], marker='o')
            plt.xlabel('Generation')
            plt.ylabel('Best MSE')
            plt.title('GA Best MSE per Generation' + title_suffix)
            plt.tight_layout()
            p = outdir / f'ga_mse_per_gen{("_"+ds_name) if ds_name else ""}.png'
            plt.savefig(p)
        except Exception as e:
            p = outdir / f'ga_mse_per_gen_placeholder.png'
            plt.figure(figsize=(6,2))
            plt.text(0.5,0.5,f'Plot error: {e}', ha='center')
            plt.axis('off')
            plt.savefig(p)

def plot_comparisons(parsed, outdir: Path, ds_name: str = None):
    comps = parsed.get('comparisons', {})
    title_suffix = f" - {ds_name}" if ds_name else ''
    methods = []
    mses = []
    counts = []
    times = []
    methods.append('GA')
    mses.append(parsed.get('best_mse'))
    counts.append(len(parsed.get('selected', [])))
    times.append(parsed.get('ga_time', 0.0))
    for name in ['SelectKBest', 'VarianceThreshold', 'MutualInfo_topK', 'LassoCV', 'RandomForest_topK', 'RFE']:
        if name in comps:
            v = comps[name]
            methods.append(name)
            mses.append(v.get('cv_mse'))
            counts.append(len(v.get('selected', []) if v.get('selected') is not None else []))
            times.append(v.get('time_s', 0.0))
    # Plot MSE
    try:
        plt.figure(figsize=(8,5))
        sns.barplot(x=mses, y=methods)
        plt.xlabel('CV MSE (lower is better)')
        plt.title('Comparison: CV MSE by method' + title_suffix)
        plt.tight_layout()
        p = outdir / f'comparison_mse{("_"+ds_name) if ds_name else ""}.png'
        plt.savefig(p)
    except: pass
    # Plot counts
    try:
        plt.figure(figsize=(8,5))
        sns.barplot(x=counts, y=methods)
        plt.xlabel('Number of selected features')
        plt.title('Comparison: selected feature counts')
        plt.tight_layout()
        p2 = outdir / f'comparison_counts{("_"+ds_name) if ds_name else ""}.png'
        plt.savefig(p2)
    except: pass
    # Plot Jaccard
    ga_sel = parsed.get('selected', [])
    jac_vals = []
    names = []
    for name in ['SelectKBest', 'VarianceThreshold', 'MutualInfo_topK', 'LassoCV', 'RandomForest_topK', 'RFE']:
        if name in comps:
            v = comps[name]
            sel = v.get('selected', []) or []
            jac = jaccard(ga_sel, sel)
            jac_vals.append(jac)
            names.append(name)
    try:
        plt.figure(figsize=(8,4))
        sns.barplot(x=jac_vals, y=names)
        plt.xlabel('Jaccard similarity with GA')
        plt.xlim(0,1)
        plt.title('Overlap between GA and other methods')
        plt.tight_layout()
        p3 = outdir / f'comparison_jaccard{("_"+ds_name) if ds_name else ""}.png'
        plt.savefig(p3)
    except: pass
    # Plot Time vs MSE
    if any(t > 0 for t in times):
        try:
            plt.figure(figsize=(8,5))
            plt.scatter(times, mses, s=100)
            for i, method in enumerate(methods):
                plt.text(times[i], mses[i], method, ha='center', va='bottom')
            plt.xlabel('Time (seconds)')
            plt.ylabel('CV MSE')
            plt.title('Time vs MSE by method' + title_suffix)
            plt.tight_layout()
            p4 = outdir / f'comparison_time_vs_mse{("_"+ds_name) if ds_name else ""}.png'
            plt.savefig(p4)
        except: pass