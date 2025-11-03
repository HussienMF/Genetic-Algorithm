
"""
Optimized Genetic Algorithm with:
- Parallel fitness evaluation
- Smart sampling for large datasets
- Feature count penalty for minimal feature selection
- Performance improvements
"""
import random
from typing import List, Tuple, Callable, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import logging
logger = logging.getLogger(__name__)

def generate_genome(length: int) -> np.ndarray:
    genome = np.random.randint(0, 2, length, dtype=np.int8)
    if not genome.any():
        genome[np.random.randint(length)] = 1
    return genome

def generate_population(size: int, genome_length: int) -> List[np.ndarray]:
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(
    genome: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    max_samples: int = 5000,
    lambda_penalty: float = 0.05
) -> float:
    selected_cols = np.where(genome)[0]
    n_selected = len(selected_cols)
    if n_selected == 0:
        return float('inf')
    
    X_subset = X.iloc[:, selected_cols]
    
    # ⚡ أخذ عينة عشوائية إذا كانت البيانات كبيرة
    if len(X_subset) > max_samples:
        indices = np.random.choice(len(X_subset), size=max_samples, replace=False)
        X_sample = X_subset.iloc[indices]
        y_sample = y.iloc[indices]
    else:
        X_sample = X_subset
        y_sample = y

    model = model_factory()
    try:
        scores = cross_val_score(model, X_sample, y_sample, cv=cv, scoring=scoring, n_jobs=1)
        base_score = -float(scores.mean())  # MSE موجب
        
        # ⚡ إضافة عقوبة بعدد الميزات (لتشجيع البساطة)
        penalty = lambda_penalty * (n_selected / X.shape[1])
        return base_score + penalty
        
    except Exception as e:
        logger.warning(f"Fitness evaluation failed: {e}")
        return float('inf')

def fitness_parallel_wrapper(genome, X, y, model_factory, cv, scoring, max_samples, lambda_penalty):
    return fitness(genome, X, y, model_factory, cv, scoring, max_samples, lambda_penalty)

def evaluate_population_parallel(
    population: List[np.ndarray],
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable,
    cv: int,
    scoring: str,
    n_jobs: int = -1,
    max_samples: int = 5000,
    lambda_penalty: float = 0.05
) -> List[float]:
    fitnesses = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(fitness_parallel_wrapper)(g, X, y, model_factory, cv, scoring, max_samples, lambda_penalty)
        for g in population
    )
    return fitnesses

def tournament_selection(pop, fitnesses, k=3):
    participants = np.random.choice(len(pop), k, replace=False)
    winner_idx = participants[np.argmin([fitnesses[i] for i in participants])]
    return pop[winner_idx].copy()

def one_point_crossover(a, b):
    if len(a) < 2:
        return a.copy(), b.copy()
    point = np.random.randint(1, len(a))
    child1 = np.concatenate([a[:point], b[point:]])
    child2 = np.concatenate([b[:point], a[point:]])
    return child1, child2

def mutate(genome, mutation_rate):
    mutations = np.random.random(len(genome)) < mutation_rate
    genome[mutations] = 1 - genome[mutations]
    if not genome.any():
        genome[np.random.randint(len(genome))] = 1

def run_ga(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable,
    pop_size: int = 50,
    generations: int = 40,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    cv: int = 5,
    patience: int = 5,
    verbose: bool = False,
    seed: Optional[int] = None,
    scoring: str = 'neg_mean_squared_error',
    use_parallel: bool = True,
    n_jobs: int = -1,
    max_samples: int = 5000,
    lambda_penalty: float = 0.05
) -> Tuple[np.ndarray, float, List[float]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    genome_length = X.shape[1]
    population = generate_population(pop_size, genome_length)
    best_genome = None
    best_fitness = float('inf')
    history = []
    no_improve = 0

    logger.info(f"Starting GA: pop_size={pop_size}, generations={generations}, parallel={use_parallel}, lambda_penalty={lambda_penalty}")

    for gen in range(generations):
        if use_parallel:
            fitnesses = evaluate_population_parallel(
                population, X, y, model_factory, cv, scoring, n_jobs, max_samples, lambda_penalty
            )
        else:
            fitnesses = [
                fitness(g, X, y, model_factory, cv, scoring, max_samples, lambda_penalty)
                for g in population
            ]

        current_best_idx = int(np.argmin(fitnesses))
        current_best = fitnesses[current_best_idx]
        if current_best < best_fitness:
            best_fitness = current_best
            best_genome = population[current_best_idx].copy()
            no_improve = 0
            if verbose:
                n_features = best_genome.sum()
                logger.info(f"Gen {gen}: New best fitness={best_fitness:.4f}, features={n_features}")
        else:
            no_improve += 1

        history.append(best_fitness)

        if no_improve >= patience:
            logger.info(f"Early stopping at generation {gen}")
            break

        elite = best_genome.copy()
        new_pop = [elite]
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            if random.random() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_pop.extend([child1, child2])
        population = new_pop[:pop_size]

    logger.info(f"GA completed: best_fitness={best_fitness:.4f}, generations={len(history)}")
    return best_genome.tolist(), best_fitness, history
