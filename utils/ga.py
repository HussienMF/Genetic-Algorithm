# utils/ga.py
"""
Professional Genetic Algorithm for Feature Selection.
Supports both regression and classification via scoring parameter.
"""
import random
from typing import List, Tuple, Callable, Optional
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.model_selection import cross_val_score

def generate_genome(length: int) -> List[int]:
    genome = [random.choice([0, 1]) for _ in range(length)]
    if not any(genome):
        genome[random.randrange(length)] = 1
    return genome

def generate_population(size: int, genome_length: int) -> List[List[int]]:
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(
    genome: List[int],
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], RegressorMixin],
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error'
) -> float:
    selected_cols = [i for i, bit in enumerate(genome) if bit]
    if not selected_cols:
        return float('inf')
    X_subset = X.iloc[:, selected_cols]
    model = model_factory()
    try:
        scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring, n_jobs=-1)
        if scoring == 'neg_mean_squared_error':
            return -float(scores.mean())
        else:
            return -float(scores.mean())
    except Exception:
        return float('inf')

def tournament_selection(pop: List[List[int]], fitnesses: List[float], k: int = 3) -> List[int]:
    participants = random.sample(range(len(pop)), k)
    winner_idx = min(participants, key=lambda i: fitnesses[i])
    return pop[winner_idx][:]

def one_point_crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
    if len(a) < 2:
        return a[:], b[:]
    point = random.randrange(1, len(a))
    return a[:point] + b[point:], b[:point] + a[point:]

def mutate(genome: List[int], mutation_rate: float) -> None:
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] = 1 - genome[i]
    if not any(genome):
        genome[random.randrange(len(genome))] = 1

def run_ga(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], RegressorMixin],
    pop_size: int = 50,
    generations: int = 40,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    cv: int = 5,
    tol: float = 1e-6,
    patience: int = 5,
    verbose: bool = False,
    seed: Optional[int] = None,
    scoring: str = 'neg_mean_squared_error'
) -> Tuple[List[int], float, List[float]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    genome_length = X.shape[1]
    population = generate_population(pop_size, genome_length)
    best_genome = None
    best_fitness = float('inf')
    history = []
    no_improve = 0
    prev_best = best_fitness
    for gen in range(generations):
        fitnesses = [fitness(g, X, y, model_factory, cv, scoring) for g in population]
        current_best_idx = int(np.argmin(fitnesses))
        current_best = fitnesses[current_best_idx]
        if current_best < best_fitness:
            best_fitness = current_best
            best_genome = population[current_best_idx][:]
        history.append(best_fitness)
        if verbose:
            print(f"Gen {gen+1}/{generations} best score: {-best_fitness:.4f}" if scoring != 'neg_mean_squared_error' else f"Gen {gen+1}/{generations} best MSE: {best_fitness:.4f}")
        if prev_best - best_fitness > tol:
            no_improve = 0
            prev_best = best_fitness
        else:
            no_improve += 1
        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at generation {gen+1}")
            break
        new_pop = []
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            if random.random() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_pop.extend([child1, child2][:pop_size - len(new_pop)])
        population = new_pop
    return best_genome, best_fitness, history