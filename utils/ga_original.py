import random
from typing import List, Tuple, Callable, Optional
import pandas as pd
import numpy as np
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
    model_factory: Callable,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    max_samples: int = 5000,
    lambda_penalty: float = 0.05
) -> float:
    selected_cols = [i for i, bit in enumerate(genome) if bit]
    n_selected = len(selected_cols)
    if not selected_cols:
        return float('inf')
    
    X_subset = X.iloc[:, selected_cols]
    
    if len(X_subset) > max_samples:
        indices = random.sample(range(len(X_subset)), k=max_samples)
        X_sample = X_subset.iloc[indices]
        y_sample = y.iloc[indices]
    else:
        X_sample = X_subset
        y_sample = y

    model = model_factory()
    try:
        scores = cross_val_score(model, X_sample, y_sample, cv=cv, scoring=scoring, n_jobs=1)
        base_score = -float(scores.mean())
        penalty = lambda_penalty * (n_selected / X.shape[1])
        return base_score + penalty
    except Exception:
        return float('inf')

def tournament_selection(pop, fitnesses, k=3):
    participants = random.sample(range(len(pop)), k)
    winner_idx = min(participants, key=lambda i: fitnesses[i])
    return pop[winner_idx][:]

def one_point_crossover(a, b):
    if len(a) < 2:
        return a[:], b[:]
    point = random.randrange(1, len(a))
    return a[:point] + b[point:], b[:point] + a[point:]

def mutate(genome, mutation_rate):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] = 1 - genome[i]
    if not any(genome):
        genome[random.randrange(len(genome))] = 1

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
    max_samples: int = 5000,
    lambda_penalty: float = 0.05
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

    for gen in range(generations):
        fitnesses = [fitness(g, X, y, model_factory, cv, scoring, max_samples, lambda_penalty) for g in population]
        current_best_idx = int(np.argmin(fitnesses))
        current_best = fitnesses[current_best_idx]
        if current_best < best_fitness:
            best_fitness = current_best
            best_genome = population[current_best_idx][:]
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_fitness)
        if no_improve >= patience:
            break

        elite = best_genome[:]
        new_pop = [elite]
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            if random.random() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_pop.extend([child1, child2])
            if len(new_pop) > pop_size:
                new_pop.pop()
        population = new_pop

    return best_genome, best_fitness, history