"""Discrete search engines implemented with NumPy. No native-baseline claims."""
from __future__ import annotations

import numpy as np


def initial_population(space, rng, size, initial=None):
    if size < 2:
        raise ValueError("Population must contain at least two candidates")
    if initial is None:
        return np.array([space.sample(rng) for _ in range(size)])
    if len(initial) != size:
        raise ValueError("Shared initial population has wrong size")
    return np.array([space.validate(g) for g in initial])


class RandomSearch:
    name = "random"

    def run(self, session, rng, initial=None):
        for g in ([] if initial is None else initial):
            if session.remaining:
                session.evaluate(g)
        while session.remaining:
            session.evaluate(session.oracle.space.sample(rng))


class GeneticSearch:
    name = "ga"

    def __init__(self, population=20, mutation=0.15):
        if population < 2 or not 0 <= mutation <= 1:
            raise ValueError("Invalid GA settings")
        self.population, self.mutation = population, mutation

    def run(self, session, rng, initial=None):
        space = session.oracle.space
        pop = initial_population(space, rng, self.population, initial)
        scores = np.full(self.population, -np.inf)
        for i, g in enumerate(pop):
            if session.remaining:
                scores[i] = session.evaluate(g)
        while session.remaining:
            def tournament():
                ids = rng.integers(self.population, size=3)
                return pop[ids[np.argmax(scores[ids])]]
            children, values = [], []
            # Elitism retains the best evaluated parent without charging again.
            elite = int(np.argmax(scores))
            children.append(pop[elite].copy())
            values.append(scores[elite])
            while len(children) < self.population and session.remaining:
                a, b = tournament(), tournament()
                g = np.where(rng.random(space.dimension) < 0.5, a, b)
                for j in np.flatnonzero(rng.random(space.dimension) < self.mutation):
                    g[j] = rng.integers(space.lower_bound, space.cardinalities[j])
                g = space.repair(g, rng)
                values.append(session.evaluate(g))
                children.append(g)
            pop, scores = np.array(children), np.array(values)


class AntColonySearch:
    """Categorical construction ACO with rank deposits and bounded pheromone.

    Uniform heuristic desirability; no hidden, unbudgeted objective lookups.
    Order constraint is imposed during construction, not after depositing trails.
    """
    name = "aco"

    def __init__(self, population=20, evaporation=0.2, alpha=1.0, exploration=0.1):
        if population < 2 or not 0 < evaporation < 1 or alpha <= 0 or not 0 <= exploration <= 1:
            raise ValueError("Invalid ACO settings")
        self.population, self.evaporation = population, evaporation
        self.alpha, self.exploration = alpha, exploration

    def run(self, session, rng, initial=None):
        space = session.oracle.space
        trails = [np.ones(c - space.lower_bound) for c in space.cardinalities]
        shared = initial_population(space, rng, self.population, initial)
        first = True
        while session.remaining:
            ants, scores = [], []
            for i in range(min(self.population, session.remaining)):
                if first:
                    g = shared[i].copy()
                else:
                    g, active = np.full(space.dimension, -1), 0
                    for j in rng.permutation(space.dimension):
                        if active == space.max_order:
                            break
                        p = trails[j] ** self.alpha
                        p = (1 - self.exploration) * p / p.sum() + self.exploration / len(p)
                        g[j] = rng.choice(len(p), p=p) + space.lower_bound
                        active += int(g[j] >= 0)
                    g = space.repair(g, rng)
                ants.append(g)
                scores.append(session.evaluate(g))
            first = False
            for t in trails:
                t *= 1 - self.evaporation
            good = [i for i in np.argsort(scores)[::-1] if np.isfinite(scores[i])]
            for rank, i in enumerate(good[:max(1, len(good) // 3)]):
                for j, v in enumerate(ants[i]):
                    trails[j][v - space.lower_bound] += 1 / (rank + 1)
            for t in trails:
                np.clip(t, 0.05, 20, out=t)


class DiscretePSOReference:
    """Rounded continuous PSO on category codes: engineering reference ONLY.

    Arbitrary ordinal distances are a limitation. Never label this FairRec-DPSO.
    Native double-ended DPSO must be attached via NativeSearchAdapter.
    """
    name = "pso_reference"

    def __init__(self, population=20, inertia=0.7, cognitive=1.4, social=1.4):
        self.population = population
        self.inertia, self.cognitive, self.social = inertia, cognitive, social

    def run(self, session, rng, initial=None):
        space = session.oracle.space
        x = initial_population(space, rng, self.population, initial).astype(float)
        velocity = np.zeros_like(x)
        personal, values = x.copy(), np.full(len(x), -np.inf)
        while session.remaining:
            for i in range(len(x)):
                if not session.remaining:
                    break
                score = session.evaluate(x[i])
                if score > values[i]:
                    personal[i], values[i] = x[i].copy(), score
            best = personal[int(np.argmax(values))].copy()
            velocity = (self.inertia * velocity + self.cognitive * rng.random(x.shape) * (personal - x)
                        + self.social * rng.random(x.shape) * (best - x))
            velocity = np.clip(velocity, -2, 2)
            x = np.array([space.repair(g, rng) for g in x + velocity], dtype=float)


class SimulatedAnnealing:
    name = "sa_local"

    def __init__(self, start_temperature=0.1, end_temperature=0.001):
        if min(start_temperature, end_temperature) <= 0:
            raise ValueError("Temperatures must be positive")
        self.t0, self.t1 = start_temperature, end_temperature

    def run(self, session, rng, initial=None):
        space = session.oracle.space
        g = space.sample(rng) if initial is None else np.array(initial[0]).copy()
        value = session.evaluate(g)
        total = max(session.remaining, 1)
        while session.remaining:
            temperature = self.t0 * (self.t1 / self.t0) ** (1 - session.remaining / total)
            h = space.mutate(g, rng)
            trial = session.evaluate(h)
            if np.isfinite(trial) and (not np.isfinite(value) or trial >= value
                                      or rng.random() < np.exp((trial - value) / temperature)):
                g, value = h, trial


class TabuSearch:
    name = "tabu_local"

    def __init__(self, tenure=10, neighborhood=8):
        if min(tenure, neighborhood) < 1:
            raise ValueError("Positive tabu settings required")
        self.tenure, self.neighborhood = tenure, neighborhood

    def run(self, session, rng, initial=None):
        space = session.oracle.space
        g = space.sample(rng) if initial is None else np.array(initial[0]).copy()
        session.evaluate(g)
        tabu = [tuple(g)]
        while session.remaining:
            options = []
            for _ in range(min(self.neighborhood, session.remaining)):
                h = space.mutate(g, rng)
                previous_best = session.best
                value = session.evaluate(h)
                if tuple(h) not in tabu or value > previous_best:
                    options.append((value, h))
            g = max(options, key=lambda z: z[0])[1] if options else space.sample(rng)
            tabu.append(tuple(g))
            tabu = tabu[-self.tenure:]


class GreedyLocalSearch:
    """Best improving candidate from a sampled neighborhood, with no tabu/annealing.

    Use through the SAME FairAS replacement hook as SA and tabu to compare
    acceptance policies under a shared neighborhood rather than conflating them.
    """
    name = "greedy_local"

    def __init__(self, neighborhood=8):
        if neighborhood < 1:
            raise ValueError("Positive neighborhood sample count required")
        self.neighborhood = neighborhood

    def run(self, session, rng, initial=None):
        space = session.oracle.space
        g = space.sample(rng) if initial is None else np.array(initial[0]).copy()
        current = session.evaluate(g)
        while session.remaining:
            options = []
            for _ in range(min(self.neighborhood, session.remaining)):
                h = space.mutate(g, rng)
                options.append((session.evaluate(h), h))
            value, best = max(options, key=lambda z: z[0])
            if value > current:
                g, current = best, value


class HybridTestingFramework:
    """Global exploration and local refinement share ONE budget and archive.

    A proposed composition, not native FairAS. Select engines/fraction using
    validation experiments only. Both search ends are orchestrated by runner.
    """
    name = "hybrid"

    def __init__(self, global_search=None, local_search=None, global_fraction=0.65):
        if not 0 < global_fraction < 1:
            raise ValueError("global_fraction must lie in (0,1)")
        self.global_search = global_search or GeneticSearch()
        self.local_search = local_search or SimulatedAnnealing()
        self.global_fraction = global_fraction

    def run(self, session, rng, initial=None):
        original_budget = session.budget
        original_unique = session.unique_budget
        session.budget = len(session.logs) + max(1, int((original_budget - len(session.logs)) * self.global_fraction))
        if original_unique is not None:
            session.unique_budget = len(session.cache) + max(1, int((original_unique - len(session.cache)) * self.global_fraction))
        try:
            self.global_search.run(session, rng, initial)
        finally:
            session.budget = original_budget
            session.unique_budget = original_unique
        valid = sorted(((session.sign * getattr(v, session.objective_field), g)
                        for g, v in session.cache.items() if v.valid),
                       reverse=True)
        # Refine several distinct seeds rather than only the global maximum.
        seeds = [np.array(g) for _, g in valid[:5]] or [session.oracle.space.sample(rng)]
        for i, g in enumerate(seeds):
            if not session.remaining:
                break
            left = len(seeds) - i
            session.budget = len(session.logs) + max(1, (original_budget - len(session.logs)) // left)
            if original_unique is not None:
                session.unique_budget = len(session.cache) + max(1, (original_unique - len(session.cache)) // left)
            try:
                self.local_search.run(session, rng, [g])
            finally:
                session.budget = original_budget
                session.unique_budget = original_unique
