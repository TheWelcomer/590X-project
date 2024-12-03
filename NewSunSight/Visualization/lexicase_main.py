import sys
import os
import pandas as pd
from plexicase.CBGP.problems import SolarOptimization
from plexicase.CBGP.run_lexicase import run_lexicase
from plexicase.CBGP.run_lexiprob import run_lexiprob
from data_load_util import *

total_panels_to_build = 1000000000
weights = {
    'income_equity': 1,
    'racial_equity': 1,
    'carbon_offset': 1,
    'energy_generation': 1,
    'geographic_equity': 1
}

def analyze_solution(individual, problem, combined_df):
    test_input = problem.test_cases[0]["input1"]
    bool_distribution = individual.program.eval(input1=test_input)
    bool_distribution = np.array(bool_distribution, dtype=bool)
    panel_distribution = problem.boolean_to_panels(bool_distribution)
    selected_zips = combined_df['zip'][bool_distribution].values
    metrics = {
        'income_equity': problem.calculate_income_equity(panel_distribution),
        'racial_equity': problem.calculate_racial_equity(panel_distribution),
        'carbon_offset': problem.calculate_carbon_offset(panel_distribution),
        'energy_generation': problem.calculate_energy_generation(panel_distribution),
        'geographic_equity': problem.calculate_geographic_equity(panel_distribution),
        'total_panels': panel_distribution.sum(),
        'selected_zips': selected_zips
    }

    return metrics


def genetic_algorithm(downsample_rate=1.0, alpha=1.0, max_generations=300, population_size=1, initial_genome_size=(10, 60)):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "Clean_Data", "training_data.csv")
    dataset = pd.read_csv(path)
    solarProblem = SolarOptimization(dataset, total_panels_to_build)
    top_10, best_error, logs = run_lexicase(solarProblem, downsample_rate, max_generations, population_size, initial_genome_size)
    solutions = []
    for i, individual in enumerate(top_10):
        print(f"\nAnalyzing solution {i + 1}:")
        metrics = analyze_solution(individual, solarProblem, dataset)
        solutions.append(metrics)
        print(f"Total panels: {metrics['total_panels']:,}")
        print(f"Income equity score: {metrics['income_equity']:.4f}")
        print(f"Racial equity score: {metrics['racial_equity']:.4f}")
        print(f"Carbon offset score: {metrics['carbon_offset']:.4f}")
        print(f"Energy generation score: {metrics['energy_generation']:.4f}")
        print(f"Geographic equity score: {metrics['geographic_equity']:.4f}")
        print(f"Number of selected zip codes: {len(metrics['selected_zips'])}")
    return solutions


if __name__ == "__main__":
    solutions = genetic_algorithm(max_generations=10, population_size=1000, initial_genome_size=(10, 60))