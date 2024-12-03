import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
def create_advanced_placement_strategy(df, n=1850000, batch_size=10000, bandwidth_factor=0.05):
    print("Starting advanced batch placement strategy...")
    print(f"Total target installations: {n}")
    print(f"Maximum batch size: {batch_size}")
    placements = []
    panels_placed = []
    impact_scores = []
    carbon_offset_scores = [0.]
    energy_generation_scores = [0.]
    zips_placed =[]
    remaining_panels = n
    current_installs = np.zeros(len(df))
    coords = df[['latitude', 'longitude']].values
    max_installs = df['panel_install_limit'].values
    carbon_potential = df['carbon_offset_kg_per_panel'].values
    energy_potential = df['energy_generation_per_panel'].values
    income_component = (df['median_income'].max() - df['median_income']) / (df['median_income'].max() - df['median_income'].min())
    diversity_component = df['black/native_pop'] / (df['white_pop'] + df['asian_pop']+ 1e-6)
    coord_range = np.max([
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min()
    ])
    bandwidth = coord_range * bandwidth_factor
    print("\nInitial parameters:")
    print(f"Geographic bandwidth: {bandwidth:.2f}")
    print(f"Coordinate range: {coord_range:.2f}")
    spatial_impact = np.zeros(len(df))
    installed_density = np.zeros(len(df))
    iteration = 0
    pbar = tqdm(total=n, desc="Placing panels")
    while remaining_panels > 0:
        available_capacity = np.maximum(max_installs - current_installs, 0)
        if np.sum(available_capacity) == 0:
            print("\nNo more available capacity")
            break
        for j in range(len(df)):
            if available_capacity[j] == 0:
                continue
            dists = np.sqrt(np.sum((coords - coords[j]) ** 2, axis=1))
            geo_weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
            local_density = np.sum(current_installs * geo_weights) / np.sum(geo_weights)
            installed_density[j] = local_density
            local_income = np.average(income_component, weights=geo_weights)
            local_diversity = np.average(diversity_component, weights=geo_weights)
            environmental_score = carbon_potential[j] * energy_potential[j]
            dispersion_score = 1 / (local_density + 1)
            equity_score = (local_income + local_diversity) / 2
            spatial_impact[j] = (
                    environmental_score * 0.4 * (1 + dispersion_score * 0.2) +
                    equity_score * 0.4 * (1 + dispersion_score * 0.2) +
                    dispersion_score * 0.2
            )
        scores = spatial_impact * (available_capacity > 0)
        location = np.argmax(scores)
        batch = np.min([
            available_capacity[location],
            batch_size,
            remaining_panels
        ])
        for _ in range(round(float(batch))):
            current_carbon = carbon_potential[location]
            current_energy = energy_potential[location]
            carbon_offset_scores.append(current_carbon + carbon_offset_scores[-1])
            energy_generation_scores.append(current_energy + energy_generation_scores[-1])
            zips_placed.append(df['zip'].iloc[location])
        placements.append(df['zip'].iloc[location])
        panels_placed.append(int(batch))
        impact_scores.append(spatial_impact[location])
        current_installs[location] += batch
        remaining_panels = remaining_panels - batch
        pbar.update(int(batch))
        iteration += 1
        if iteration % 100 == 0:
            total_installed = n - remaining_panels
            print(f"\nProgress update:")
            print(f"Total panels placed: {total_installed:,} ({(total_installed / n * 100):.1f}%)")
            print(f"Unique locations used: {np.sum(current_installs > 0)}")
            print(f"Average batch size: {np.mean(panels_placed):.1f}")
            density_std = np.std(installed_density[installed_density > 0])
            bandwidth = max(coord_range * 0.05, bandwidth * (1 + density_std))
            print(f"Current bandwidth: {bandwidth:.2f}")
    pbar.close()
    final_metrics = {
        'placements': pd.Series(placements),
        'panels_per_placement': pd.Series(panels_placed),
        'impact_scores': pd.Series(impact_scores),
        'final_density': installed_density,
        'total_carbon': np.sum(carbon_potential * current_installs),
        'avg_equity': np.mean((income_component + diversity_component) * current_installs),
        'spatial_distribution': installed_density / np.sum(current_installs),
        'installation_counts': current_installs,
        'unique_locations': np.sum(current_installs > 0),
        'avg_batch_size': np.mean(panels_placed),
        'panels_placed': zips_placed,
        'carbon_offset_scores': carbon_offset_scores,
        'energy_generation_scores': energy_generation_scores
    }
    print("\nFinal Results:")
    print(f"Total Panels Placed: {n - remaining_panels:,}")
    print(f"Total Carbon Offset: {final_metrics['total_carbon']:,.0f} kg")
    print(f"Average Equity Score: {final_metrics['avg_equity']:.3f}")
    print(f"Unique Locations Used: {final_metrics['unique_locations']}")
    print(f"Average Batch Size: {final_metrics['avg_batch_size']:.1f}")
    # projections_carbon_offset_kg_per_panel = pd.read_csv("Clean_Data/projections_carbon_offset_kg_per_panel.csv")
    # projections_carbon_offset_kg_per_panel = projections_carbon_offset_kg_per_panel.drop(columns=['GWR'])
    # projections_carbon_offset_kg_per_panel.insert(6, "GWR", carbon_offset_scores, True)
    # projections_carbon_offset_kg_per_panel.to_csv("Clean_Data/projections_carbon_offset_kg_per_panel.csv", index=False)
    # projections_energy_generation_per_panel = pd.read_csv("Clean_Data/projections_energy_generation_per_panel.csv")
    # projections_energy_generation_per_panel = projections_energy_generation_per_panel.drop(columns=['GWR'])
    # projections_energy_generation_per_panel.insert(6, "GWR", energy_generation_scores, True)
    # projections_energy_generation_per_panel.to_csv("Clean_Data/projections_energy_generation_per_panel.csv", index=False)
    # projections_picked = pd.read_csv("Clean_Data/projections_picked.csv", sep=',', on_bad_lines='skip', index_col=False, dtype='unicode')
    zips_placed.append(zips_placed[-1])
    # projections_picked = projections_picked.drop(columns=['GWR'])
    # projections_picked.insert(5, "GWR", zips_placed, True)
    # projections_picked.to_csv("Clean_Data/projections_picked.csv", index=False)
    return final_metrics

def analyze_batch_results(metrics, df):
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 15))
    ax1 = plt.subplot(221)
    ax1.hist(metrics['panels_per_placement'], bins=50, alpha=0.7)
    ax1.set_title('Distribution of Batch Sizes')
    ax1.set_xlabel('Panels per Batch')
    ax1.set_ylabel('Frequency')
    ax2 = plt.subplot(222)
    scatter = ax2.scatter(
        df['longitude'],
        df['latitude'],
        c=metrics['installation_counts'],
        cmap='viridis',
        s=50,
        alpha=0.6
    )
    plt.colorbar(scatter, label='Total Panels Installed')
    ax2.set_title('Geographic Distribution of Installations')
    ax3 = plt.subplot(223)
    cumulative = np.cumsum(metrics['panels_per_placement'])
    ax3.plot(cumulative)
    ax3.set_title('Cumulative Panel Installations')
    ax3.set_xlabel('Placement Number')
    ax3.set_ylabel('Total Panels Installed')
    ax4 = plt.subplot(224)
    ax4.scatter(metrics['panels_per_placement'],
                metrics['impact_scores'],
                alpha=0.5)
    ax4.set_title('Impact Score vs Batch Size')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Impact Score')
    plt.tight_layout()
    plt.savefig('batch_placement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Loading data...")
    # df = pd.read_csv("Clean_Data/training_data.csv")
    # projections_picked = pd.read_csv("Clean_Data/projections_picked.csv", sep=',', on_bad_lines='skip', index_col=False, dtype='unicode')
    # metrics = create_advanced_placement_strategy(df, n=1850000, batch_size=10000, bandwidth_factor=0.05)
    # analyze_batch_results(metrics, df)