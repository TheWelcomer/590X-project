from gwr import create_advanced_placement_strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from os.path import exists
import os

# Creates a projection of carbon offset if the current ratio of panel locations remain the same 
# allowing partial placement of panels in zips and not accounting in the filling of zip codes.
def create_continued_projection(combined_df, n=1000, metric='carbon_offset_metric_tons'):
    total_panels = np.sum(combined_df['existing_installs_count'])
    # print("total, current existing panels:", total_panels)
    panel_percentage = combined_df['existing_installs_count'] / total_panels
    ratiod_carbon_offset_per_panel = np.sum(panel_percentage * combined_df[metric])
    return np.arange(n+1) * ratiod_carbon_offset_per_panel

# Greedily adds 1-> n solar panels to zips which maximize the sort_by metric until no more can be added
# Returns the Carbon offset for each amount of panels added
def create_greedy_projection(combined_df, n=1000, sort_by='carbon_offset_metric_tons_per_panel', ascending=False, metric='carbon_offset_metric_tons_per_panel', record=True):
    sorted_combined_df = combined_df.sort_values(sort_by, ascending=ascending, inplace=False, ignore_index=True)
    projection = np.zeros(n+1)
    greedy_best_not_filled_index = 0
    existing_count = sorted_combined_df['existing_installs_count'][greedy_best_not_filled_index]
    i = 0

    if record:
        picked = [sorted_combined_df['region_name'][greedy_best_not_filled_index]]

    while (i < n):
        if existing_count >= sorted_combined_df['count_qualified'][greedy_best_not_filled_index]:
            greedy_best_not_filled_index += 1
            existing_count = sorted_combined_df['existing_installs_count'][greedy_best_not_filled_index]

        else:
            projection[i+1] = projection[i] + sorted_combined_df[metric][greedy_best_not_filled_index]
            existing_count += 1
            i += 1
            if record:
                picked.append(sorted_combined_df['region_name'][greedy_best_not_filled_index])
    
    return projection, picked

# Creates a projection which decides each placement alternating between different policies
def create_round_robin_projection(projection_list, picked_list):
    n = len(projection_list[0])
    number_of_projections = len(projection_list)
    projection = np.zeros(n)
    picked = [picked_list[0][0]]
    for i in range(1, n):
        chosen_projection = projection_list[i % number_of_projections]
        projection[i] = projection[i-1] + (chosen_projection[i] - chosen_projection[i-1])
        picked.append(picked_list[i % number_of_projections][i])
    return projection, picked

# Creates the projection of a policy which weighs multiple different factors (objectives)
# and greedily chooses zips based on the weighted total of proportions to national avg. 
def create_weighted_proj(combined_df, n=1000, objectives=['carbon_offset_metric_tons_per_panel'], weights=[1], metric='carbon_offset_metric_tons_per_panel'):

    new_df = combined_df
    new_df['weighted_combo_metric'] = combined_df[objectives[0]] * 0

    for weight, obj in zip(weights,objectives):
        new_df['weighted_combo_metric'] = new_df['weighted_combo_metric'] + (combined_df[obj] / np.mean(combined_df[obj])) * weight

    return create_greedy_projection(combined_df=new_df, n=n, sort_by='weighted_combo_metric', metric=metric)

# Creates a projection of the carbon offset if we place panels to normalize the panel utilization along the given "demographic"
# I.e. if we no correlation between the demographic and the panel utilization and only fous on that, how Carbon would we offset
# TODO
def create_pop_demo_normalizing_projection(combined_df, n=1000, demographic="black_prop", metric='carbon_offset_metric_tons_per_panel'):
    pass

# Creates a projection of carbon offset for adding solar panels to random zipcodes
# The zipcode is randomly chosen for each panel, up to n panels
def create_random_proj(combined_df, n=1000, metric='carbon_offset_metric_tons_per_panel'):
    projection = np.zeros(n+1)
    picks = np.random.randint(0, len(combined_df['region_name']) -1, (n))
    for i, pick in enumerate(picks):

        while math.isnan(combined_df[metric][pick]):
            pick = np.random.randint(0, len(combined_df[metric]))
        projection[i+1] = projection[i] + combined_df[metric][pick]

    return projection

# Creates multiple different projections and returns them
def create_projections(combined_df, n=1000, load=False, metric='carbon_offset_metric_tons_per_panel', save=True):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Define paths relative to script directory
    CLEAN_DATA_DIR = os.path.join(SCRIPT_DIR, 'Clean_Data')

    # When loading the file, use the absolute path
    train_df = pd.read_csv(os.path.join(CLEAN_DATA_DIR, 'training_data.csv'))
    print("Available columns:", combined_df.columns.tolist())

    ## TODO remove rrtest (just for a new version of round robin)
    # if load and exists("Clean_Data/projections_"+metric+".csv") and exists("Clean_Data/projections_picked.csv"):
    #     print("Loading Projections")
    #     print(pd.read_csv("Clean_Data/projections_"+metric+".csv").columns)
    #     return pd.read_csv("Clean_Data/projections_"+metric+".csv"), pd.read_csv("Clean_Data/projections_picked.csv")
    
    picked = pd.DataFrame()
    proj = pd.DataFrame()
    print("Creating Continued Projection")
    proj['Status-Quo'] = create_continued_projection(combined_df, n, metric)
    print("Creating Greedy Carbon Offset Projection")
    proj['Carbon-Efficient'], picked['Carbon-Efficient'] = create_greedy_projection(combined_df, n, sort_by='carbon_offset_metric_tons_per_panel', metric=metric)
    print("Creating Greedy Average Sun Projection")
    proj['Energy-Efficient'], picked['Energy-Efficient'] = create_greedy_projection(combined_df, n, sort_by='yearly_sunlight_kwh_kw_threshold_avg', metric=metric)
    print("Creating Greedy Black Proportion Projection")
    proj['Racial-Equity-Aware'], picked['Racial-Equity-Aware'] = create_greedy_projection(combined_df, n, sort_by='black_prop', metric=metric)
    print("Creating Greedy Low Median Income Projection")
    proj['Income-Equity-Aware'], picked['Income-Equity-Aware'] = create_greedy_projection(combined_df, n, sort_by='Median_income', ascending=True, metric=metric)

    print("Creating Round Robin Projection")
    proj['Round Robin'], picked['Round Robin'] = create_round_robin_projection(projection_list=
                                                                                                   [proj['Carbon-Efficient'], proj['Energy-Efficient'], proj['Racial-Equity-Aware'], proj['Income-Equity-Aware']],
                                                                                                   picked_list=
                                                                                                   [picked['Carbon-Efficient'], picked['Energy-Efficient'], picked['Racial-Equity-Aware'], picked['Income-Equity-Aware']])
    metrics = create_advanced_placement_strategy(train_df, batch_size=10000)
    if metric == 'carbon_offset_metric_tons_per_panel':
        proj['GWR'] = metrics['carbon_offset_scores']
        picked['GWR'] = metrics['panels_placed']
    if metric == 'energy_generation_per_panel':
        proj['GWR'] = metrics['energy_generation_scores']
        picked['GWR'] = metrics['panels_placed']

    # print("Creating Weighted Greedy Projection")
    # proj['Weighted Greedy'], picked['Weighted Greedy'] = create_weighted_proj(combined_df, n, ['carbon_offset_metric_tons_per_panel', 'yearly_sunlight_kwh_kw_threshold_avg', 'black_prop'], [2,4,1], metric=metric)


    # uniform_samples = 10

    # print("Creating uniform random projection with", uniform_samples, "samples")

    # proj['Uniform Random (' + str(uniform_samples) + ' samples)' ] = np.zeros(n+1)
    # for i in range(uniform_samples):
    #     proj['Uniform Random (' + str(uniform_samples) + ' samples)' ] += create_random_proj(combined_df, n)/uniform_samples
    
    ## TODO remove rrtest (just for a new version of round robin)
    print("saving projections")
    print(proj.columns)
    print(picked.columns)
    proj.to_csv("Clean_Data/projections_"+metric+".csv",index=False)
    picked.to_csv("Clean_Data/projections_picked.csv", index=False)

    return proj, picked

# Searches over many different weight settings, with the first weight being set permenantly to 1 and the other two being set proportionally
# Returns a 2d array of projections (i.e. 3d array)
def create_many_weighted(combined_df, n=1000, objectives=['carbon_offset_metric_tons_per_panel'], weight_starts=[], weight_ends=[], number_of_samples=1, metric='carbon_offset_metric_tons_per_panel', save=None, load=None):

    if exists(load):
       return np.load(load)

    all_projections = np.zeros((number_of_samples,number_of_samples,n+1))

    for i, weight1 in enumerate(np.arange(weight_starts[0], weight_ends[0], (weight_ends[0] - weight_starts[0]) / number_of_samples)):
        for j, weight2 in enumerate(np.arange(weight_starts[1], weight_ends[1], (weight_ends[1] - weight_starts[1]) / number_of_samples)):

            print("weighted proj number:", (i*number_of_samples + j))
            
            all_projections[i][j],_ = create_weighted_proj(combined_df, n=n, objectives=objectives, weights=[1, weight1, weight2], metric=metric)
    

    if save is not None:
        np.save(save, all_projections)

    return all_projections

def plot_combined_comparison(projections, gwr_metrics, panel_estimations_by_year=None):
    """Create comprehensive comparison visualization"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 15))

    # 1. Carbon Offset Comparison (top left)
    ax1 = plt.subplot(221)
    for strategy, data in projections.items():
        if strategy != 'GWR-Batch':
            ax1.plot(data.values, label=strategy, alpha=0.7)
    # Plot GWR strategy last to make it stand out
    ax1.plot(projections['GWR-Batch'].values, label='GWR-Batch',
             linewidth=3, linestyle='--')

    if panel_estimations_by_year:
        for label, value in panel_estimations_by_year:
            ax1.axvline(x=value, color='gray', linestyle=':', alpha=0.5)
            ax1.text(value, ax1.get_ylim()[0], label, rotation=90)

    ax1.set_title('Cumulative Carbon Offset')
    ax1.set_xlabel('Panels Installed')
    ax1.set_ylabel('Carbon Offset (kg)')
    ax1.legend(fontsize=8)

    # 2. Strategy Performance Comparison (top right)
    ax2 = plt.subplot(222)
    final_values = {k: v.values[-1] for k, v in projections.items()}
    strategies = list(final_values.keys())
    values = list(final_values.values())
    normalized_values = np.array(values) / max(values)

    bars = ax2.bar(strategies, normalized_values)
    ax2.set_title('Relative Strategy Performance')
    ax2.set_ylabel('Normalized Carbon Offset')
    plt.xticks(rotation=45)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val / 1e6:.1f}M',
                 ha='center', va='bottom', rotation=0)

    # 3. Convergence Rate (bottom left)
    ax3 = plt.subplot(223)
    for strategy, data in projections.items():
        if strategy != 'Status-Quo':  # Exclude baseline
            # Calculate percentage of final value
            normalized = data.values / data.values[-1] * 100
            ax3.plot(normalized, label=strategy, alpha=0.7)

    ax3.set_title('Strategy Convergence Rate')
    ax3.set_xlabel('Panels Installed')
    ax3.set_ylabel('Percentage of Final Value')
    ax3.legend(fontsize=8)

    # 4. Improvement Over Status-Quo (bottom right)
    ax4 = plt.subplot(224)
    baseline = projections['Status-Quo'].values
    improvements = {
        k: (v.values[-1] / baseline[-1] - 1) * 100
        for k, v in projections.items()
        if k != 'Status-Quo'
    }

    bars = ax4.bar(improvements.keys(), improvements.values())
    ax4.set_title('Improvement Over Status-Quo')
    ax4.set_ylabel('Percentage Improvement')
    plt.xticks(rotation=45)

    # Add value labels
    for bar, val in zip(bars, improvements.values()):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1f}%',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed comparison
    print("\nDetailed Strategy Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Carbon Offset (M kg)':<20} {'vs Status-Quo':<15} {'vs Best Naive':<15}")
    print("-" * 80)

    best_naive = max(v.values[-1] for k, v in projections.items()
                     if k not in ['Status-Quo', 'GWR-Batch'])

    for strategy, data in projections.items():
        final_val = data.values[-1]
        vs_baseline = (final_val / baseline[-1] - 1) * 100
        vs_best = (final_val / best_naive - 1) * 100 if strategy != 'Status-Quo' else 0

        print(f"{strategy:<20} {final_val / 1e6:>19.1f} {vs_baseline:>14.1f}% "
              f"{vs_best:>14.1f}%")


# Example usage:
if __name__ == "__main__":
    df = pd.read_csv("Clean_Data/data_by_zip.csv")

    panel_estimates = [
        ("2030", 479000 * 1),
        ("2034", 479000 * 2),
        ("Net-Zero", 479000 * 3),
    ]

    projections, picked, gwr_metrics = create_and_compare_strategies(
        df, n=1850000, batch_size=1000
    )

    # Plot the comparison with your existing plot_projections function
    plot_projections(
        projections,
        panel_estimations=panel_estimates,
        net_zero_horizontal=True,
        interval=100000,
        ylabel="Carbon Offset (kg)"
    )