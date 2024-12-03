from plot_util import *
from data_load_util import *
from projections_util import *
from tqdm import tqdm
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

combined_df = Make_dataset(remove_outliers=True)
max_num_added = 1850000
Energy_projections, Energy_picked = create_projections(combined_df, n=max_num_added, load=False, metric='energy_generation_per_panel')
Carbon_offset_projections, Carbon_offset_picked = create_projections(combined_df, n=max_num_added, load=False, metric='carbon_offset_kg_per_panel')

panel_estimations_by_year = [("Net-Zero" , 479000 * 3), ("  2030  ", 479000 * 1), ("  2034  ", 479000 * 2)]

def plot_projections(projections, panel_estimations=None, net_zero_horizontal=False, interval=1, fontsize=30, fmts=["-X", "-H", "o-", "D-", "v-", "-8", "-p", "--"], upper_bound='Carbon-Efficient', ylabel=None):
    print(projections)
    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Explicitly specify which strategies to plot and in what order
    strategies_to_plot = [
        'Status-Quo',
        'Carbon-Efficient',
        'Energy-Efficient',
        'Racial-Equity-Aware',
        'Income-Equity-Aware',
        'Round Robin',
        'GWR'  # Added GWR strategy
    ]
    
    # Make sure we have enough format strings for all strategies
    if len(fmts) < len(strategies_to_plot):
        fmts.extend(['-'] * (len(strategies_to_plot) - len(fmts)))
    
    # Print debugging information
    print("Strategies to plot:", strategies_to_plot)
    print("Available columns:", projections.columns)
    print("Format strings:", fmts)
    
    # Downsample the data
    total_points = len(projections[strategies_to_plot[0]])
    downsample_factor = max(1, total_points // 1000)
    x = np.arange(0, total_points, downsample_factor) * interval

    # Plot each strategy
    for i, strategy in enumerate(strategies_to_plot):
        if strategy in projections.columns:
            y_data = np.array(projections[strategy])[::downsample_factor]
            plt.plot(x, y_data, fmts[i], label=strategy, linewidth=3, markersize=8, alpha=0.9)
            print(f"Plotted {strategy} with format {fmts[i]}")

    if panel_estimations is not None:
        for label, value in panel_estimations:
            plt.vlines(value, 
                      np.array(projections[upper_bound])[::downsample_factor][-1]/18, 
                      np.array(projections[upper_bound])[::downsample_factor][-1], 
                      colors='darkgray', linestyles='dashed', linewidth=2, alpha=0.7)
            plt.text(value - len(x)/23, 
                    np.array(projections[upper_bound])[::downsample_factor][-1]/80, 
                    label, alpha=0.7, fontsize=25)

    if net_zero_horizontal:
        two_mill_continued = np.array(projections['Status-Quo'])[479000 * 3]
        plt.hlines(two_mill_continued, 0, len(x) * interval, 
                  colors='black', linestyles='dashed', linewidth=2, alpha=0.5)
        plt.text(0, two_mill_continued*1.1, 
                "Continued trend at\nNet-zero prediction", 
                alpha=0.95, fontsize=18, color='black')

    plt.locator_params(axis='x', nbins=8) 
    plt.locator_params(axis='y', nbins=8) 
    plt.yticks(fontsize=fontsize/(1.2))
    plt.xticks(fontsize=fontsize/(1.2))
    
    plt.xlabel("Additional Panels Built", fontsize=fontsize, labelpad=20)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=20)
    plt.legend(fontsize=fontsize/1.5)
    
    plt.tight_layout()
    save_path = os.path.join(os.getcwd(), 'test.png')
    plt.savefig(save_path)
    plt.close()

# Plots a map of where the zip_codes picked are located
def plot_picked(combined_df, picked, metric, title=""):

    if metric is None:
        region_list = list(combined_df['region_name'])
        occurence_counts = picked.value_counts()
        times_picked = np.zeros_like(combined_df['region_name'])
        for pick in picked.unique():
            times_picked[region_list.index(pick)] += occurence_counts[pick]
        
        combined_df['times_picked'] = times_picked
        metric ='times_picked'

        dot_size_scale = (40 * times_picked[combined_df['times_picked']>0]/ (np.max(combined_df['times_picked'][combined_df['times_picked']>0]))) + 40     
    picked = picked.astype(str)

    geo_plot(combined_df['times_picked'][combined_df['times_picked']>0], color_scale='agsunset', title=title, edf=combined_df[combined_df['times_picked']>0], zipcodes=picked.unique(), colorbar_label="", size=dot_size_scale)

# Creates a DF with updated values of existing installs, carbon offset potential(along with per panel), and realized potential
# After a set of picks (zip codes with a panel placed in them)
def df_with_updated_picks(combined_df, picks, load=None, save=None):

    if load is not None and exists(load):
        return pd.read_csv(load)

    new_df = combined_df
    new_co = np.array(new_df['carbon_offset_metric_tons'])
    new_existing = np.array(new_df['existing_installs_count'])

    for pick in tqdm(picks):
        index = list(new_df['region_name']).index(pick)
        new_co[index] -= new_df['carbon_offset_metric_tons_per_panel'][index]
        new_existing[index] += 1
    
    print('carbon offset difference:', np.sum(new_df['carbon_offset_metric_tons'] - new_co))
    new_df['carbon_offset_metric_tons'] = new_co
    new_df['carbon_offset_kg'] = new_co * 1000
    print('Number install change:', np.sum(new_existing - new_df['existing_installs_count']) )
    new_df['existing_installs_count'] = new_existing
    new_df['existing_installs_count_per_capita'] = new_existing / new_df['Total_Population']
    new_df['panel_utilization'] = new_existing / new_df['number_of_panels_total']

    if save is not None:
        new_df.to_csv(save, index=False)

    return new_df

def plot_demo_state_stats(new_df,save="Clean_Data/data_by_state_proj.csv"):
    state_df = load_state_data(new_df, load=None, save=save)

    hatches=['o','o','o','o','o','x','x','x','x','x']
    annotate = False
    type = 'paper'
    stacked = False

    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop","Median_income", "asian_prop", "Republican_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income','Republican'] , type=type, stacked=stacked, ylabel="Realized Potential (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True)
    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="existing_installs_count_per_capita", type=type, stacked=stacked, ylabel="Existing Installs Per Capita (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True)
    bar_plot_demo_split(state_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop", "Republican_prop"], xticks=['Black', 'White', 'Asian', 'Income', 'Republican'], key="carbon_offset_kg", type=type, stacked=stacked, ylabel="Carbon Offset Potential (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True) 

    hatches=['o','o','o','o','x','x','x','x']

    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="panel_utilization", xticks=['Black', 'White', 'Asian','Income'] , type=type, stacked=stacked, ylabel="Realized Potential (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True) 
    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop","Median_income", "asian_prop"], key="carbon_offset_kg", xticks=['Black', 'White', 'Asian','Income'] , type=type, stacked=stacked, ylabel="Potential Carbon Offset (x Avg)", title="", hatches=hatches, annotate=annotate, legend=True)
    bar_plot_demo_split(new_df, demos=["black_prop", "white_prop", "Median_income", "asian_prop"], xticks=['Black', 'White', 'Asian', 'Income'], key="existing_installs_count_per_capita", type=type, stacked=stacked, ylabel="Existing Installs Per Capita (x Avg)", title="", hatches=hatches, annotate=annotate,  legend=True)

plot_projections(Carbon_offset_projections, panel_estimations_by_year, net_zero_horizontal=True, interval=100000, upper_bound='Carbon-Efficient', ylabel="Carbon Offset (kg)")
plot_projections(Energy_projections, panel_estimations_by_year, net_zero_horizontal=True, interval=100000, upper_bound='Energy-Efficient', ylabel="Additional Energy Capacity (kWh)")

# print(Energy_picked[''])

# for key in ['Energy-Efficient', 'Carbon-Efficient', 'Racial-Equity-Aware', 'Income-Equity-Aware', 'Round Robin']:
#     plot_picked(combined_df, Energy_picked[key], None, title="")


def weighted_proj_heatmap(combined_df, metric='carbon_offset_kg_per_panel', objectives=['carbon_offset_kg_per_panel', 'energy_generation_per_panel', 'black_prop']):
    weight_starts = [0.0, 0.0]
    weight_ends = [0.5,1.5]
    number_of_samples = 7
    weighted_proj_array = create_many_weighted(combined_df, n=1850000, objectives=objectives, weight_starts=weight_starts, weight_ends=weight_ends, number_of_samples=7, metric=metric,
                                               save='Projection_Data/weighted_map_5_energy', load='Projection_Data/weighted_map_5_energy.npy')


    ax = sns.heatmap(weighted_proj_array[:,:,-1], xticklabels=np.round(np.arange(weight_starts[0],weight_ends[0], (weight_ends[0] - weight_starts[0])/number_of_samples), 1), yticklabels=np.round(np.arange(weight_starts[1],weight_ends[1], (weight_ends[1] - weight_starts[1])/number_of_samples), 1))
    ax.set_xlabel("Energy Potential Weight")
    ax.set_ylabel("Black Prop Weight")
    plt.show()


# weighted_proj_heatmap(combined_df, metric='energy_generation_per_panel')

# quit()

# co_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Carbon Offset'], load='Projection_Data/df_greedy_co.csv', save='Projection_Data/df_greedy_co.csv')
# round_robin_df = df_with_updated_picks(combined_df, Energy_picked['Round Robin Policy'], load='Projection_Data/df_greedy_rrtest_rr.csv', save='Projection_Data/df_greedy_rrtest_rr.csv')
# energy_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Average Sun'], load='Projection_Data/df_greedy_sun.csv', save='Projection_Data/df_greedy_sun.csv')
# black_df = df_with_updated_picks(combined_df, Energy_picked['Greedy Black Proportion'], load='Projection_Data/df_greedy_black.csv', save='Projection_Data/df_greedy_black.csv')
# weighted_df = df_with_updated_picks(combined_df, Energy_picked['Weighted Greedy'], load='Projection_Data/df_greedy_weighted.csv', save='Projection_Data/df_greedy_weighted.csv')

# plot_demo_state_stats(round_robin_df, save="Projection_Data/data_by_state_proj_greedy_round_robink.csv")
# plot_demo_state_stats(energy_df, save="Projection_Data/data_by_state_proj_greedy_weighted.csv")