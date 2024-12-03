import pandas as pd
import numpy as np
import os


def create_fitness_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    fitness_df = pd.DataFrame()
    fitness_df['zip'] = df['region_name']
    # fitness_df['state'] = df['state_name']
    fitness_df['panel_install_limit'] = (df['count_qualified'] - df['existing_installs_count']).round()
    fitness_df['latitude'] = df['Latitude']
    fitness_df['longitude'] = df['Longitude']
    fitness_df['median_income'] = df['Median_income']
    fitness_df['white_pop'] = df['white_population']
    fitness_df['black/native_pop'] = df['black_population'] + df['native_population']
    fitness_df['asian_pop'] = df['asian_population']
    # fitness_df['native_pop'] = df['native_population']
    fitness_df['carbon_offset_kg_per_panel'] = df['carbon_offset_metric_tons_per_panel'] * 1000
    fitness_df['energy_generation_per_panel'] = df['yearly_sunlight_kwh_kw_threshold_avg'] * .4

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to CSV
    fitness_df.to_csv(output_file, index=False)

    print(f"File saved to: {output_file}")
    print(f"Shape of saved data: {fitness_df.shape}")

    return fitness_df


if __name__ == "__main__":
    input_file = "./Clean_Data/data_by_zip.csv"
    output_file = "./Clean_Data/training_data.csv"
    df = create_fitness_csv(input_file, output_file)