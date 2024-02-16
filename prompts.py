""" script for rough prompt analysis 
TODO
5 verschiedene prompts jeweils 100 triplets
gucken wie sich antworten ändern (wie sehr ist Antwort abhängig von Prompt)
für 48 objekte die Triplets sampeln um mit humans zu vergleichen/ Ähnlichkeitsmatrix zu kriegen
    
potentially put load_data and sim matrix in utils file 

"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import re

def load_data(dir): 
    """seed parameter from metadata refers to the ordering of the 
    triplets, which is 0 for current purposes
    loads data and adds a column for the temperature parameter"""

    dataframes = []
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    for i, file in enumerate(files): 
        full_path = os.path.join(dir, file)
        with open(full_path, 'rb') as f: 
                data = pickle.load(f)
                dataframe = pd.DataFrame(data['dataframe'])
        temperature = data['metadata']['temperature']
        dataframe['Temperature'] = temperature

        column_to_convert = 'gpt_image_ooo'
        dataframe[column_to_convert] = pd.to_numeric(dataframe[column_to_convert], errors='coerce').astype('Int64')

        matches = re.findall(r'\d+', file)
        prompt_number = int(matches[2])
        # batch = int(matches[3])
        name = f'Prompt_{prompt_number}' #_batch{batch}'
        dataframe.columns = pd.MultiIndex.from_product([[name], dataframe.columns])

        dataframes.append(dataframe)

    all_data = pd.concat(dataframes, axis=1)
    all_data = all_data.reindex(sorted(all_data.columns), axis=1)

    return all_data

def accuracy(df, column, comparison, save=False): 
    """
    compares column with comparison column within each sub-dataframe 
    """
    col_list = df.columns.get_level_values(0).unique()
    comparison_results = pd.DataFrame(index=[column, 'NaNs'], columns=col_list)
    for i in col_list:
        accuracy = (df[i][column].eq(df[i][comparison])).sum()
        nan = df[i][column].isna().sum()
        accuracy = accuracy / (len(df)-nan) * 100
        nan = nan / len(df) * 100 
        comparison_results.at[column, i] = accuracy
        comparison_results.at['NaNs', i] = nan

    sns.set_theme(style="whitegrid")
    df2 = comparison_results.reset_index().melt(id_vars='index', var_name='Prompt', value_name='Values')
    p = sns.catplot(data=df2, x='Prompt', y='Values', hue='index', kind='bar', 
                    palette='crest', height=6)
    # p.set_xticklabels(rotation=90)
    p.despine(left=True)
    p.set_axis_labels("", "Accuracy (corrected for NaNs) / NaNs in %")
    name = column.split('_')[1]
    p.fig.suptitle(f"{name} accuracies by prompt", y=1.02)
    # plt.text(x=0.8, y=0.05, s="[Accuracy corrected for NaNs]", ha='left', va='center', transform=plt.gcf().transFigure, fontsize=12, color='black')
    if save: p.savefig("/home/muellerk/gpt-thesis/gpt-behavior/analysis/prompts_sim_matrix.png")

def similarity(df, column1, column2, save=False):

    col_list = df.columns.get_level_values(0).unique()
    comparison_results = pd.DataFrame(index=col_list, columns=col_list)
    for i in col_list: 
        for j in col_list:
            similarity = (df[i][column1].eq(df[j][column2])).sum()
            similarity = similarity / len(data) * 100 
            comparison_results.at[i, j] = similarity
    comparison_matrix = comparison_results.astype(int)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid") 
    sns.heatmap(comparison_matrix, cmap='crest', annot=False, 
                linewidths=.5, xticklabels=col_list, yticklabels=col_list)

    name1 = column1.split('_')[1]
    name2 = column2.split('_')[1]
    plt.title(f"{name1} {name2} similarity matrix")
    plt.xlabel(column1, labelpad=20)
    plt.ylabel(column2, labelpad=20)
    if save: plt.savefig("/home/muellerk/gpt-thesis/gpt-behavior/analysis/prompts_sim_matrix.png")
    plt.show()

def nan_handling(df): 
    filtered_df = df.loc[(df['Prompt_1']['errors'] == 1) | (df['Prompt_2']['errors'] == 1)].copy()
    
    print((filtered_df['Prompt_1']['gpt_image_triplet_indices'].eq(filtered_df['Prompt_2']['gpt_image_triplet_indices'])).sum())
    nan_df = filtered_df.loc[(filtered_df['Prompt_1']['gpt_image_triplet_indices'].eq(filtered_df['Prompt_2']['gpt_image_triplet_indices']))]
    
    output_file_path = "/home/muellerk/gpt-thesis/gpt-behavior/nan_triplets.txt"
    with open(output_file_path, 'w') as output_file:
        for triplet in nan_df['Prompt_1']['gpt_image_triplet_indices']:
            output_file.write(' '.join(map(str, triplet)) + '\n')

path = "/home/muellerk/gpt-thesis/gpt-behavior/output/1ktriplets"
data = load_data(path)
nan_handling(data)

# accuracy(data, 'gpt_image_ooo', 'human_ooo_index')

# similarity(data, 'gpt_image_ooo', 'gpt_image_ooo')

