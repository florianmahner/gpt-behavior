"""Script for a coarse temperature analysis for the gpt4 odd one out task. 

loads the data in the given directory, calculates NaNs in the gpt_image_ooo column, 
noise ceilings, and a similarity matrix for the different temperatures and runs 
currently temps 0, 1 and 2, and 5 runs each 
-> default temperature (1) is probably best 

TODO: 
    noise ceilings plotten

    idee: für consistency und accuracy ein paar mal sampeln mit verschiedenen triplets 
     
    hübsche Visualisierungen schonmal machen
    
    ggf für andere 100 triplets ausprobieren 
    """

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns

def load_data(): 
    """loads data into multiindex df"""

    dataframes = []
    dir = "/home/muellerk/gpt-thesis/gpt-behavior/output/temperature"
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    for i, file in enumerate(files): 
        full_path = os.path.join(dir, file)
        with open(full_path, 'rb') as f: 
                data = pickle.load(f)
                dataframe = pd.DataFrame(data['dataframe'])
        temperature = data['metadata']['temperature']
        matches = re.findall(r'\d+', file)
        batch = int(matches[2])
        # Add a column for the temperature parameter
        dataframe['Temperature'] = temperature
        name = f'Temp {temperature}, Run {batch}'
        dataframe.columns = pd.MultiIndex.from_product([[name], dataframe.columns])
        dataframes.append(dataframe)

    all_data = pd.concat(dataframes, axis=1)
    all_data = all_data.reindex(sorted(all_data.columns), axis=1)
    return all_data

def count_nans(df):
    nans = df.isna().groupby(level=0).sum().sum()
    return nans

def noise_ceilings(all_values): 
    """
    computes the most common answer per row (mode), once without the 
    sample in question (lower bound) and once including it (upper bound), 
    and compares to answer in question.
    NaN are not counted for mode.

    all_values: df of shape (100, x) 
    """   

    mode_upper = all_values.iloc[:, :].mode(axis=1, numeric_only=True)
    noise_ceilings = pd.DataFrame(columns=['lower', 'upper'])

    for col_name in all_values.iloc[:, :]:
        # count how many rows chose the most common answer
        noise_ceil_upper = (all_values[col_name].eq(mode_upper[0], axis=0)).sum()
        
        mode_lower = all_values.iloc[:, :].drop(columns=col_name).mode(axis=1, numeric_only=True)
        noise_ceil_lower = (all_values[col_name].eq(mode_lower[0], axis=0)).sum()

        noise_ceilings[col_name]['lower']=noise_ceil_lower
        noise_ceilings[col_name]['upper']=noise_ceil_upper

    return noise_ceilings

def similarity(df, column1, column2, save=False):
    
    col_list = df.columns.get_level_values(0).unique()
    comparison_results = pd.DataFrame(index=col_list, columns=col_list)
    for i in col_list: 
        for j in col_list:
            similarity = (df[i][column1].eq(df[j][column2])).sum()
            comparison_results.at[i, j] = similarity
    comparison_matrix = comparison_results.astype(int)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid") 
    sns.heatmap(comparison_matrix, cmap='crest', annot=False, 
                linewidths=.5, xticklabels=col_list, yticklabels=col_list)

    name1 = column1.split('_')[1]
    name2 = column2.split('_')[1]
    if name1 == name2:
        plt.title(f"{name1} similarity matrix")
    else: 
        plt.title(f"{name1} {name2} similarity matrix")
    plt.xlabel(column1, labelpad=20)
    plt.ylabel(column2, labelpad=20)
    if save: plt.savefig("/home/muellerk/gpt-thesis/gpt-behavior/analysis/temp_sim_matrix.png")
    plt.show()

def accuracy(df, img_column, word_column, comparison):
    """accuracies of the specified columns
    removes NaNs and calculates percentages
    """
    col_list = df.columns.get_level_values(0).unique()
    accuracies = pd.DataFrame(index=col_list, 
                              columns=['image_accuracy', 'word_accuracy', 
                                       'image_nans', 'word_nans', 'temperature'])

    for i in col_list:
        image_accuracy = (df[i][img_column].eq(df[i][comparison])).sum()
        word_accuracy = (df[i][word_column].eq(df[i][comparison])).sum()
        image_nans = df[i][img_column].isna().sum()
        word_nans = df[i][word_column].isna().sum()
        image_accuracy = image_accuracy / 100 * (100-image_nans) # so that accuracy in percent
        word_accuracy = word_accuracy / 100 * (100-word_nans)
        accuracies.at[i, 'image_accuracy'] = image_accuracy
        accuracies.at[i, 'word_accuracy'] = word_accuracy
        matches = re.findall(r'\d+', i)
        temp = matches[0]
        accuracies.at[i, 'temperature'] = temp
        accuracies.at[i, 'image_nans'] = image_nans
        accuracies.at[i, 'word_nans'] = word_nans

    # plot them 
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    for i, column in enumerate(accuracies.columns):
        if column != 'temperature':
            sns.barplot(x='temperature', y=column, data=accuracies, ax=axes[i%2][i//2], dodge=True, palette='crest')
            axes[i%2][i//2].set_title(column)
            axes[i%2][i//2].set_xlabel('Temperature')
            axes[i%2][i//2].set_ylabel(column)

    plt.tight_layout()
    plt.show()
    return accuracies

def consistency(dfs):
    # removed NaN for this
    consistency = pd.DataFrame(index=dfs.columns.levels[0], 
                              columns=['word_image_similarity', 'total_nans', 'temperature'])
    for df_name in dfs.columns.levels[0]:
        word_image_similarity = (dfs[df_name]['gpt_image_ooo'].eq(dfs[df_name]['gpt_word_ooo'])).sum()
        total_nans = dfs[df_name][['gpt_image_ooo', 'gpt_word_ooo']].isna().any(axis=1).sum()
        word_image_similarity = word_image_similarity / 100 * (100-total_nans)
        consistency.at[df_name, 'word_image_similarity'] = word_image_similarity
        temp = int(df_name.split('_')[1])
        consistency.at[df_name, 'temperature'] = temp
        consistency.at[df_name, 'total_nans'] = total_nans

    # plot them 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    for i, column in enumerate(consistency.columns):
        if column != 'temperature':
            sns.barplot(x='temperature', y=column, data=consistency, ax=axes[i], dodge=True, palette='crest')
            axes[i].set_title(column)
            axes[i].set_xlabel('Temperature')
            axes[i].set_ylabel(column)

    plt.tight_layout()
    plt.show()
    return consistency

all_data = load_data()

accuracy(all_data, 'gpt_image_ooo', 'gpt_word_ooo', 'human_ooo_index')
similarity(all_data, 'gpt_image_ooo', 'gpt_image_ooo')


all_values = all_data.xs(key='gpt_word_ooo', axis=1, level=1)
noise_ceilings_temp0 = noise_ceilings(all_values.iloc[:, :5], save=True)
noise_ceilings_temp1 = noise_ceilings(all_values.iloc[:, 5:10], save=True)
noise_ceilings_temp2 = noise_ceilings(all_values.iloc[:, 10:], save=True)

# TODO barplot of noiseceilings

