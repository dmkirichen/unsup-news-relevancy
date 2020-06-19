import numpy as np
import pandas as pd
import warnings
import os

# Visualization
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from IPython.utils import io

# Setting parameters
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


def countplot(x, hue, **kwargs):
    """
    Wrapper function for sns.countplot.
    """
    sns.countplot(x=x, hue=hue, **kwargs)


# Loading dataset
train_df = pd.read_csv('training.csv', index_col='id')

# Split features into input and output categories
predictor = train_df.drop('rank', axis=1)
target = train_df['rank']

# Saving all columns with number of unique values less than 31 as 'categorical' variables.
cat_cols = []  # categorical columns
cnt_cols = []  # continuous columns
for col in predictor.columns:
    if predictor[col].nunique() < 31:
        cat_cols.append(col)
    else:
        cnt_cols.append(col)

ignored_features = ['reactivity', 'poster_id', 'poster_is_lead', 'poster_order',
                    'owner_id', 'participant1_id', 'participant1_is_lead', 
                    'participant1_order', 'participant2_id', 
                    'participant2_is_lead', 'participant2_order', 
                    'participant3_id', 'participant3_is_lead', 
                    'participant3_order', 'participant4_id', 
                    'participant4_is_lead', 'participant4_order',
                    'participant5_id', 'participant5_is_lead', 
                    'participant5_order',]

predictor = predictor.drop(ignored_features, axis=1)

# fivethirtyeight color palette
pal_538 = ('#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b')
sns.set_palette(pal_538)

# Dictionary for the best selected paired crosses, which we analysed earlier.
combinations = {'app_type': ['is_liked_by_connections', 'number_of_comments', 
                             'participant1_is_employee', 
                             'participant1_is_in_connections',
                             'participant2_action', 'participant3_action'],
                
                'is_commented_by_connections': ['participant1_action', 'participant2_action', 
                                                'participant3_action', 'participant5_action', 
                                                'participant3_focus', 'participant4_focus'],
                
                'is_liked_by_connections': ['participant1_action', 'participant2_action',  
                                            'participant3_action', 'participant4_action',
                                            'participant5_action', 'participant1_focus', 
                                            'participant3_focus', 'participant4_focus', 
                                            'participant1_is_employee', 'participant2_is_employee', 
                                            'participant3_is_employee', 'participant4_is_employee'],
                
                'is_mentions_connections': ['is_commented_by_connections', 'participant4_focus',
                                            'participant1_action', 'participant2_action', 
                                            'participant3_action', 'participant3_is_employee',
                                            'participant4_is_employee'],
                
                'owner_type': ['number_of_comments', 'number_of_likes', 'owner_influence',
                               'participant2_action', 'participant5_action'],
                
                'poster_focus': ['owner_influence', 'poster_influence',
                                 'poster_is_employee', 'poster_is_in_connections',
                                 'participant1_focus',],
                
                'poster_gender': ['participant2_action', 'participant3_action', 
                                  'participant4_action'],
                
                'poster_is_employee': ['participant1_is_employee', 'participant2_is_employee', 
                                       'participant3_is_employee', 'poster_is_in_connections'],
                
                'poster_is_in_connections': ['poster_focus', 'poster_is_employee'],

                # Participant 1
                'participant1_action': ['number_of_comments', 'number_of_likes',
                                        'is_commented_by_connections', 'is_liked_by_connections', 
                                        'is_mentions_connections', 'participant3_is_employee',
                                        'participant4_is_employee', 'participant5_focus',
                                        'participant2_action', 'participant3_action',
                                        'participant3_influence', 'participant4_influence',
                                        'participant5_influence'],
                
                'participant1_focus': ['number_of_comments', 'number_of_likes', 
                                       'is_commented_by_connections', 'is_liked_by_connections',
                                       'participant2_action', 'participant3_action', 
                                       'participant1_influence', 'participant3_influence',
                                       'participant1_is_employee', 'participant2_is_employee', 
                                       'participant3_is_employee','participant3_focus', 
                                       'participant4_focus'],
                 
                'participant1_gender': ['number_of_likes', 'owner_influence', 'participant4_focus',
                                        'participant5_focus', 'participant1_action', 'participant2_action',
                                        'participant3_action', 'participant4_action', 'participant3_is_employee',
                                        'poster_gender'],
                
                'participant1_is_employee': ['is_commented_by_connections', 'is_liked_by_connections',
                                             'number_of_likes', 'participant2_action', 
                                             'participant3_action', 'participant1_focus', 
                                             'participant3_focus', 'participant4_focus',
                                             'participant1_influence', 'participant2_influence',
                                             'participant3_influence', 'participant4_influence', 
                                             'participant5_influence', 'participant2_is_employee', 
                                             'participant3_is_employee'],
                
                'participant1_is_in_connections': ['app_type', 'number_of_comments',
                                                   'participant2_action', 'participant3_action',
                                                   'participant1_focus', 'participant1_is_employee',
                                                   'participant3_is_employee'],

                # Participant 2
                'participant2_action': ['is_commented_by_connections', 'is_liked_by_connections',
                                        'is_mentions_connections', 'number_of_comments', 'number_of_likes',
                                        'participant1_action', 'participant3_action',
                                        'participant4_action', 'participant5_action',
                                        'participant1_focus', 'participant3_focus', 
                                        'participant4_focus', 'participant2_influence', 
                                        'participant3_influence', 'participant4_influence', 
                                        'participant5_influence', 'participant1_is_employee', 
                                        'participant2_is_employee', 'participant3_is_employee', 
                                        'participant4_is_employee', 'participant5_is_employee'],
                
                'participant2_focus': ['number_of_comments', 'number_of_likes',
                                       'participant2_action', 'participant2_influence', 
                                       'participant3_influence', 'participant4_influence',
                                       'participant5_influence', 'participant2_is_employee', 
                                       'participant4_focus', 'participant5_focus'],
                
                'participant2_gender': ['number_of_likes', 'participant3_action',
                                        'participant4_action', 'participant5_action',
                                        'participant2_influence', 'participant3_influence',
                                        'participant4_influence', 'participant5_influence', 
                                        'participant2_is_employee', 'participant3_is_employee',
                                        'participant4_is_employee'],
                
                'participant2_is_employee': ['is_commented_by_connections', 'is_liked_by_connections',
                                             'number_of_comments', 'number_of_likes',
                                             'participant2_action', 'participant3_action',
                                             'participant5_action', 'participant1_focus',
                                             'participant4_focus', 'participant2_influence', 
                                             'participant3_influence', 'participant4_influence', 
                                             'participant5_influence', 'participant3_is_employee', 
                                             'participant4_is_employee'],
                
                'participant2_is_in_connections': ['number_of_comments', 'number_of_likes',
                                                   'participant2_action', 'participant3_action',
                                                   'participant4_action', 'participant5_action',
                                                   'participant2_influence', 'participant3_influence',
                                                   'participant4_influence', 'participant5_influence',
                                                   'participant3_is_employee', 'participant4_is_employee'],
                
                # Participant 3
                'participant3_action': ['is_commented_by_connections', 'is_liked_by_connections',
                                        'number_of_comments', 'number_of_likes',
                                        'participant1_action', 'participant2_action',
                                        'participant4_action', 'participant5_action',
                                        'participant3_focus', 'participant4_focus',
                                        'participant2_influence', 'participant3_influence',
                                        'participant4_influence', 'participant5_influence',
                                        'participant1_is_employee', 'participant2_is_employee',
                                        'participant3_is_employee','participant4_is_employee',
                                        'participant5_is_employee'],
                
                'participant3_focus': ['is_commented_by_connections', 'is_liked_by_connections',
                                       'number_of_comments', 'number_of_likes',
                                       'participant1_action', 'participant2_action',
                                       'participant4_action', 'participant1_focus',
                                       'participant3_influence', 'participant4_influence',
                                       'participant5_influence', 'participant1_is_employee',
                                       'participant2_is_employee', 'participant3_is_employee'],
                
                'participant3_gender': ['number_of_comments', 'number_of_likes',
                                        'participant2_action', 'participant3_action',
                                        'participant4_action', 'participant5_action',
                                        'participant2_influence', 'participant3_influence',
                                        'participant4_influence', 'participant5_influence',
                                        'participant3_is_employee', 'participant4_is_employee',
                                        'participant5_is_employee'],
                
                'participant3_is_employee': ['is_commented_by_connections', 'is_liked_by_connections',
                                             'number_of_comments', 'number_of_likes',
                                             'participant2_action', 'participant3_action',
                                             'participant4_action', 'participant5_action',
                                             'participant3_focus', 'participant4_focus',
                                             'participant2_influence', 'participant3_influence',
                                             'participant4_influence', 'participant5_influence',
                                             'participant2_is_employee', 'participant4_is_employee',
                                             'participant5_is_employee'],
                
                'participant3_is_in_connections': ['number_of_comments', 'number_of_likes',
                                                   'participant2_action', 'participant3_action',
                                                   'participant4_action', 'participant5_action',
                                                   'participant2_influence', 'participant3_influence',
                                                   'participant4_influence', 'participant5_influence',
                                                   'participant4_is_employee', 'participant5_is_employee'],
               
                # Participant 4
                'participant4_action': ['is_commented_by_connections', 'is_liked_by_connections',
                                        'number_of_comments', 'number_of_likes',
                                        'participant1_action', 'participant2_action',
                                        'participant3_action', 'participant5_action',
                                        'participant1_focus', 'participant3_focus', 
                                        'participant4_focus', 'participant2_influence',
                                        'participant3_influence', 'participant4_influence',
                                        'participant5_influence', 'participant1_is_employee',
                                        'participant2_is_employee', 'participant3_is_employee',
                                        'participant4_is_employee', 'participant5_is_employee'],
                
                'participant4_focus': ['is_commented_by_connections', 'is_liked_by_connections',
                                       'number_of_comments', 'number_of_likes',
                                       'participant2_action', 'participant3_action',
                                       'participant4_action', 'participant5_action',
                                       'participant1_focus', 'participant2_influence', 
                                       'participant3_influence', 'participant4_influence',
                                       'participant5_influence', 'participant1_is_employee',
                                       'participant2_is_employee', 'participant3_is_employee',
                                       'participant4_is_employee'],
                
                'participant4_gender': ['number_of_comments', 'number_of_likes',
                                        'participant3_action', 'participant4_action',
                                        'participant5_action', 'participant2_influence',
                                        'participant3_influence', 'participant4_influence',
                                        'participant5_influence', 'participant3_is_employee',
                                        'participant4_is_employee', 'participant5_is_employee',
                                        'participant4_focus'],
                
                'participant4_is_employee': ['is_commented_by_connections', 'is_liked_by_connections',
                                             'number_of_comments', 'number_of_likes',
                                             'participant2_action', 'participant3_action',
                                             'participant4_action', 'participant5_action',
                                             'participant2_influence', 'participant3_influence',
                                             'participant4_influence', 'participant5_influence',
                                             'participant2_is_employee', 'participant3_is_employee',
                                             'participant5_is_employee', 'participant4_focus'],
                
                'participant4_is_in_connections': ['number_of_comments', 'number_of_likes',
                                                   'participant3_action', 'participant4_action',
                                                   'participant5_action', 'participant2_influence',
                                                   'participant3_influence', 'participant4_influence',
                                                   'participant5_influence', 'participant3_is_employee',
                                                   'participant4_is_employee', 'participant5_is_employee'],
                
                # Participant 5
                'participant5_action': ['is_commented_by_connections', 'is_liked_by_connections',
                                        'number_of_comments', 'number_of_likes',
                                        'participant2_action', 'participant3_action',
                                        'participant4_action', 'participant3_focus',
                                        'participant4_focus', 'participant5_focus',
                                        'participant2_influence', 'participant3_influence',
                                        'participant4_influence', 'participant5_influence',
                                        'participant2_is_employee', 'participant3_is_employee',
                                        'participant4_is_employee', 'participant5_is_employee'],
                
                'participant5_focus': ['number_of_comments', 'number_of_likes',
                                       'participant1_action', 'participant3_action',
                                       'participant2_influence', 'participant3_influence',
                                       'participant4_influence', 'participant5_influence',
                                       'participant1_is_employee', 'participant3_is_employee',
                                       'participant2_focus', 'participant4_focus'],
                
                'participant5_gender': ['number_of_comments', 'number_of_likes',
                                        'participant3_action', 'participant4_action',
                                        'participant5_action', 'participant2_influence',
                                        'participant3_influence', 'participant4_influence',
                                        'participant5_influence', 'participant3_is_employee',
                                        'participant4_is_employee', 'participant5_is_employee'],
                
                'participant5_is_employee': ['is_commented_by_connections', 'number_of_comments',
                                             'number_of_likes', 'participant2_action',
                                             'participant3_action', 'participant4_action',
                                             'participant5_action', 'participant2_influence',
                                             'participant3_influence', 'participant4_influence',
                                             'participant5_influence', 'participant2_is_employee',
                                             'participant3_is_employee', 'participant4_is_employee',
                                             'participant4_focus', 'participant5_focus'],
                
                'participant5_is_in_connections': ['number_of_comments', 'number_of_likes',
                                                   'participant3_action', 'participant4_action',
                                                   'participant5_action', 'participant2_influence', 
                                                   'participant3_influence', 'participant4_influence',
                                                   'participant5_influence', 'participant3_is_employee',
                                                   'participant4_is_employee', 'participant5_is_employee',
                                                   'participant4_focus']}

# Generating multivariate plots
counter = 0

for key in combinations.keys():
    os.makedirs('eda_results/multivariate/{}'.format(key), exist_ok=True)
    
    possible_col_i = [col for col in combinations[key] if col in cat_cols]  # possible pair crosses with 'key' feature
    for col_i in possible_col_i:
        possible_col_j = combinations[col_i]  # possible pair crosses with 'col_i' feature
        for col_j in possible_col_j:
            if key == col_j:
                continue
            
            # debug purposes
            counter += 1
            print("{}) {} - {} - {}".format(counter, key, col_i, col_j))
            
            g = sns.FacetGrid(predictor, col=key, size=8)
            if col_j in cat_cols:
                g.map(countplot, col_i, col_j, palette=pal_538)
            else:
                g.map(sns.barplot, col_i, col_j, palette=pal_538)
            g.add_legend()
            
            plt.savefig('eda_results/multivariate/{0}/{0}_x_{1}_x_{2}.png'\
                        .format(key, col_i, col_j))
            plt.close()
