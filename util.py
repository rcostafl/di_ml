# Estruturas de dados
import numpy as np
import pandas as pd
pd.options.display.max_rows     = None
pd.options.display.max_columns  = 100
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_colwidth = 1000

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import OneHotEncoder

#from sklearn.metrics import plot_confusion_matrix 
#from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import confusion_matrix

# graphic
import matplotlib.pyplot as plt

# others
import re
import itertools   

df_di     = None
adicao_df = None

import gc


######################################################################################
# Objective:
#    - This function format the result dataset for display
# Params:
#   - result_df : the dataframe to be formated
#   - ds_name   : dataset name (original or another with some change)
# Returns:
#   - A dataframe containig the values for each metric of each model (classifier)
#######################################################################################
def organize_result_df(result_df, ds_name):
    # Seting multilevel index to indicate the dataframe_name
    ds_name_list    = [ds_name for x in range(0,result_df.index.values.size)]
    new_index_names = [x +"_mean" for x in result_df.index.values]
    index_tuple  = tuple(zip(ds_name_list, new_index_names))
    new_index    = pd.MultiIndex.from_tuples(index_tuple)
    result_df.set_index(new_index, inplace=True)
    
    return result_df

def get_di_sample(df, frac_di):
    # shuffle dataframe
    df = df.sample(frac=1)
    di_series = pd.Series(df['cd_di'].unique()).sample(frac=frac_di)
    df_sample = df[df['cd_di'].isin(di_series.values)]    
    return df_sample
    
    
######################################################################################
# Objective:
#    - This function use the dataset received as a parameter to trein somes ML models
#      and return the related metrics, so they can be compared
# Params:
#   - X      : independent variables 
#   - y      : dependent variable
#   - ds_name: dataset name
# Returns:
#   - A dataframe containig the values for each metric of each model (classifier)
#######################################################################################
def evaluate_models(ds_name, df, frac_di):
    
    # shuffle dataframe
    #df = df.sample(frac=1)
    #di_series = pd.Series(df['cd_di'].unique()).sample(frac=frac_di)
    #df_sample = df[df['cd_di'].isin(di_series.values)]
    
    df_sample = get_di_sample(df, frac_di)

    X = df_sample.drop(columns=['retificada', 'cd_di'], inplace=False)
    y = df_sample['retificada']


    # The classifiers that will be tested
    clf_dict = {
        'knn'    :KNeighborsClassifier(n_neighbors=5),
        'log_reg':LogisticRegression(max_iter=5000, solver='lbfgs', multi_class='auto'),
        'decision_tree': DecisionTreeClassifier(max_depth=2),
        'random_florest': RandomForestClassifier(max_depth=3, bootstrap=True)
    }

    # The scoring that will be tested
    scoring = ['accuracy','precision', 'recall', 'f1']

    # The dataframe that will store the result
    result_df = pd.DataFrame()

    # Iteration over the classifiers instantiated in the clf_dic to teste every one and get the result
    for clf_name in clf_dict.keys():

        # Getting the classifier instantiated in the clf_dict
        print(f'Testando Classifier: {clf_name}')
        clf = clf_dict[clf_name] 

        # The cross_validate returns a dic with the result for every scoring and the time to fit and to score
        # i.g: {
        #   'fit_time': array([70.53535438, 82.758564  , 77.34784317, 78.66951418]), 
        #   'score_time': array([58.67853379, 61.4980619 , 60.44315505, 63.22279215]), 
        #   'test_accuracy': array([0.89729683, 0.89565217, 0.89479122, 0.89255273]), 
        #   'test_precision': array([0.91385424, 0.90943445, 0.91027988, 0.90954379]), 
        #   'test_recall': array([0.92836179, 0.93102988, 0.9284952 , 0.9255603 ]), 
        #   'test_f1': array([0.92105089, 0.92010547, 0.91929732, 0.91748215])
        #}
        result_dict = cross_validate(clf, X, y, cv=3, scoring=scoring, return_train_score=True, n_jobs=8)

        # The result_dict is converted to a series by converting it to a DataFrame and calling the DataFrame.mean() function
        # i.g:
        # fit_time      	77.33 (mean)
        # score_time    	60.96 (mean)
        # test_accuracy 	0.90  (mean)
        # so on....
        result_mean_series = pd.DataFrame(result_dict).mean()

        # The result series is added to the result dataframe as a column
        result_df[clf_name] = result_mean_series

    # Organizing the result_dataframe: rename columns, create index, transpose
    result_df = organize_result_df(result_df, ds_name)
    
    return result_df


#####################################################################
# from svm_grid_cv_iris.py / pucminas
#####################################################################
def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                          title="Matriz de confusão",
                          cmap=plt.cm.Blues):
    """
    Esta função plota a matriz de confusão.
    Normalização pode ser feita usando o parâmetro 'normalize=True'.
    O mapa de cores pode ser alterado pelo parâmetro cmap. Default: 'cmap=plt.cm.Blues'.
    O parâmetro 'title' altera o título.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print("Matriz de confusão sem normalização")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe real')
    plt.xlabel('Classe prevista')
    
def plot_eval_graphic(eval_df):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(15, 5))

    fig.suptitle('Métricas', fontsize=20)

    models_names = eval_df.columns.values.tolist()

    ds_names = eval_df.index.get_level_values(0).unique().to_list()

    ds_names_format = {}
    
    for name in ds_names:
        idx_slicer = pd.IndexSlice

        # adding a line to axis 1
        line_data = eval_df.loc[idx_slicer[:,['test_precision_mean']],: ].reset_index().set_index('ds_name').drop(columns=['metric']).loc[name]
        ax1.plot(models_names, line_data)

        # adding line to axis 2
        line_data2 = eval_df.loc[idx_slicer[:,['test_accuracy_mean']],: ].reset_index().set_index('ds_name').drop(columns=['metric']).loc[name]
        ax2.plot(models_names, line_data2)

    ###################### AX1 ######################
    ax1.set_xticks(models_names)
    ax1.set_xticklabels(models_names, rotation=45)
    ax1.set_title('Precision')
    ax1.set_xlabel('Modelos')
    ax1.legend(ds_names)
    
    # Setting the max y values for each element in x
    ax1_max_values = eval_df.loc[idx_slicer[:,'test_precision_mean'],:].max().values
    for x,y in enumerate(ax1_max_values):
        ax1.text(x=(x-0.05), y=(y+0.015), s=round(y,3)) #x= x position, y = y position, s = string
        
    # Icreasing the y bound (outline size) to create room for the max value just inserted
    ax1_outline_size = max(ax1_max_values)*1.05 # it will be 10% bigger than the biggest value in the y axis
    ax1.set_ybound(lower=None, upper=ax1_outline_size)
    ax1.set_xbound(lower=None, upper=ax1.get_xbound()[1]*1.03)  

    
    ###################### AX2 ######################
    ax2.set_xticks(models_names)
    ax2.set_xticklabels(models_names, rotation=45)
    ax2.set_title('Accurary')
    ax2.set_xlabel('Modelos')
    ax2.legend(ds_names)
    
     # Setting the max y values for each element in x
    ax2_max_values = eval_df.loc[idx_slicer[:,'test_accuracy_mean'],:].max().values
    for x,y in enumerate(ax2_max_values):
        ax2.text(x=(x-0.05), y=(y+0.015), s=round(y,3)) #x= x position, y = y position, s = string
        
    # Icreasing the y bound (outline size) to create room for the max value just inserted
    ax2_outline_size = max(ax2_max_values)*1.05 # it will be 10% bigger than the biggest value in the y axis
    ax2.set_ybound(lower=None, upper=ax2_outline_size)    
    ax2.set_xbound(lower=None, upper=ax2.get_xbound()[1]*1.03)    


    plt.show()

# Params:
# - corr_list: correlation list
#   - shape: (nlines, 2cols)
#   - each line indicates the list of columns to make de correlation
#- df: the dataframe where the data is
#def plot_corr(corr_list, df):
 #   nrows = ceil(corr_list/3)
  #  fig, (axes) = plt.subplots(nrows=nrows, ncols=3, sharey=False, figsize=(15, 5))
   # plt.show()  
    
    