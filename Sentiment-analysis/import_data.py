'''
Step 1
import of the data and 1rst preprocessing to have a correct dataframe.
- importation(pathToFile, sep = ",")
- export_df(dataframe,path_name)
- run_import_data(nb_user = 5, path = 'data/merge_df.csv')
'''

import pandas as pd

def importation(pathToFile, sep = ","):

    '''
    function for importation of the data from the data file into the variables
    depending on the extension
    '''
    df = pd.read_csv(pathToFile, sep = sep)
    return df
