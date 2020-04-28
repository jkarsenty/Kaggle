'''
Step 1
import of the data and 1rst preprocessing to have a correct dataframe.
- importation(pathToFile, sep = ",")
- export_df(dataframe,path_name)
- run_import_data(nb_user = 5, path = 'data/merge_df.csv')
'''

import pandas as pd
import json

def importation(pathToFile, format = 'csv', sep = ","):

    '''
    function for importation of the data from the data file into the variables
    depending on the extension
    '''
    print('Importation ' + pathToFile + ' en cours ...')
    if format == 'csv':
        df = pd.read_csv(pathToFile, sep = sep)
        print('Importation done')
        return df

    elif format == 'json':
        with open(pathToFile) as json_file:
                dict = json.load(json_file)
                print('Importation '+ pathToFile +' done')
                return dict

def export_df(dataframe,path_name):
    '''
    export of a dataframe into a csv file
    '''
    dataframe.to_csv (path_name, index = False, header=True)
    return

def export_json(dictionary, path_name):
    with open(path_name, 'w') as fp:
        json.dump(dictionary, fp, indent=1)
