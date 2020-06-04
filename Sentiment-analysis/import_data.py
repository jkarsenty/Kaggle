'''
Step 0
import of the data and 1rst preprocessing to have a correct dataframe.
- importation(pathToFile, format = 'csv', sep = ",")
- export_file(variable,path_name,format)
'''

import pandas as pd
import json

def importation(pathToFile, format = 'csv', sep = ","):

    '''
    function for importation of the data from the data file into the variables
    depending on the extension
    '''
    print('Importation ',pathToFile,' en cours ...')

    if format == 'csv':
        df = pd.read_csv(pathToFile, sep = sep)
        print('Importation done')
        return df

    elif format == 'json':
        with open(pathToFile) as json_file:
                dict = json.load(json_file)
                print('Importation ',pathToFile,'done')
                return dict

def export_file(variable, path_name, format):
    '''
    export a variable into a file of format csv or json
    '''
    if format not in ['csv','json']:
        print('ERROR format')

    else:
        if format == 'csv':
            variable.to_csv (path_name, index = False, header=True)

        elif format == 'json':
            with open(path_name, 'w') as fp:
                json.dump(variable, fp, indent=3)
        print('export done into '+ path_name)

    return

def export_df(dataframe,path_name):
    '''
    export of a dataframe into a csv file
    '''
    dataframe.to_csv (path_name, index = False, header=True)
    return

def export_json(dictionary, path_name):
    '''export a dict into a json file'''

    with open(path_name, 'w') as fp:
        json.dump(dictionary, fp, indent=1)
    return

def submission(output_file, test_df, selected_text_predict):
    '''
    fonction pour soumettre les data a kaggle
    output_file = le chemin du fichier a soumettre
    '''

    with open(output_file, 'w') as f :
        f.write('textID,selected_text\n')
        for i in range(len(test_df)) :
            f.write(''.join(str(test_df.textID[i])+','+str(selected_text_predict[i])+'\n'))
    return
