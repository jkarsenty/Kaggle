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

def export_df(dataframe,path_name):
    '''
    export of a dataframe into a csv file
    '''
    dataframe.to_csv (path_name, index = False, header=True)
    return

def run_import_data(nb_user = 5, path = 'data/merge_df.csv'):
    '''
    function to import and call in the other modules to run all the import_data process

    input:
        nb_user the number of users we want
        path where we want the merge file (output) to be

    output:
        a csv file
    '''

    ### preparation des Data ###
    '''
    We keep only the columns that matter for our problem that means for now only
    the 'user_id', 'order_id', 'product_id' and 'department_id'.
    And then only 'user_id', 'order_id' and department_id.
    '''

    N = nb_user
    orders =  importation("data/orders.csv")
    orders = orders[['user_id','order_id']]
    orders = orders[orders['user_id'].isin(range(N+1))]

    order_products = importation("data/order_products__prior.csv")
    order_products = order_products[['order_id','product_id']]

    products = importation("data/products.csv")
    products = products[['product_id','department_id']]

    #print(order_products.head())
    #print(products.columns)
    #print(products.head())

    ## Merge des dataset selon pour avoir un unique dataframe

    order_products = pd.merge(orders,order_products, on = 'order_id' )
    products_df = pd.merge(order_products, products, on = 'product_id')
    #print(products_df)

    ## Export df_merge ##
    p_df = products_df.drop('product_id', axis = 1)
    #p_df = p_df.sort_values('user_id')

    export_df(p_df,'data/merge_df.csv')

    #nb_user = products_df['user_id']
    #print(np.unique(nb_user))
    #data = products_df.groupby('user_id')
    #print(len(data))
    #print(products_df.head())
    #print(data.head())

    return
