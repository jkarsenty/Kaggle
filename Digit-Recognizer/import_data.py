
"""
Ici on va definir les fonctions qui vont nous permettre d'importer des donnees

liste des fonctions :
* importation(pathToFile)

"""

import pandas as pd

def importation(pathToFile):

    data = pd.read_csv(pathToFile)
    return data
