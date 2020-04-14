"""
Step 4 (final)
On definit ici les fonctions utiles pour tester et evaluer l'apprentissage
de nos data et realiser les predictions
liste des fonctions :
* evaluation(x, y, model)
"""

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


def evaluation(x, y, model):
    '''
    fonction qui permet de tester l'apprentissage de notre modele "model".
    elle realise d'abord une prediction p des data x.
    ensuite elle renvoie la matrice de confusion et la precision de notre modele.
    en comparant p et y
    '''
    p = model.predict(x)

    #argmax permet d'avoir les labels au lieu de juste la proba de chaque classe
    #y = y.argmax(axis = 1)
    #p = p.argmax(axis = 1)
    p_acc = accuracy_score(y,p)
    conf_mat = confusion_matrix(y,p)

    return p, p_acc, conf_mat
