"""
Step 4 (final)
On definit ici les fonctions utiles pour tester et evaluer l'apprentissage
de nos data et realiser les predictions
liste des fonctions :
* evaluation(x, y, model)
"""

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


def evaluation(x, y, nb_categories, K_topscore, model):
    '''
    fonction qui permet de tester l'apprentissage de notre modele "model".
    elle realise d'abord une prediction p des data x.
    ensuite elle renvoie la matrice de confusion et la precision de notre modele.
    en comparant p et y
    Input :
        K_topscore les top score
    '''
    p = model.predict(x)

    #argsort permet d'avoir le classement au lieu de juste la proba de chaque classe
    #y = y.argsort(axis = 1)
    p = p.argsort(axis = 1)
    #print(p)

    t = nb_categories + 1
    for p_order in p:

        for c in range(t):
            if p_order[c] in [t-k for k in range(K_topscore)]:
                p_order[c] = 1
            else:
                p_order[c] = 0
        #print(p_order)

    p_acc = accuracy_score(y,p)
    #conf_mat = confusion_matrix(y,p)
    #print(p_acc)

    return p #,p_acc ,conf_ma
