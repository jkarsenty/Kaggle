
'''Step Last'''

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def eval_metric(history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric_name.
    Training and validation metric are plotted for each epoch.
    Input:
        history: model training history
        metric_name: loss or accuracy
    Output:
        graph with epochs of x-axis and metric on y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_'+metric_name]

    NB_EPOCHS = len(metric)
    e = range(1, NB_EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train '+metric_name)
    plt.plot(e, val_metric, 'b', label='Validation '+metric_name)
    plt.legend()
    plt.show()

    return

def evaluation(x, y, model):
    '''
    fonction qui permet de tester l'apprentissage de notre modele "model".
    elle realise d'abord une prediction p des data x.
    ensuite elle renvoie la matrice de confusion et la precision de notre modele.
    en comparant p et y
    '''
    p = model.predict(x)

    #argmax permet d'avoir les labels au lieu de juste la proba de chaque classe
    y = y.argmax(axis = 1)
    p = p.argmax(axis = 1)
    p_acc = accuracy_score(y,p)
    conf_mat = confusion_matrix(y,p)

    return p, p_acc, conf_mat

def submission(output_file, y_pred):
    '''
    fonction pour soumettre les data a kaggle
    output_file = le chemin du fichier a soumettre
    '''
    ypred = y_pred.argmax(axis = 1)
    with open(output_file, 'w') as f :
        f.write('ImageId,Label\n')
        for i in range(len(ypred)) :
            f.write(''.join(str(i+1)+','+str(ypred[i])+'\n'))

    return
