import matplotlib.pyplot as plt
import numpy as np
import pickle


"""
Plot the top-weighted features for classifcation.
"""


def plot_coefficients(classifier, model_name, feature_set, feature_names, top_features=20):
    if top_features > len(feature_names):
        top_features = len(feature_names)

    coef = classifier.coef_.ravel()

    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # create plot
    plt.figure(figsize=(15, 7))
    colors = ['crimson' if c < 0 else 'forestgreen' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.savefig('./output/svm_' + feature_set + '_top_features.png', bbox_inches='tight')
    plt.show()


feature_set = 'STRUCT'

path = './output/SVM_clf_30d_' + feature_set + '_args_for_coeff_mapping.pickle'

args = pickle.load(open(path, 'rb'))

feature_set = args['feature_set']
feature_names = args['feature_names']
model_name = args['model_name']
clf = args['clf']

plot_coefficients(clf, model_name, feature_set, feature_names, top_features=20)