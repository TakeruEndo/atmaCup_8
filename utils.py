import sys

from models import lightGBM, xgboost, catboost, kneigbors, neural_net, linear_model


def select_model(name, X, y, X_test, output_path, fold_type, n_splits):
    if name == 'lgbm':
        model = lightGBM.lightGBM(
            X, y, X_test, output_path, fold_type, n_splits)
        return model
    elif name == 'xgb':
        model = xgboost.XGBoost(X, y, X_test, output_path, fold_type, n_splits)
        return model
    elif name == 'cat':
        model = catboost.Catboost(
            X, y, X_test, output_path, fold_type, n_splits)
        return model
    elif name == 'knn':
        model = kneigbors.Kneighbor(
            X, y, X_test, output_path, fold_type, n_splits)
        return model
    elif name == 'nn':
        model = neural_net.NeuralNet(
            X, y, X_test, output_path, fold_type, n_splits)
        return model
    elif name == 'linear':
        model = linear_model.Linear_models(
            X, y, X_test, output_path, fold_type, n_splits)
        return model
    else:
        print('model selection error')
        sys.exit()
