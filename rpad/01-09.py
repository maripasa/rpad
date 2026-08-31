import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from sklearn.datasets import load_diabetes
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge
    import numpy as np
    import matplotlib.pyplot as plt

    return (
        KNeighborsRegressor,
        PolynomialFeatures,
        Ridge,
        StandardScaler,
        load_diabetes,
        make_pipeline,
        np,
        train_test_split,
    )


@app.cell
def _(load_diabetes):
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    y = y.ravel()
    print(X.shape, X[:, 1].shape, y.shape)
    return X, diabetes, y


@app.cell
def _(diabetes):
    diabetes.feature_names
    return


@app.cell
def _(
    KNeighborsRegressor,
    PolynomialFeatures,
    Ridge,
    StandardScaler,
    make_pipeline,
):
    def knn(X, y, k):
        model = KNeighborsRegressor(k)
        return model.fit(X, y)

    def poly(X, y, p, lamb):
        model = make_pipeline(
            PolynomialFeatures(degree=p),
            StandardScaler(),
            Ridge(alpha=lamb)
        )
        return model.fit(X, y)   

    return


@app.cell
def _(X, np, train_test_split, y):
    p = range(1, 11)
    k = range(1, X.shape[0])
    seeds = [42, 67, 102, 333, 12]
    f_amount = 5
    test_size = 3
    test_mse = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        idx = np.random.permutation(len(X_train))
        folds = np.array_split(idx, f_amount)
    
        X_folds = [X_train[p] for f in folds]
        y_folds = [y_train[p] for f in folds]

        for i in range(X_folds):
            X_folds[i]
    return


if __name__ == "__main__":
    app.run()
