import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from sklearn.datasets import load_diabetes
    import numpy as np
    import matplotlib.pyplot as plt

    return load_diabetes, np, plt


@app.cell
def _(np):
    def knn_regression(sample, X, k):
        distance = np.sum((X - sample) ** 2)
        idx = np.argsort(distance)
    


    return


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
def _(X, diabetes, plt, y):
    fig, ax = plt.subplots(10, figsize=(8, 8 * X.shape[1]))

    for idx, name in enumerate(diabetes.feature_names):
        ax[idx].set_title(name)
        ax[idx].scatter(X[:, idx], y)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
