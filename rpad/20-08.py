import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import matplotlib.pyplot as plt

    return LinearRegression, fetch_ucirepo, np, plt


@app.cell
def _(fetch_ucirepo, np):
    auto_mpg = fetch_ucirepo(id=9)

    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X = X[mask]
    y = y[mask]

    X_horsepower = X["horsepower"].tonumpy()

    print(X.shape)
    print(y.shape)
    print(X_horsepower.shape)
    return X_horsepower, y


@app.cell
def _(X_horsepower, plt, y):
    fig, ax = plt.subplots()

    ax.scatter(X_horsepower, y)

    ax.set_title("Horsepower x MPG")
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("MPG")

    plt.show()
    return


@app.cell
def _(LinearRegression, X_horsepower, y):
    for i in range(1, 11):
    
        reg = LinearRegression()
        reg.fit(X_horsepower, y)
    return


if __name__ == "__main__":
    app.run()
