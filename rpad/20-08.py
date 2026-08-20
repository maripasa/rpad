import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from ucimlrepo import fetch_ucirepo

    return (
        LinearRegression,
        PolynomialFeatures,
        StandardScaler,
        fetch_ucirepo,
        mean_squared_error,
        mo,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(fetch_ucirepo, np):
    auto_mpg = fetch_ucirepo(id=9)

    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    X = X[mask]
    y = y[mask].to_numpy().ravel()

    X_horsepower = X[["horsepower"]].to_numpy()

    print(X.shape)
    print(y.shape)
    print(X_horsepower.shape)
    return X, X_horsepower, y


@app.cell
def _(X_horsepower, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_horsepower, y, test_size=0.3, random_state=42
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X, train_test_split, y):
    X_train_full, X_validation_full, y_train_full, y_validation_full = (
        train_test_split(X, y, test_size=0.3, random_state=42)
    )

    print("Full feature train:", X_train_full.shape, y_train_full.shape)
    print(
        "Full feature validation:",
        X_validation_full.shape,
        y_validation_full.shape,
    )
    return X_train_full, X_validation_full, y_train_full, y_validation_full


@app.cell
def _(X_horsepower, X_train, np, y_train):
    X_plot = np.linspace(
        np.floor(X_horsepower.min()),
        np.ceil(X_horsepower.max()),
        500,
    ).reshape(-1, 1)

    np.random.seed(42)
    _idx = np.random.choice(len(X_train), size=len(X_train) // 10, replace=False)

    X_train_random = X_train[_idx]
    y_train_random = y_train[_idx]
    return X_plot, X_train_random, y_train_random


@app.cell
def _(LinearRegression, PolynomialFeatures):
    def polynomial_predict(X, y, X_plot, degree):
        poly = PolynomialFeatures(degree, include_bias=False)
        reg = LinearRegression()

        reg.fit(poly.fit_transform(X), y)
        return reg.predict(poly.transform(X_plot))

    return (polynomial_predict,)


@app.cell
def _(X_test, X_train, plt, y_test, y_train):
    _fig, _ax = plt.subplots()

    _ax.scatter(X_train, y_train, c="orange", label="train")
    _ax.scatter(X_test, y_test, c="blue", label="test")

    _ax.set_title("Horsepower x MPG, training data")
    _ax.set_xlabel("Horsepower")
    _ax.set_ylabel("MPG")

    _ax.legend()

    plt.show()
    return


@app.cell
def _(X_plot, X_train, plt, polynomial_predict, y_train):
    _fig, _ax = plt.subplots()

    _ax.scatter(X_train, y_train)

    for _exp in range(1, 11):
        _y_plot = polynomial_predict(X_train, y_train, X_plot, _exp)
        _ax.plot(X_plot, _y_plot, label=f"exp={_exp}")

    _ax.set_title("Polynomial regressions fitted on training data")
    _ax.set_xlabel("Horsepower")
    _ax.set_ylabel("MPG")
    _ax.legend()

    plt.show()
    return


@app.cell
def _(
    X_plot,
    X_train_random,
    np,
    plt,
    polynomial_predict,
    y_train,
    y_train_random,
):
    _fig, _ax = plt.subplots()

    _exp = 10
    _y_plot = polynomial_predict(X_train_random, y_train_random, X_plot, _exp)

    _ax.scatter(X_train_random, y_train_random)
    _ax.plot(X_plot, _y_plot, color="crimson", label=f"exp={_exp}")

    _ax.set_title("Degree 10 on a random training subset")
    _ax.set_xlabel("Horsepower")
    _ax.set_ylabel("MPG")
    _ax.set_ylim(np.min(y_train), np.max(y_train))
    _ax.legend()

    plt.show()
    return


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    StandardScaler,
    X_plot,
    X_train_random,
    np,
    plt,
    y_train,
    y_train_random,
):
    _fig, _ax = plt.subplots()

    _exp = 10
    _scaler = StandardScaler()
    _poly = PolynomialFeatures(_exp, include_bias=False)
    _reg = LinearRegression()

    _X_scaled = _scaler.fit_transform(X_train_random)
    _X_plot_scaled = _scaler.transform(X_plot)

    _reg.fit(_poly.fit_transform(_X_scaled), y_train_random)
    _y_plot = _reg.predict(_poly.transform(_X_plot_scaled))

    _ax.scatter(X_train_random, y_train_random)
    _ax.plot(X_plot, _y_plot, color="crimson", label=f"exp={_exp}")

    _ax.set_title("Degree 10 on a random training subset, scaled")
    _ax.set_xlabel("Horsepower")
    _ax.set_ylabel("MPG")
    _ax.set_ylim(np.min(y_train), np.max(y_train))
    _ax.legend()

    plt.show()
    return


@app.cell
def _(mo):
    degree = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=1,
        label="Polynomial degree",
        show_value=True,
    )

    degree
    return (degree,)


@app.cell(hide_code=True)
def _(
    LinearRegression,
    PolynomialFeatures,
    StandardScaler,
    X_plot,
    X_train,
    degree,
    plt,
    y_train,
):
    _fig, _ax = plt.subplots()

    _exp = degree.value
    _scaler = StandardScaler()
    _poly = PolynomialFeatures(_exp, include_bias=False)
    _reg = LinearRegression()

    _X_scaled = _scaler.fit_transform(X_train)
    _X_plot_scaled = _scaler.transform(X_plot)

    _reg.fit(_poly.fit_transform(_X_scaled), y_train)
    _y_plot = _reg.predict(_poly.transform(_X_plot_scaled))

    _ax.scatter(X_train, y_train)
    _ax.plot(X_plot, _y_plot, color="crimson", label=f"exp={_exp}")

    _ax.set_title(f"Polynomial regression fitted on training data, exp={_exp}, scaled")
    _ax.set_xlabel("Horsepower")
    _ax.set_ylabel("MPG")
    _ax.legend()

    plt.show()
    return


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    StandardScaler,
    X_train_full,
    X_validation_full,
    mean_squared_error,
    np,
    plt,
    y_train_full,
    y_validation_full,
):
    _train_losses = []
    _validation_losses = []

    for _exp in range(1, 11):
        _scaler = StandardScaler()
        _poly = PolynomialFeatures(_exp, include_bias=False)
        _reg = LinearRegression()

        _X_train_scaled = _scaler.fit_transform(X_train_full)
        _X_validation_scaled = _scaler.transform(X_validation_full)

        _X_train_poly = _poly.fit_transform(_X_train_scaled)
        _X_validation_poly = _poly.transform(_X_validation_scaled)

        _reg.fit(_X_train_poly, y_train_full)

        _train_losses.append(
            mean_squared_error(y_train_full, _reg.predict(_X_train_poly))
        )
        _validation_losses.append(
            mean_squared_error(
                y_validation_full,
                _reg.predict(_X_validation_poly),
            )
        )

    _degrees = np.arange(1, 11)

    _fig, _ax = plt.subplots()
    _ax.plot(_degrees, _train_losses, marker="o", label="Training loss")
    _ax.plot(_degrees, _validation_losses, marker="o", label="Validation loss")

    _ax.set_title(
        "Training and validation loss by polynomial degree, all features, scaled"
    )
    _ax.set_xlabel("Polynomial degree")
    _ax.set_ylabel("MSE")
    _ax.set_xticks(_degrees)
    _ax.legend()

    plt.show()
    return


if __name__ == "__main__":
    app.run()
