import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing, load_diabetes
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    return (
        LinearRegression,
        StandardScaler,
        fetch_california_housing,
        load_diabetes,
        np,
        plt,
    )


@app.cell
def _(fetch_california_housing, load_diabetes):
    california = fetch_california_housing()
    diabetes = load_diabetes()

    X_california = california.data
    y_california = california.target

    X_diabetes = diabetes.data
    y_diabetes = diabetes.target

    print(X_california.shape, y_california.shape)
    print(X_diabetes.shape, y_diabetes.shape)
    return X_california, X_diabetes, y_california, y_diabetes


@app.cell
def _(
    LinearRegression,
    X_california,
    X_diabetes,
    np,
    y_california,
    y_diabetes,
):
    # CALIFORNIA
    _sk_reg = LinearRegression()
    _sk_reg.fit(X_california, y_california)
    print("CALIFORNIA WEIGHTS:\n", _sk_reg.coef_, _sk_reg.intercept_)

    _y_hat = _sk_reg.predict(X_california)
    print("LOSS: ", np.mean((y_california - _y_hat)**2))

    print("")
    # DIABETES
    _sk_reg = LinearRegression()
    _sk_reg.fit(X_diabetes, y_diabetes)
    print("DIABETES WEIGHTS:\n", _sk_reg.coef_, _sk_reg.intercept_)

    _y_hat = _sk_reg.predict(X_diabetes)
    print("LOSS: ", np.mean((y_diabetes - _y_hat)**2))
    return


@app.cell
def _(
    LinearRegression,
    StandardScaler,
    X_california,
    X_diabetes,
    np,
    y_california,
    y_diabetes,
):
    # CALIFORNIA
    std_california = StandardScaler()
    X_california_std = std_california.fit_transform(X_california)

    _sk_reg = LinearRegression()
    _sk_reg.fit(X_california_std, y_california)
    print("CALIFORNIA WEIGHTS:\n", _sk_reg.coef_, _sk_reg.intercept_)

    _y_hat = _sk_reg.predict(X_california_std)
    print("LOSS: ", np.mean((y_california - _y_hat) ** 2))

    print("")

    # DIABETES
    std_diabetes = StandardScaler()
    X_diabetes_std = std_diabetes.fit_transform(X_diabetes)

    _sk_reg = LinearRegression()
    _sk_reg.fit(X_diabetes_std, y_diabetes)
    print("DIABETES WEIGHTS:\n", _sk_reg.coef_, _sk_reg.intercept_)

    _y_hat = _sk_reg.predict(X_diabetes_std)
    print("LOSS: ", np.mean((y_diabetes - _y_hat) ** 2))
    return X_california_std, X_diabetes_std


@app.cell
def _(np):
    class MyLinearRegression:
        def __init__(self):
            self.loss_over_time = []
            self.w = None

        def fit(self, X, y, lr, epochs):
            X_intercept = np.column_stack((np.ones(X.shape[0]), X))
            self.w = np.zeros(X_intercept.shape[1])

            for _ in range(epochs):
                residual = X_intercept @ self.w - y
                mse = np.mean(residual ** 2)
                self.loss_over_time.append(mse)

                gradient = (2 / X.shape[0]) * X_intercept.T @ residual
                self.w = self.w - lr * gradient

        def predict(self, X):
            X_intercept = np.column_stack((np.ones(X.shape[0]), X))
            return X_intercept @ self.w

    return (MyLinearRegression,)


@app.cell
def _(MyLinearRegression, X_california_std, plt, y_california):
    _lrs = [0.0001, 0.001, 0.001, 0.1, 0.5]

    _fig, _ax = plt.subplots()

    for _lr in _lrs:
        _lin_reg = MyLinearRegression()
        _lin_reg.fit(X_california_std, y_california, _lr, 1000)
        _ax.plot(range(1000), _lin_reg.loss_over_time, label=f"lr={_lr}")
        print(_lin_reg.w)
        print(_lin_reg.loss_over_time[-1])

    _ax.set_ylim(0, 7)
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Loss")
    _ax.legend()

    plt.show()
    return


@app.cell
def _(MyLinearRegression, X_diabetes_std, plt, y_diabetes):
    _lrs = [0.0001, 0.001, 0.001, 0.1, 0.5]

    _fig, _ax = plt.subplots()

    for _lr in _lrs:
        _lin_reg = MyLinearRegression()
        _lin_reg.fit(X_diabetes_std, y_diabetes, _lr, 1000)
        _ax.plot(range(1000), _lin_reg.loss_over_time, label=f"lr={_lr}")
        print(_lin_reg.w)
        print(_lin_reg.loss_over_time[-1])

    _ax.set_ylim(0, 30000)
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Loss")
    _ax.legend()

    plt.show()
    return


if __name__ == "__main__":
    app.run()
