import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing, load_diabetes
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from sklearn.preprocessing import StandardScaler
    import torch

    return (
        LinearRegression,
        SGDRegressor,
        StandardScaler,
        fetch_california_housing,
        load_diabetes,
        np,
        plt,
        torch,
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
def _():
    lrs = [0.0001, 0.001, 0.001, 0.1, 0.5]
    return (lrs,)


@app.cell
def _(MyLinearRegression, X_california_std, lrs, plt, y_california):
    _fig, _ax = plt.subplots()

    for _lr in lrs:
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
def _(MyLinearRegression, X_diabetes_std, lrs, plt, y_diabetes):
    _fig, _ax = plt.subplots()

    for _lr in lrs:
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


@app.cell
def _(X, torch):
    class PytorchLinearRegressor:
        def __init__(self):
            self.loss_over_time = []
            self.w = []
    
        def fit(self, X, y, lr, epochs):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()

            X_intercept = torch.column_stack((torch.ones(X.shape[0]), X))
            self.w = torch.zeros(X_intercept.shape[1], requires_grad=True)

            for _ in range(epochs):
                y_hat = X_intercept @ self.w
                residual = y_hat - y
                loss = torch.mean(residual ** 2)

                self.loss_over_time.append(loss.item())

                loss.backward()

                with torch.no_grad():
                    self.w -= lr * self.w.grad
                    self.w.grad.zero_()

        def predict(self, x):
            x = torch.column_stack((torch.ones(x.shape[0], torch.from_numpy(X).float())))
            with torch.no_grad():
                return self.x @ self.w

    return (PytorchLinearRegressor,)


@app.cell
def _(PytorchLinearRegressor, X_california_std, lrs, plt, y_california):
    _fig, _ax = plt.subplots()

    for _lr in lrs:
        _lin_reg = PytorchLinearRegressor()
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
def _(SGDRegressor, X_california_std, np, self, y_california):
    class ScikitLinearRegression:
        def __init__(self, lr):
            self.loss_over_time = []
            self.w = []
            self.reg = SGDRegressor(
                loss="squared_error",
                penalty=None,
                learning_rate="constant",
                eta0=lr,
                max_iter=1,
                tol=None,
            )

        def fit(self, X, y, epochs):
            for _ in range(1000):
                self.reg.partial_fit(X, y)
                y_hat = self.reg.predict(X_california_std)
                loss = np.mean((y_hat - y_california) ** 2)
                self.loss_over_time.append(loss)

            self.w = [self.reg.intercept_] + self.reg.coef_

        def predict(X):
            return self.reg.predict(X)

    return (ScikitLinearRegression,)


@app.cell
def _(ScikitLinearRegression, X_california_std, lrs, plt, y_california):
    _fig, _ax = plt.subplots()

    for _lr in lrs:
        _lin_reg = ScikitLinearRegression(_lr)
        _lin_reg.fit(X_california_std, y_california, 1000)
        _ax.plot(range(1000), _lin_reg.loss_over_time, label=f"lr={_lr}")
        print(_lin_reg.w)
        print(_lin_reg.loss_over_time[-1])

    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Loss")
    _ax.legend()

    plt.show()

    return


if __name__ == "__main__":
    app.run()
