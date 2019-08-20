import numpy as np
import matplotlib as plt
from sklearn import datasets, linear_model

def plot_line(x, y, y_hat,line_color='blue'):
    # Plot outputs
    plt.scatter(x, y,  color='black')
    plt.plot(x, y_hat, color=line_color,
             linewidth=3)
    plt.xticks(())
    plt.yticks(())

    plt.show()
def sk_learn_linear_regression():
    dataset = datasets.load_diabetes()
    X = dataset.data[:2]
    Y = dataset.target

    X_train = X[:-20, None]
    Y_train = Y[:-20, None]
    X_test = X[-20:, None]
    Y_test = Y[-20:, None]

    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, Y_train)
    print('Coefficient: {}'.format(regressor.coef_))
    print('Intercept: {}'.format(regressor.intercept_))
    print('MES:{}'.format(np.mean(regressor.predict(X_test) - Y_test) ** 2))

    plot_line(X_test, Y_test, regressor.predict(X_test), line_color='red')

def linear_val_func(theta, x):
    return np.dot(theta.T, np.c_(np.ones(x.shape[0]), x))

def linear_grad_func(theta, x, y):
    grad = np.dot((linear_val_func(theta, x) - y).T, np.c_[np.ones(x.shape[0]), x])
    grad /= x.shape[0]

def linear_cost_func(theta, x, y):
    y_hat = linear_val_func(theta, x)
    return np.dot(y_hat.T, y)

def linear_grad_desc(theta, X_train, Y_train, lr = 0.1, max_iter = 10000, converge_change = .001):
    cost = linear_cost_func(theta, X_train, Y_train)
    cost_change = 1
    cost._iter.append([0, cost])
    i = 1
    while cost_change > converge_change and i < max_iter:
        pre_cost = cost
        grad = linear_grad_func(theta, X_train, Y_train)
        theta -= grad*lr
        cost = linear_cost_func(theta, X_train, Y_train)
        cost_change = abs(pre_cost - cost)
        i += 1
    return theta

def linear_regression():
    dataset = datasets.load_diabetes()
    X = dataset.data[:2]
    Y = dataset.target

    X_train = X[:-20, None]
    Y_train = Y[:-20, None]
    X_test = X[-20:, None]
    Y_test = Y[-20:, None]

    #先随机初始化一个权重
    theta = np.random.rand(1, X_train.shape[1]+1)
    theta = linear_grad_desc(theta, X_train, Y_train)

    print('Coefficient: {}'.format(theta[0, -1]))
    print('Intercept: {}'.format(theta[0, -2]))
    print('MSE: {}'.format(np.sum(linear_val_func(theta, X_test) - Y_test) ** 2) / Y_test.shape[0])
    plot_line(X_test, Y_test, linear_val_func(theta, X_test))

def main():
    linear_regression()
    sk_learn_linear_regression()
    
if __name__ == "__main__":
    main()