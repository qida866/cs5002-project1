import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2.1 simple quadratic functions

def f1(x):
    # simple quadratic function
    return x * x


def deriv_f1(x):
    # derivative of f1(x)
    return 2 * x


def f2(x):
    # another quadratic function
    return x * x - 2 * x + 3


def deriv_f2(x):
    # derivative of f2(x)
    return 2 * x - 2


def gradient_descent(f, df, x0, alpha, eps, max_iter):
    """
    basic gradient descent for one variable
    I used a while loop so I can clearly see each step
    """
    x = x0
    step = 0

    while step < max_iter:
        grad = df(x)
        new_x = x - alpha * grad

        # check stopping condition
        if abs(new_x - x) < eps:
            break

        x = new_x
        step = step + 1

    return x, step


def plot_1d_result(f, x_star, xmin, xmax, title):
    xs = np.linspace(xmin, xmax, 400)
    ys = f(xs)

    plt.figure()
    plt.plot(xs, ys)
    plt.scatter(x_star, f(x_star))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.grid(True)
    plt.show()


# 2.2 non-convex function with many minima

def f3(x):
    return np.sin(x) + np.cos(np.sqrt(2) * x)


def deriv_f3(x):
    return np.cos(x) - np.sqrt(2) * np.sin(np.sqrt(2) * x)


def plot_f3():
    xs = np.linspace(0, 10, 800)
    ys = f3(xs)

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("f3(x)")
    plt.title("f3(x) = sin(x) + cos(sqrt(2)x)")
    plt.grid(True)
    plt.show()


# 3. numerical derivatives

def approx_derivative_1d(f, x):
    # small step size for finite difference
    h = 0.00001
    value1 = f(x + h)
    value2 = f(x)
    return (value1 - value2) / h


def gradient_descent_numeric(f, x0, alpha, eps, max_iter):
    x = x0
    step = 0

    while step < max_iter:
        grad = approx_derivative_1d(f, x)
        new_x = x - alpha * grad

        if abs(new_x - x) < eps:
            break

        x = new_x
        step = step + 1

    return x, step


# 4. gradient descent in 2D

def approx_derivative_2d(f, x, y):
    h = 0.00001
    dfx = (f(x + h, y) - f(x, y)) / h
    dfy = (f(x, y + h) - f(x, y)) / h
    return dfx, dfy


def gradient_descent_2d(f, x0, y0, alpha, eps, max_iter):
    x = x0
    y = y0
    step = 0

    while step < max_iter:
        dfx, dfy = approx_derivative_2d(f, x, y)

        new_x = x - alpha * dfx
        new_y = y - alpha * dfy

        # stop only if both x and y change very little
        if abs(new_x - x) < eps and abs(new_y - y) < eps:
            break

        x = new_x
        y = new_y
        step = step + 1

    return x, y, step


def f_xy(x, y):
    return x * x + y * y


def plot_3d_result(x_star, y_star):
    xs = np.linspace(-3, 3, 40)
    ys = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(xs, ys)
    Z = f_xy(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7)
    ax.scatter(x_star, y_star, f_xy(x_star, y_star))
    ax.set_title("f(x, y) = x^2 + y^2")
    plt.show()


# main testing part

def main():
    # 2.1 basic test
    print("2.1 basic gradient descent test")

    x1, s1 = gradient_descent(f1, deriv_f1, 3, 0.1, 0.001, 1000)
    x2, s2 = gradient_descent(f2, deriv_f2, 3, 0.1, 0.001, 1000)

    print("f1 result:", x1, "steps:", s1)
    print("f2 result:", x2, "steps:", s2)

    plot_1d_result(f1, x1, -3, 3, "f1(x)")
    plot_1d_result(f2, x2, -1, 4, "f2(x)")

    # different alpha values
    print("\nTesting different learning rates")
    alphas = [1.0, 0.1, 0.0001]

    for a in alphas:
        x, s = gradient_descent(f1, deriv_f1, 3, a, 0.001, 1000)
        print("alpha =", a, "x* =", x, "steps =", s)

    # f3 tests
    print("\nTesting f3 with different starting points")
    plot_f3()

    starts = [1, 4, 5, 7]
    for x0 in starts:
        x_star, s = gradient_descent(f3, deriv_f3, x0, 0.1, 0.001, 1000)
        print("start at", x0, "=> x* =", x_star)

    # numeric derivative test
    print("\nNumeric gradient descent test")
    x_num, s_num = gradient_descent_numeric(f1, 3, 0.1, 0.001, 1000)
    print("numeric GD result:", x_num, "steps:", s_num)

    # 2D test
    print("\n2D gradient descent test")
    x_star, y_star, s2d = gradient_descent_2d(f_xy, 3, 3, 0.1, 0.001, 1000)
    print("2D result:", x_star, y_star, "steps:", s2d)

    plot_3d_result(x_star, y_star)


if __name__ == "__main__":
    main()
