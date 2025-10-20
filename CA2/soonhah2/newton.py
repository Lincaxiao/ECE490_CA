import numpy as np
import matplotlib.pyplot as plt

def newton_method(z0: np.array, N=50)->np.array:

    zN = z0.copy()

    for _ in range(N):

        x = zN[0]
        y = zN[1]

        df_dx = 3*x**2*y - y**3
        df_dy = x**3 -3*x*y**2 - 1

        d2f_dx2   = 6 * x * y
        d2f_dxdy  = 3 * x**2 - 3 * y**2
        d2f_dy2   = -6 * x * y

        grad = np.array([df_dx, df_dy])
        hess = np.array([[d2f_dx2, d2f_dxdy],
                        [d2f_dxdy, d2f_dy2]])

        update = np.linalg.solve(hess, grad)

        zN = zN - update

    return zN

def plot_image(s_points: np.ndarray, n=500, domain=(-1, 1, -1, 1)):
    m = np.zeros((n, n))
    xmin, xmax, ymin, ymax = domain
    for ix, x in enumerate(np.linspace(xmin, xmax, n)):
        for iy, y in enumerate(np.linspace(ymin, ymax, n)):
            z0 = np.array([x, y])
            zN = newton_method(z0)
            code = np.argmin(np.linalg.norm(s_points - zN, ord=2, axis=1))
            m[iy, ix] = code

    # Display and save the image
    plt.imshow(m, cmap='brg')
    plt.axis('off')
    plt.savefig("q2_hw3.png")

if __name__ == '__main__':
    #stationary points:  z*_1 = (1, 0)
    # z*_2 = (-1/2, sqrt(3)/2)
    # z*_3 = (-1/2, -sqrt(3)/2)

    stationary_points = np.array([
            [1.0, 0.0],
            [-0.5, np.sqrt(3)/2],
            [-0.5, -np.sqrt(3)/2]
        ])    
    plot_image(stationary_points)