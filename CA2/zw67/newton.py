import numpy as np
import matplotlib.pyplot as plt


def newton_method(z0: np.array, N: int = 50) -> np.array:
    """Placeholder Newton method.

    Replace the body of this function with your Newton's method implementation.
    Input:
      - z0: numpy.ndarray of shape (2,)
      - N: maximum number of iterations
    Output:
      - zN: numpy.ndarray of shape (2,) final iterate
    """
    # Write your code here.
    # If needed, you can define other functions as well to be used here.
    # Input z0 and output zN should be numpy.ndarray objects with 2 elements:
    # e.g. np.array([x, y]).

    # Current placeholder just returns the initial point (no iterations).
    for _n in range(N):
        x, y = z0[0], z0[1]
        grad = np.array([3*x**2*y - y**3, x**3 - 3*x*y**2 - 1])
        H = np.array([[6*x*y, 3*x**2 - 3*y**2], [3*x**2 - 3*y**2, -6*x*y]])
        H_inv = np.linalg.inv(H)
        z0 = z0 - H_inv @ grad
        zN = z0
    return zN



def plot_image(s_points: np.array , n=500 , domain=(-1, 1, -1, 1)):
    m = np.zeros((n, n))
    xmin , xmax , ymin , ymax = domain
    for ix, x in enumerate(np.linspace(xmin , xmax , n)):
        for iy, y in enumerate(np.linspace(ymin , ymax , n)):
            z0 = np.array([x,y])
            zN = newton_method(z0)
            code = np.argmin(np.linalg.norm(s_points-zN,ord=2,axis=1))
            m[iy , ix] = code
    plt.imshow(m, cmap='brg')
    plt.axis('off')
    plt.savefig("q2_hw3.png")


if __name__ == '__main__':
    # Example of usage.
    # In this example, the stationary points are (0,0), (1,1), (2,2) and (3,3).
    # Replace these with the ones obtained in part a).
    stationary_points = np.array([
        [1.0, 0.0],
        [-0.5, np.sqrt(3) / 2],
        [-0.5, -np.sqrt(3) / 2]
    ])
    plot_image(stationary_points)
