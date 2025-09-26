import numpy as np

# General help functions
# Generate different Q, b, c
def get_Q_b_c(n):
    Q = np.random.rand(n,n)-0.5
    Q = 10*Q @ Q.T # make Q positive definite
    b = 5*(np.random.rand(n)-0.5) # b is a vector
    c = 2*(np.random.rand(1)-0.5) # c is a scalar
    return Q, b, c

def cal_f(x, Q, b, c):
    # f(x) = 0.5*x^T*Q*x + b^T*x + c
    return 0.5 * x.T @ Q @ x + b @ x + c

def cal_grad_f(x, Q, b):
    # grad_f(x) = Q*x + b
    return Q @ x + b

# Find the largest eigenvalue of Q
def find_max_and_min_eigenvalue(Q):
    eigenvalues, _ = np.linalg.eigh(Q) # use eigh for symmetric matrices
    return np.max(eigenvalues), np.min(eigenvalues)

# Find the inverse of Q
def find_inverse(Q):
    return np.linalg.inv(Q)

# Task 1: Constant step size
def grad_descent_with_constant_step(Q, b, c, epsilon, x0=None, step_scale=1):
    # clamp step_scale to be in (0,2)
    step_scale = max(1e-8, min(step_scale, 2-1e-8))
    # compute step size, between (0, 2/L)
    L, _ = find_max_and_min_eigenvalue(Q)
    step_size = step_scale * 1 / L
    if x0 is None:
        x0 = np.zeros(Q.shape[0])
    k = 0 # iteration counter
    # calculate gradient at starting point
    grad_cur = cal_grad_f(x0, Q, b)
    x_cur = x0.copy()
    # iterate until the norm of gradient is less than epsilon
    while np.linalg.norm(grad_cur, ord=2) >= epsilon:
        x_cur -= step_size * grad_cur # gradient descent step
        grad_cur = cal_grad_f(x_cur, Q, b)
        k += 1 # increment iteration counter
    # return x*, k, f(x*), ||grad_f(x*)||_2, L
    return x_cur, k, cal_f(x_cur, Q, b, c), np.linalg.norm(cal_grad_f(x_cur, Q, b), ord=2), L

#Task 2: Armijo's rule
def find_alpha_with_armijo(Q, b, c, x_k, beta=0.5):
    # s = 1
    #Set the adequate parameter. Sigma and beta ranges are specified in the textbook
    sigma = 10e-5 # can be between 10^-5 to 10^-1
    # beta = 1/2 # can be between 1/10 to 1/2
    # L, _ = find_max_and_min_eigenvalue(Q)
    alpha = 1

    # For easier comparison in the while statement
    grad_k = cal_grad_f(x_k, Q, b)
    d_k = -grad_k
    f_k = cal_f(x_k, Q, b, c)

    # So long as f(x+a*d_k) is less than f(x)+sigma*a*grad(f(x))d_k, we keep updating alpha
    while cal_f(x_k + alpha * d_k, Q, b, c) > f_k + sigma * alpha * (grad_k @ d_k):
        alpha = alpha * beta
    return alpha


def grad_descent_with_armijo(Q, b, c, epsilon, x0=None, beta=0.5):
    # mostly same as part 1 but just the step size is calculated with Armijo.
    if x0 is None:
        x0 = np.zeros(Q.shape[0])
    k = 0 # iteration counter
    grad_cur = cal_grad_f(x0, Q, b)
    x_cur = x0.copy()
    while np.linalg.norm(grad_cur, ord=2) >= epsilon:
        step_size = find_alpha_with_armijo(Q, b, c, x_cur, beta=beta)
        # print(f"armijo found={step_size}, K={k}, {np.linalg.norm(grad_cur, ord=2)}, {epsilon}")
        x_cur -= step_size * grad_cur
        grad_cur = cal_grad_f(x_cur, Q, b)
        k += 1
    # return x*, k, f(x*), ||grad_f(x*)||_2, L
    return x_cur, k, cal_f(x_cur, Q, b, c), np.linalg.norm(cal_grad_f(x_cur, Q, b), ord=2)

# Task 3: Matrix Inversion
def solve_by_matrix_inversion(Q, b, c):
    x_star = -find_inverse(Q) @ b # x* = -Q^(-1)b
    f_star = cal_f(x_star, Q, b, c) # f* = f(x*)
    return x_star, f_star

# For Testing
if __name__ == "__main__":
    seed = 114
    np.random.seed(seed)
    n = 10
    epsilon = 1e-5
    Q, b, c = get_Q_b_c(n)
    x0 = np.random.rand(n)
    step_scale = 1 # can be between (0,2), for task 1
    beta = 0.5 # theoretically can be between (0,1), for task 2
    # Task 1
    x_task1, k_task1, f_task1, grad_norm_task1, L = grad_descent_with_constant_step(Q, b, c, epsilon, x0)
    # Task 2
    x_task2, k_task2, f_task2, grad_norm_task2 = grad_descent_with_armijo(Q, b, c, epsilon, x0, beta)
    # Task 3
    x_task3, f_task3 = solve_by_matrix_inversion(Q, b, c)

    # print parameters
    print("=== Parameters ===")
    print(f"seed = {seed}, n = {n}, epsilon = {epsilon}")
    
    # print task 1
    print("\n=== Task1: Constant Step Size Gradient Descent ===")
    print(f"L  = {float(L)}, step_scale = {float(step_scale)}")
    print(f"iterations = {k_task1}")
    print(f"f(x*) = {float(f_task1.item())}")
    print(f"x* = {x_task1.astype(float)}")
    print(f"||grad_f(x*)||_2 = {float(grad_norm_task1)}")

    # print task 2
    print("\n=== Task2: Backtracking Line Search ===")
    print(f"beta = {float(beta)}")
    print(f"iterations = {k_task2}")
    print(f"f(x*) = {float(f_task2.item())}")
    print(f"x* = {x_task2.astype(float)}")
    print(f"||grad_f(x*)||_2 = {float(grad_norm_task2)}")

    # print task 3
    print("\n=== Task3: Matrix Inversion ===")
    print(f"f(x*) = {float(f_task3.item())}")
    print(f"x* = {x_task3.astype(float)}")
