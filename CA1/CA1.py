import numpy as np

# General help functions
# Generate different Q, b, c
def get_Q_b_c(n):
    Q = np.random.rand(n,n)-0.5
    Q = 10*Q @ Q.T
    b = 5*(np.random.rand(n)-0.5)
    c = 2*(np.random.rand(1)-0.5) # c is a scalar
    return Q,b,c

def cal_f(x, Q, b, c):
    return 0.5 * x.T @ Q @ x + b @ x + c

def cal_grad_f(x, Q, b):
    return Q @ x + b

# Find the largest eigenvalue of Q
def find_max_and_min_eigenvalue(Q):
    eigenvalues, _ = np.linalg.eig(Q)
    return np.max(eigenvalues), np.min(eigenvalues)

# Find the inverse of Q
def find_inverse(Q):
    return np.linalg.inv(Q)

# Task 1: Constant step size
def grad_descent_with_constant_step(Q, b, c, epsilon, x0=None):
    L, mu = find_max_and_min_eigenvalue(Q)
    step_size = 2 / (L + mu)
    if x0 is None:
        x0 = np.zeros(Q.shape[0])
    k = 0 # iteration counter
    grad_cur = cal_grad_f(x0, Q, b)
    x_cur = x0.copy()
    while np.linalg.norm(grad_cur, ord=2) >= epsilon:
        x_cur -= step_size * grad_cur
        grad_cur = cal_grad_f(x_cur, Q, b)
        k += 1
    return x_cur, k, cal_f(x_cur, Q, b, c), np.linalg.norm(cal_grad_f(x_cur, Q, b), ord=2), L, mu

# Task 3: Matrix Inversion
def solve_by_matrix_inversion(Q, b, c):
    x_star = -find_inverse(Q) @ b
    f_star = cal_f(x_star, Q, b, c)
    return x_star, f_star

# For Testing
if __name__ == "__main__":
    seed = 114
    np.random.seed(seed)
    n = 10
    epsilon = 1e-8
    Q, b, c = get_Q_b_c(n)
    x0 = np.random.rand(n)
    alpha_scale = 1
    # Task 1
    x_task1, k_task1, f_task1, grad_norm_task1, L, mu = grad_descent_with_constant_step(Q, b, c, epsilon, x0)
    # Task 3
    x_task3, f_task3 = solve_by_matrix_inversion(Q, b, c)

    # print results
    # print parameters
    print("=== Parameters ===")
    print(f"seed = {seed}, n = {n}, epsilon = {epsilon}, alpha_scale = {alpha_scale}")
    
    # print task 1
    print("=== Task1: Constant Step Size Gradient Descent ===")
    print(f"iterations = {k_task1}")
    print(f"f(x*) = {float(f_task1.item())}")
    print(f"x* = {x_task1.astype(float)}")
    print(f"||grad_f(x*)||_2 = {float(grad_norm_task1)}")
    print(f"L  = {float(L)}")
    print(f"mu = {float(mu)}")

    # print task 2
    print("\n=== Task2: Backtracking Line Search ===")

    # print task 3
    print("=== Task3: Matrix Inversion ===")
    print(f"f(x*) = {float(f_task3.item())}")
    print(f"x* = {x_task3.astype(float)}")

