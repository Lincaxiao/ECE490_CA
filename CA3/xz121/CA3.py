import numpy as np
import matplotlib.pyplot as plt

def get_Q_A_b(m, n):
    """
    Q : (n, n) positive definite matrix
    A : (m, n)
    b : (m,)
    Example:
        np.random.seed()
        Q, A, b = get_Q_A_b(10, 25)"""
    Q = np.random.rand(n, n) - 0.5
    Q = 10 * Q @ Q.T + 0.1*np.eye(n)
    A = np.random.normal(size=(m, n))
    b = 2 * (np.random.rand(m) - 0.5)
    return Q, A, b

def get_h_x(A, b, x):
    h = A @ x - b
    return h

def get_f(Q, x):
    f = x.T @ Q @ x
    return f

def get_soln(Q, A, b):
    """
    Get soln of equality constrained quadratic program:
    """
    kkt_matrix = np.block([[2*Q, A.T],
                           [A, np.zeros((A.shape[0], A.shape[0]))]])
    rhs = np.zeros(Q.shape[0] + A.shape[0])
    rhs[Q.shape[0]:] = b
    sol = np.linalg.solve(kkt_matrix, rhs)
    return sol[:Q.shape[0]]

def get_grad_Lc_respect_to_x(Q, A, b, c, lamb, x):
    """
    L_c(x, lambda) = x^T Q x + c/2 ||Ax - b||^2 + lambda^T (Ax - b);
    grad_x L_c(x, lambda) = 2 Q x + A^T (c (Ax - b) + lambda)
    """
    return 2 * Q @ x + A.T @ (c * (A @ x - b) + lamb)

def get_Lc(Q, A, b, c, lamb, x):
    """
    L_c(x, lambda) = x^T Q x + c/2 ||Ax - b||^2 + lambda^T (Ax - b);
    """
    f_x = get_f(Q, x)
    h_x = get_h_x(A, b, x)
    Lc = f_x + lamb.T @ h_x + c / 2 * (h_x.T @ h_x)
    return Lc

def find_alpha_with_armijo(Q, A, b, c, x_k, lamb_k, grad_Lc_of_x_k, beta=0.5, sigma=1e-4):
    """
    Find the step size with Armijo's rule
    Lc(x_k + alpha_j * d_j) <= Lc(x_k) + sigma * alpha_j * grad_Lc(x_k)^T dj"""
    curr_val = get_Lc(Q, A, b, c, lamb_k, x_k)
    alpha = 1.0
    d_k = - grad_Lc_of_x_k
    update_dir = - grad_Lc_of_x_k.T @ grad_Lc_of_x_k  

    while get_Lc(Q, A, b, c, lamb_k, x_k + alpha * d_k) > curr_val + sigma * alpha * update_dir:
        alpha *= beta
    return alpha

def inner_loop_with_fixed_lamb_k_and_c_k(Q, A, b, c_k, lamb_k, epsilon, x0=None, beta=0.5, max_iter=1000):
    """
    Minimize L_c_k(x, lambda_k) with gradient descent with Armijo's rule
    """
    if x0 is None:
        x0 = np.zeros(Q.shape[0])
    
    k = 0 # iteration counter
    x_cur = x0.copy()
    grad_cur = get_grad_Lc_respect_to_x(Q, A, b, c_k, lamb_k, x_cur)
    
    while np.linalg.norm(grad_cur, ord=2) >= epsilon:
        step_size = find_alpha_with_armijo(Q, A, b, c_k, x_cur, lamb_k, grad_cur, beta=beta)
        x_cur -= step_size * grad_cur
        grad_cur = get_grad_Lc_respect_to_x(Q, A, b, c_k, lamb_k, x_cur)
        k += 1
        
    return x_cur, k

def get_next_c(current_c, select_method_idx, beta=1.1, constant_c=10, A=None, b=None, gamma=None, x_curr=None, x_prev=None):
    if select_method_idx == 1:
        return constant_c
    elif select_method_idx == 2:
        return current_c + beta
    elif select_method_idx == 3:
        return current_c * beta
    elif select_method_idx == 4:
        if np.linalg.norm(get_h_x(A, b, x_curr)) > gamma * np.linalg.norm(get_h_x(A, b, x_prev)):
            return current_c * beta
        else:
            return current_c
    else:
        raise ValueError("Invalid select_method_idx")
    
def outer_loop(Q, A, b, x_star, c_select_method_idx, epsilon=1e-4
               , beta=1.1, gamma=0.9, constant_c=10, max_outer_iter=100):
    x = np.zeros(Q.shape[0])
    lamb = np.zeros(A.shape[0])
    c = 1.0

    outer_error_list = []
    outer_error_list.append(np.linalg.norm(x - x_star, ord=2) / np.linalg.norm(x_star, ord=2))

    outer_iter = 0
    inner_iter_list = []

    while np.linalg.norm(get_h_x(A, b, x), ord=2) >= epsilon and outer_iter < max_outer_iter:
        x_prev = x.copy()
        x, inner_loop_iter = inner_loop_with_fixed_lamb_k_and_c_k(
            Q, A, b, c, lamb, epsilon, x0=x)

        h_k = get_h_x(A, b, x)
        lamb += c * h_k
        c = get_next_c(c, c_select_method_idx, 
                       beta=beta, A=A, b=b, gamma=gamma, x_curr=x, x_prev=x_prev)

        outer_error_list.append(np.linalg.norm(x - x_star, ord=2) / np.linalg.norm(x_star, ord=2))
        outer_iter += 1
        inner_iter_list.append(inner_loop_iter)
    return x, outer_error_list, outer_iter, inner_iter_list

import matplotlib.pyplot as plt

def plot_results(results_dict):
    """
    results_dict: { 'Strategy Name': [error_k0, error_k1, ...] }
    """
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'd']
    line_styles = ['-', '--', '-.', ':']
    
    for i, (name, errors) in enumerate(results_dict.items()):
        plt.semilogy(errors, 
                     label=name, 
                     marker=markers[i % len(markers)], 
                     linestyle=line_styles[i % len(line_styles)],
                     markevery=max(1, len(errors)//10))
    
    plt.xlabel('Outer Iteration $k$')
    plt.ylabel('Relative Error $\\frac{||x^* - x^{(k)}||_2}{||x^*||_2}$')
    plt.title('Convergence of Method of Multipliers with Different $\{c_k\}$')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    m, n = 10, 25
    Q, A, b = get_Q_A_b(m, n)
    epsilon = 1e-4

    print("=======Exact solution========")
    x_star = get_soln(Q, A, b)
    print(f"x_star = {x_star}")
    print(f"||Ax* - b|| = {np.linalg.norm(A @ x_star - b):.2e}")
    print("-" * 50)

    strategies = [
        {"idx": 1, "name": "Constant c=5", "beta": 0, "gamma": 0},
        {"idx": 2, "name": "Linear c+=2", "beta": 2.0, "gamma": 0},
        {"idx": 3, "name": "Geometric c*=1.5", "beta": 1.5, "gamma": 0},
        {"idx": 4, "name": "Adaptive c*=2.0", "beta": 2.0, "gamma": 0.25} 
    ]

    results = {}

    for strat in strategies:
        print(f"Running Strategy: {strat['name']}...")
        
        x_final, errors, outer_k, total_inner_k = outer_loop(
            Q, A, b, x_star, 
            c_select_method_idx=strat['idx'], 
            epsilon=epsilon,
            beta=strat['beta'],
            gamma=strat['gamma'],
            constant_c=5
        )
        
        results[strat['name']] = errors
        
        print(f"     Converged in {outer_k} outer iters")
        print(f"     Final Relative Error: {errors[-1]:.2e}")
        print(f"     Total Inner Gradient Steps: {total_inner_k}")
        print("-" * 30)
        
    plot_results(results)
