import asgn_source as asou
import numpy as np
import scipy


def project_coordinate(x,constraint_min,constraint_max,n):
    
    ### write your code here      
    x_projected = np.clip(x, constraint_min, constraint_max)
    return x_projected

def cal_grad_f(x, Q, b):
    # grad_f(x) = Q*x + b
    return Q @ x + b

def run_projected_GD(constraint_min,constraint_max, Q,b, c,n):
    
    ### write your code here 
    # initial point
    x0 = np.random.uniform(constraint_min, constraint_max, n)
    # Get L and m
    eigvals = np.linalg.eigvalsh(Q)
    L, m = np.max(eigvals), np.min(eigvals)
    # Set step size and stopping criterion
    epsilon = 0.01 # || x_{k+1} - x_k ||_2 < epsilon
    step_size = 2 / (m + L)

    k = 0 # iteration counter
    # calculate gradient at starting point
    grad_cur = cal_grad_f(x0, Q, b)
    x_cur = x0.copy()
    delta = float('inf')
    # iterate until the norm of gradient is less than epsilon
    while (k == 0) or (delta > epsilon):
        x_next = x_cur - step_size * grad_cur # gradient descent step
        x_next = project_coordinate(x_next, constraint_min, constraint_max, n) # projection step
        grad_cur = cal_grad_f(x_next, Q, b)
        delta = np.linalg.norm(x_next - x_cur, 2)
        x_cur = x_next.copy()
        k += 1 # increment iteration counter

    result = x_cur
    
    return result



if __name__ == '__main__':
    n =25
    np.random.seed(22)
    constraint_min,constraint_max, Q_val,b_val, c_val = asou.get_parameters(n)
    armijo_sol = run_projected_GD(constraint_min,constraint_max,Q_val,b_val, c_val,n)
    # using scipy to get the optimal solution
    actual_sol = scipy.optimize.minimize(lambda x: asou.opfun(x, Q_val, b_val, c_val),
                                        x0 = np.random.uniform(constraint_min,constraint_max,n),
                                        bounds = scipy.optimize.Bounds(constraint_min, constraint_max))
    print('Projected Gradient Descent solution: ', armijo_sol)
    print('Scipy solution: ', actual_sol.x)
    print(f'Difference: {np.linalg.norm(armijo_sol - actual_sol.x, 2):.4f}')
