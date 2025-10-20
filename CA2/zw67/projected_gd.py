import asgn_source as asou
import numpy as np
import scipy


def project_coordinate(x,constraint_min,constraint_max,n):
    
    ### write your code here      
    x_projected = np.clip(x, constraint_min, constraint_max)
    return x_projected




def run_projected_GD(constraint_min,constraint_max, Q,b, c,n):
    
    ### write your code here 
    x0 = np.random.uniform(constraint_min, constraint_max, n)
    epsilon = 0.01
    
    eigvals, _ = np.linalg.eigh(Q)
    L,m = np.max(eigvals), np.min(eigvals)
    step_size = 2 / (L + m)
    
    k = 0
    grad_cur = Q @ x0 + b
    x_cur = x0.copy()
    delta = float('inf')
    
    while (k == 0) or (delta > epsilon):
        x_next = x_cur - step_size * grad_cur
        x_next = project_coordinate(x_next, constraint_min, constraint_max, n)
        grad_cur = Q @ x_next + b
        delta = np.linalg.norm(x_next - x_cur, 2)
        x_cur = x_next.copy()
        k += 1
    
    return x_cur



if __name__ == '__main__':
    n =25
    np.random.seed(22)
    constraint_min,constraint_max, Q_val,b_val, c_val = asou.get_parameters(n)
    armijo_sol = run_projected_GD(constraint_min,constraint_max,Q_val,b_val, c_val,n)
    print('The objective function value found by projected GD is:', armijo_sol)

    # using scipy to get the optimal solution
    actual_sol = scipy.optimize.minimize(lambda x: asou.opfun(x, Q_val, b_val, c_val),
                                        x0 = np.random.uniform(constraint_min,constraint_max,n),
                                        bounds = scipy.optimize.Bounds(constraint_min, constraint_max))
    print('The objective function value found by scipy is:', actual_sol.x)

