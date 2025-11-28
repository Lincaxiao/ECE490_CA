import asgn_source as asou
import numpy as np
import scipy

def project_coordinate(x,constraint_min,constraint_max,n):
    
    ### write your code here      
    x_projected = np.clip(x, constraint_min, constraint_max)

    return x_projected




def run_projected_GD(constraint_min,constraint_max, Q,b, c,n):
    
    ### write your code here 
    #getting the step size
    eigenvalue = np.linalg.eigvalsh(Q)
    L = np.max(eigenvalue)
    alpha = 1/L
    x_k = project_coordinate(np.zeros(n), constraint_min, constraint_max, n)
    epsilon = 0.01
    x_norm = 999999

    while x_norm > epsilon:
        x_k_prev = x_k.copy()
        x_k_hat = x_k - (Q@x_k + b)*alpha
        x_k = project_coordinate(x_k_hat, constraint_min, constraint_max, n)
        x_norm = np.linalg.norm(x_k - x_k_prev, ord = 1)

    result = x_k

    return result



if __name__ == '__main__':
    n =25
    np.random.seed()
    constraint_min,constraint_max, Q_val,b_val, c_val = asou.get_parameters(n)
    armijo_sol = run_projected_GD(constraint_min,constraint_max,Q_val,b_val, c_val,n)
    
    print("Our PGD Solution : \n", armijo_sol)
    print("Solution from scipy.optimize: \n", scipy.optimize.minimize(lambda x: asou.opfun(x, Q_val, b_val, c_val),
                                        x0 = np.random.uniform(constraint_min,constraint_max,n),
                                        bounds = scipy.optimize.Bounds(constraint_min, constraint_max)).x)


