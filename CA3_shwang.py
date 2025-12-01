import numpy as np
import matplotlib.pyplot as plt

def get_Q_A_b(m, n):
    """
    Generates random matrices Q, A and vector b based on the assignment snippet.
    """
    Q = np.random.rand(n, n) - 0.5
    Q = 10 * Q @ Q.T + 0.1*np.eye(n)
    A = np.random.normal(size=(m, n))
    b = 2 * (np.random.rand(m) - 0.5)
    return Q, A, b
    

def solve_exact_kkt(Q, A, b):
    """
    Solves the KKT system to find the exact analytical solution x*.
    System:
    [ 2Q   A.T ] [ x ] = [ 0 ]
    [ A     0  ] [ v ]   [ b ]
    """
    n = Q.shape[0]
    m = A.shape[0]
    
    # Construct KKT Matrix
    upper = np.hstack([2 * Q, A.T])
    lower = np.hstack([A, np.zeros((m, m))])
    KKT = np.vstack([upper, lower])
    
    rhs = np.concatenate([np.zeros(n), b])
    
    # Solve linear system
    solution = np.linalg.solve(KKT, rhs)
    x_star = solution[:n]
    
    return x_star

def objective_function(Q, x):
    return x.T @ Q @ x

def augmented_lagrangian(x, lamb, c, Q, A, b):
    """
    Calculates the value of the Augmented Lagrangian L_c(x, lambda).
    L_c(x, lambda) = x.T Q x + lambda.T (Ax - b) + (c/2) ||Ax - b||^2
    """
    h_x = A @ x - b
    term1 = x.T @ Q @ x
    term2 = lamb.T @ h_x
    term3 = (c / 2) * (np.linalg.norm(h_x)**2)
    return term1 + term2 + term3

def grad_augmented_lagrangian(x, lamb, c, Q, A, b):
    """
    Calculates the gradient of L_c with respect to x.
    grad L_c = 2Qx + A.T lambda + c * A.T (Ax - b)
             = (2Q + c A.T A)x + A.T(lambda - c b)
    """
    h_x = A @ x - b
    grad = 2 * (Q @ x) + A.T @ lamb + c * (A.T @ h_x)
    return grad

def gradient_descent_armijo(x_init, lamb, c, Q, A, b, tol=1e-4, max_iter=1000):
    """
    Minimizes the Augmented Lagrangian wrt x using Gradient Descent with Armijo Rule.
    """
    x = x_init.copy()
    
    # Armijo parameters
    beta = 0.5   # Step size reduction factor
    sigma = 1e-4  # Acceptance parameter (0 < sigma < 1)
    
    for i in range(max_iter):
        grad = grad_augmented_lagrangian(x, lamb, c, Q, A, b)
        norm_grad = np.linalg.norm(grad)
        
        # Inner loop stopping criterion
        if norm_grad < tol:
            break
            
        d = -grad  # Descent direction
        
        # Backtracking line search (Armijo Rule)
        alpha = 1.0
        current_L = augmented_lagrangian(x, lamb, c, Q, A, b)
        
        while True:
            x_new = x + alpha * d
            new_L = augmented_lagrangian(x_new, lamb, c, Q, A, b)
            
            # Check Armijo condition: L(x + alpha*d) <= L(x) + sigma * alpha * grad.T * d
            if new_L <= current_L + sigma * alpha * np.dot(grad, d):
                break
            
            alpha *= beta
            if alpha < 1e-10: # Safety break for extremely small steps
                break
                
        x = x + alpha * d
        
    return x

def method_of_multipliers(Q, A, b, c_rule_func, eps=1e-5, max_outer_iter=100):
    """
    Implements the Method of Multipliers.
    
    Args:
        c_rule_func: A function that takes (c_current, h_current, h_prev) and returns c_next
    """
    n = Q.shape[0]
    m = A.shape[0]
    
    # Initialization
    x = np.zeros(n)      # Initial guess for x
    lamb = np.zeros(m)   # Initial guess for lambda
    c = 1.0              # Initial penalty parameter (can be overridden by logic)
    
    history_x = []
    
    h_prev_norm = float('inf')
    
    for k in range(max_outer_iter):
        # 1. Update x using Gradient Descent
        x = gradient_descent_armijo(x, lamb, c, Q, A, b)
        history_x.append(x.copy())
        
        # Calculate constraint violation h(x) = Ax - b
        h_x = A @ x - b
        h_norm = np.linalg.norm(h_x)
        
        # 2. Check Stopping Condition
        if h_norm < eps:
            print(f"Converged at iteration {k+1}. Violation: {h_norm:.2e}")
            break
            
        # 3. Update Lambda
        # lambda^(k+1) = lambda^(k) + c_k * h(x^(k))
        lamb = lamb + c * h_x
        
        # 4. Update c (Penalty Parameter)
        c = c_rule_func(c, h_norm, h_prev_norm)
        
        h_prev_norm = h_norm
        
    return x, history_x

def run_experiments():
    # 1. Setup Problem
    # m, n = 10, 25
    m, n = 30, 50
    # Setting random seed
    np.random.seed(42)
    Q, A, b = get_Q_A_b(m, n)
    
    # 2. Get Exact Solution for Error Plotting
    x_star = solve_exact_kkt(Q, A, b)
    x_star_norm = np.linalg.norm(x_star)
    opt_val_exact = objective_function(Q, x_star)
    
    print(f"Exact Optimal Objective Value: {opt_val_exact:.4f}")
    
    # 3. Define c_k Update Rules
    
    # (a) Constant
    def rule_constant(c, h, h_prev):
        return c # Keep constant, e.g., 5.0
    
    # (b) Additive: c_k+1 = c_k + beta
    def rule_additive(c, h, h_prev):
        beta = 2.0
        return c + beta
    
    # (c) Multiplicative: c_k+1 = beta * c_k
    def rule_multiplicative(c, h, h_prev):
        beta = 1.5
        return c * beta
        
    # (d) Conditional (Bertsekas suggestion)
    # Increase c if constraint violation doesn't decrease sufficiently (by factor gamma)
    def rule_conditional(c, h, h_prev):
        beta = 2.0
        gamma = 0.25
        if h > gamma * h_prev:
            return c * beta
        return c

    experiments = [
        ("Constant (c=5)", rule_constant),
        ("Additive (beta=2)", rule_additive),
        ("Multiplicative (beta=1.5)", rule_multiplicative),
        ("Conditional (beta=2, gamma=0.25)", rule_conditional)
    ]
    
    plt.figure(figsize=(10, 6))
    
    for name, rule in experiments:
        print(f"\nRunning Experiment: {name}")
        
        # Note: For constant rule, we might want to start with a higher c to actually see convergence
        # But to be fair to the "growth" strategies, we start all at c=1 (except pure constant maybe)
        # To strictly follow "c_k = constant", let's assume the rule handles the value.
        # I'll modify the logic slightly: initialize c=1 inside the solver, but the rule determines next.
        # For the "Constant" test, actually setting initial c=5 works better.
        
        # Wrapper to handle initial c boost for the Constant case specifically
        actual_rule = rule
        if "Constant" in name:
            # Hacky way to simulate starting at 5 and staying at 5
            # We will pass a modified rule to method_of_multipliers
            def const_wrapper(c, h, hp): return 5.0
            actual_rule = const_wrapper
            
        x_opt, history = method_of_multipliers(Q, A, b, actual_rule)
        
        # Calculate Relative Error
        errors = []
        for x_k in history:
            err = np.linalg.norm(x_star - x_k) / x_star_norm
            errors.append(err)
            
        iters = range(1, len(errors) + 1)
        plt.semilogy(iters, errors, marker='o', linestyle='-', label=name)
        
        print(f"  Final Objective: {objective_function(Q, x_opt):.4f}")
        print(f"  Final Relative Error: {errors[-1]:.2e}")

    plt.title("Relative Error vs Iterations for Method of Multipliers")
    plt.xlabel("Iteration k")
    plt.ylabel(r"Relative Error $\frac{||x^* - x^{(k)}||}{||x^*||}$")
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig("method_of_multipliers_plot.png")
    print("\nPlot saved to 'method_of_multipliers_plot.png'")
    # plt.show() # Uncomment if running locally

if __name__ == "__main__":
    run_experiments()
