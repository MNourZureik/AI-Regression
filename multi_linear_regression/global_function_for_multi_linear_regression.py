import numpy as np # type: ignore
import copy, math
def predict_single_loop(x, w, b): 
    n = x.shape[0]
    p = 0
    
    for i in range(n): 
        p = p + x[i] * w[i]         
    p += b           
         
    return p

def predict_using_dot(x, w, b): 
    p = np.dot(x, w) + b     
    return p    

def compute_total_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b          
        cost = cost + (f_wb_i - y[i])**2      
    cost = cost / (2 * m)                      
    return cost

def compute_gradient(X , Y , w , b):
    
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.    
    
    for i in range(m):
        f_wb_i = np.dot(w , X[i]) + b
        err_cost = f_wb_i - Y[i]
        for j in range(n):
            dj_dw += err_cost * X[i][j]
        dj_db += err_cost
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw , dj_db


def gradient_descent(X , Y, w_in ,b_in ,learning_rate ,compute_total_cost , compute_gradient_function , num_iters):
    
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw , dj_db = compute_gradient(X, Y , w ,b)
        
        w = w - (learning_rate * dj_dw)
        b = b - (learning_rate * dj_db)
        
        if i < num_iters:
            j_history.append(compute_total_cost(X, Y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")
        
    return w , b , j_history
        