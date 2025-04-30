import numpy as np
import matplotlib.pyplot as plt
import global_function_for_linear_regression as gflr

x_train = np.array([1.0, 2.0])          
y_train = np.array([300.0, 500.0])    

w_init = 0
b_init = 0 
iterations = 10000
tmp_alpha = 1.0e-2

print(" running gradient descent : ")
w_final, b_final, J_hist, p_hist = gflr.gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, gflr.compute_cost, gflr.compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
# ax1.plot(J_hist[:100])
# ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
# ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
# ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
# ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
# plt.show()
