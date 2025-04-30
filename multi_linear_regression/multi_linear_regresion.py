import numpy as np
import global_function_for_multi_linear_regression as gflr
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print (f"X_train :{X_train[0]}\n         {X_train[1]}\n         {X_train[2]}")
print("")
print (f"y_train :{y_train}")

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print("")
print(f"w_init shape: {w_init.shape}\nb_init type: {type(b_init)}")

f_wb = gflr.predict_single_loop(X_train[0] , w_init , b_init)

print("")
print(f"f_wb result for predict_single_loop : {f_wb}")

f_wb = gflr.predict_using_dot(X_train[0] , w_init , b_init)

print("")
print(f"f_wb result for predict_using_dot : {f_wb}")

total_cost = gflr.compute_total_cost(X_train , y_train , w_init , b_init)

print("")
print(f"total cost result for compute_total_cost for optimal w : {total_cost}")

initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7
w_final , b_final ,j_history = gflr.gradient_descent(X_train , y_train , initial_w , initial_b ,alpha, gflr.compute_total_cost , gflr.compute_gradient, iterations)

print("")
print(f"b,w found by gradient descent: b: {b_final:0.2f} , w: {w_final} ")

m = X_train.shape[0]
for i in range(m):
    print(f" f_predection : {np.dot(X_train[i] , w_final) + b_final:0.2f} ::: target y : {y_train[i]}")
    
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(j_history)
ax2.plot(100 + np.arange(len(j_history[100:])), j_history[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()