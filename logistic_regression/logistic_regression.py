import numpy as np
import matplotlib.pyplot as plt
import global_functions_logistic_regression as gf
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
# plt.style.use('./logistic_regression/deeplearning.mplstyle')
from sklearn.linear_model import LogisticRegression


# # Input is an array.
# input_array = np.array([1,2,3])
# exp_array = np.exp(input_array)

# print("Input to exp:", input_array)
# print("Output of exp:", exp_array)

# # Input is a single number
# input_val = 1
# exp_val = np.exp(input_val)

# print("Input to exp:", input_val)
# print("Output of exp:", exp_val)

# # Generate an array of evenly spaced values between -10 and 10
# z_tmp = np.arange(-10,11)
# print(z_tmp)

# # Use the function implemented above to get the sigmoid values
# y = gf.sigmoid(z_tmp)
# print(y)

# # Code for pretty printing the two arrays next to each other
# np.set_printoptions(precision=3)
# print("Input (z), Output (sigmoid(z))")
# print(np.c_[z_tmp, y])

# #! Plot z vs sigmoid(z)
# # fig,ax = plt.subplots(1,1,figsize=(5,3))
# # ax.plot(z_tmp, y, c="b")

# # ax.set_title("Sigmoid function")
# # ax.set_ylabel('sigmoid(z)')
# # ax.set_xlabel('z')
# # draw_vthresh(ax,0)
# # plt.show()

# #! interactive screen:
# x_train = np.array([0., 1, 2, 3, 4, 5])
# y_train = np.array([0,  0, 0, 1, 1, 1])

# w_in = np.zeros((1))
# b_in = 0

# plt.close('all')
# addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
# plt.show()


# ? using scikit learn for makeing logistic regression model :

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])


# lr_model = LogisticRegression()
# lr_model.fit(X, y)

# y_pred = lr_model.predict(X)

# print("Prediction on training set:", y_pred)

# print("Accuracy on training set:", lr_model.score(X, y))

w_init = np.zeros(X.shape[1])
b_init = 0

w , b ,j_history = gf.gradient_descent(X , y, w_in=w_init , b_in=b_init ,alpha=0.01 , num_iters=10000)

y_pred = np.array([] , dtype=np.int64)

X_test = np.array([[0.2 , 1.9] , [0.4 , 1.6] , [0.7 , 1.2], [4 , 5.5] , [10 , 1.3] , [2.3 , 12.56] , [0.01 , 123.6] , [0.0001 , 12.56]])
for i in range(X_test.shape[0]):
    prediction = gf.sigmoid(np.dot(X_test[i], w) + b)

    # Append the prediction to y_pred and assign the result back to y_pred
    y_pred = np.append(y_pred, int(1 if prediction >= 0.5 else 0))

print("y prediction : ", y_pred)
