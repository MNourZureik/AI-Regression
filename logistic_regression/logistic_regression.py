import numpy as np
import matplotlib.pyplot as plt
import global_functions_logistic_regression as gf
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./logistic_regression/deeplearning.mplstyle')
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


#? using scikit learn for makeing logistic regression model :

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])


lr_model = LogisticRegression()
lr_model.fit(X, y)

y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)

print("Accuracy on training set:", lr_model.score(X, y))