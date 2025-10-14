# Making a Lveneberg-Marquardt (LM) algorithm

import numpy as np
import matplotlib.pylab as plt


def grad(C, circle_data):
    C = C.reshape(-1)  # ensures shape (2,)

    grad_matrix = np.zeros((3, 25))

    for p in range(25):
        diff = C - circle_data[:, p]
        norm_diff = np.linalg.norm(diff)

        grad_matrix[0, p] = diff[0] / norm_diff
        grad_matrix[1, p] = diff[1] / norm_diff
        grad_matrix[2, p] = -1

    return grad_matrix


def total_grad(grad_matrix, f_p):
    return 2.0 * grad_matrix @ f_p  # shape (3,1)


def f_value(C, R, circle_data):
    f_p = np.zeros((25, 1))
    for p in range(25):
        f_p[p, 0] = np.linalg.norm(C - circle_data[:, p]) - R
    return f_p


circle_data = np.load("./Development/Data/circle_data_1.npy")
print(circle_data)
print(np.shape(circle_data))  # 2 rows x 25 columns.

# Creating the matrices A & b, A = 25 * 3, b = 25 * 1

A = np.zeros((25, 3))  # Matrix A empty

for i in range(25):  # i in range of the rows
    A[i, 0] = 1  # column zero, all 1's
    A[i, 1] = -2 * circle_data[0, i]  # x_i^T is just the row vector
    A[i, 2] = -2 * circle_data[1, i]

print(A)

b = np.zeros((25, 1))  # Matrix b empty

for i in range(25):
    b[i, 0] = -(np.linalg.norm(circle_data[:, i]) ** 2)

print(b)

x_atomic, residuals, rank, s = np.linalg.lstsq(
    A, b, rcond=None
)  # Atomic problem very easy to solve

print("Shape of solution:", x_atomic.shape)
y_optimal = x_atomic[0]
c_ls = x_atomic[1:]

print(c_ls[0])
print(c_ls[1])


# Computing R:

R_ls = np.sqrt(np.linalg.norm(c_ls) ** 2 - y_optimal)
print(f"The optimal value of R using least squares is {R_ls}")

# Now that the first problem is solved, we can start implementing the LM:

eps = 1e-6
lambda_0 = 1
k = 0

#print("C_ls")
#print(c_ls)
#c_ls = np.array([-1.0621100344153003, 0.9947560513264783])

grad_f = np.zeros((3, 25))
C = c_ls.reshape(-1)  # ensure shape (2,)
R = float(R_ls)  # ensure scalar
lamba = lambda_0
gradient_norms = []

while True:
    f_p = f_value(C, R, circle_data)  # (25,1)
    grad_f = grad(C, circle_data)  # (3,25)
    total_grad_f = total_grad(grad_f, f_p)  # (3,1)

    # Stopping condition
    if np.linalg.norm(total_grad_f) < eps:
        break

    gradient_norms.append(np.linalg.norm(total_grad_f))

    # Build A and b
    grad_f_T = grad_f.T  # (25,3)
    lamba_eye = np.sqrt(lamba) * np.eye(3)
    A = np.vstack((grad_f_T, lamba_eye))

    x_k = np.vstack((C.reshape(2, 1), [[R]]))  # (3,1)
    b = np.vstack((grad_f_T @ x_k - f_p, np.sqrt(lamba) * x_k))

    x_pred, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Trial update
    C_trial = x_pred[:2].ravel()
    R_trial = float(x_pred[2])

    f_x_k = float(f_p.T @ f_p)
    f_p1 = f_value(C_trial, R_trial, circle_data)
    f_x_k1 = float(f_p1.T @ f_p1)

    if f_x_k1 < f_x_k:  # Accept step
        C = C_trial
        R = R_trial
        lamba *= 0.7
    else:  # Reject step
        lamba *= 2

    k += 1

# Plotting gradient norm vs iteration with log scale
plt.figure(figsize=(10, 6))
plt.semilogy(gradient_norms)
plt.xlabel("Iteration")
plt.ylabel("Norm of Gradient (log scale)")
plt.title("Gradient Norm vs Iteration (LM) - Log Scale")
plt.grid(True)
plt.show()

print(" --- gradient_norms --- ")
print(gradient_norms)