c = 1.0
alpha = 0.5
radius_scaling = lambda r: c / (alpha * r)

print(f"Initial radius: {radius_scaling(1)}")
print(f"Final radius: {radius_scaling(2)}")

import scipy.optimize as opt

def equation(alpha):
    r = 1
    initial_value = c / (alpha * r)
    r = 2
    new_value = c / (alpha * r)
    return new_value - initial_value / 2

alpha_solution = opt.fsolve(equation, 0.5)[0]
print(f"Calculated alpha: {alpha_solution}")