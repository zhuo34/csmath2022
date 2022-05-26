from autograd import Variable
import autograd.function as F

x1 = Variable(2, require_grad=True)
x2 = Variable(5, require_grad=True)

v1 = F.log(x1)
v2 = x1 * x2
v3 = F.sin(x2)
v4 = v1 + v2
v5 = v4 - v3

print("AD for `y = f(x1, x2) = ln(x1) + x1*x2 - sin(x2)`")
y = v5

print(f"y = f({x1}, {x2}) = {y.data():.3f}")
y.backward(retain_graph=True)
print(f"dy/dx1 = {x1.grad}, dy/dx2 = {x2.grad:.3f}")

def gd(f, args0, lr=0.1, epsilon=1e-6, max_step=100, output_info=False):
    args = [Variable(arg0, require_grad=True) for arg0 in args0]
    z = f(*args)
    step = 0
    if output_info:
        arg_step = []
    while True:
        z_last = z.data()
        z.backward()
        args = [Variable(arg.data() - lr * arg.grad, require_grad=True) for arg in args]
        z = f(*args)
        delta = abs(z.data() - z_last)
        if output_info:
            arg_step.append([arg.data() for arg in args])
        step += 1
        if delta < epsilon or step == max_step:
            break
    if output_info:
        return arg_step[-1], arg_step
    return [arg.data() for arg in args]

print("gradient descent with AD for `z = x^2 + y^2`")
f = lambda x, y: x * x + y * y
x, info = gd(f, [2, 5], output_info=True)
print(x[0], x[1])
