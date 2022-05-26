import math


class Contex:
    def __init__(self) -> None:
        self.saved_values = ()
    
    def save_for_backward(self, *args):
        self.saved_values = list(args)


class Function:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *args):
        from .variable import Variable
        ctx = Contex()
        args_value = [arg.data() for arg in args]
        y = cls.forward(ctx, *args_value)

        y = Variable(y, require_grad=len([arg for arg in args if arg.require_grad]) > 0)
        if y.require_grad:
            y.grad_fn = BackwardNode(y, cls, ctx, args)
        return y


class BackwardNode:
    def __init__(self, var, fn, ctx, next_vars) -> None:
        self.fn = fn
        self.var = var
        self.ctx = ctx
        self.next_vars = next_vars
    
    def backward(self, retain_graph=False):
        grads = self.fn.backward(self.ctx, self.var.grad)
        for next_var, grad in zip(self.next_vars, grads):
            # if the input variable require autograd, then accumulate its gradient
            if next_var.require_grad:
                next_var.grad += grad
        if not retain_graph:
            self.var.grad_fn = None
            self.var.ctx = None


class _add(Function):
    @staticmethod
    def forward(ctx, x, y):
        z = x + y
        ctx.save_for_backward(x, y)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

def add(a, b):
    return _add.apply(a, b)


class _sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        z = x - y
        ctx.save_for_backward(x, y)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, -grad_output

def sub(a, b):
    return _sub.apply(a, b)


class _mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        z = x * y
        ctx.save_for_backward(x, y)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_values
        return grad_output * y, grad_output * x

def mul(a, b):
    return _mul.apply(a, b)


class _div(Function):
    @staticmethod
    def forward(ctx, x, y):
        z = x / y
        ctx.save_for_backward(x, y)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_values
        return grad_output / y, grad_output * (-x / (y ** 2))

def div(a, b):
    return _div.apply(a, b)


class _exp(Function):
    @staticmethod
    def forward(ctx, x):
        z = math.exp(x)
        ctx.save_for_backward(x)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_values
        return grad_output * math.exp(x),

def exp(a):
    return _exp.apply(a)


class _log(Function):
    @staticmethod
    def forward(ctx, x):
        z = math.log(x)
        ctx.save_for_backward(x)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_values
        return grad_output / x,

def log(a):
    return _log.apply(a)


class _sin(Function):
    @staticmethod
    def forward(ctx, x):
        z = math.sin(x)
        ctx.save_for_backward(x)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_values
        return grad_output * math.cos(x),

def sin(a):
    return _sin.apply(a)


class _cos(Function):
    @staticmethod
    def forward(ctx, x):
        z = math.cos(x)
        ctx.save_for_backward(x)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_values
        return grad_output * -math.sin(x),

def cos(a):
    return _cos.apply(a)
