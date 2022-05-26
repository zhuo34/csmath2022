import autograd.function as F


class Variable:
    def __init__(self, value, require_grad=False) -> None:
        self.value = value if not isinstance(value, Variable) else value.value
        self._require_grad = require_grad
        self.grad = 0 if require_grad else None
        self.grad_fn = None

    @property
    def require_grad(self):
        return self._require_grad
    
    @property
    def is_leaf(self):
        return self.grad_fn is None or not self.require_grad

    def __repr__(self) -> str:
        return str(self.value)

    def zero_grad(self):
        if self.require_grad:
            self.grad = 0
    
    def data(self):
        return self.value
    
    def backward(self, gradient=None, retain_graph=False):
        if not self.is_leaf:
            self.grad = 1 if gradient is None else gradient
            # BFS
            queue = [self]
            while len(queue):
                v = queue.pop(0)
                # if this node is a leaf node, do not calculate gradients
                if v.is_leaf:
                    continue
                queue += [v for v in v.grad_fn.next_vars if v not in queue]
                v.grad_fn.backward(retain_graph=retain_graph)

    def __add__(self, v):
        return F.add(self, v)

    def __sub__(self, v):
        return F.sub(self, v)

    def __mul__(self, v):
        return F.mul(self, v)

    def __truediv__(self, v):
        return F.div(self, v)
