class Variable:
    """
    Data container for automatic differentiable operations.

    You can see this class as a Node in a computation directed acyclic graph.
    Each variable, if not a leaf in the graph, stores the result of the operation
    that was applied to it and the function to compute the Vector-Jacobian Produt (VJP)
    during backpropagation.

    Forward pass:
        x = Variable("x", 1, trainable=False)
        W = Variable("W", np.random.random())
        b = Variable("b", np.random.random())
        y_true = Variable("y", 2, trainable=False)

        y_pred = x * W + b
        loss = (y_true - y_pred)**2
    
    Backward pass:
        loss.backward()
    
    Parameter update:
        loss.update()

    """
    def __init__(self, name, value, parents=[], vjp=None, trainable=True):
        self.name = name
        self.value = value
        self.parents = parents
        self.vjp = vjp
        self.grad = None
        self.trainable = trainable
    
    def __add__(self, other):
        value = self.value + other.value
        def vjp(g):
            return g, g
        return Variable(f'{self.name}+{other.name}', value, parents=[self, other], vjp=vjp)
    
    def __sub__(self, other):
        value = self.value - other.value
        def vjp(g):
            return g, -g
        return Variable(f'{self.name}-{other.name}', value, parents=[self, other], vjp=vjp)

    def __mul__(self, other):
        value = self.value * other.value
        def vjp(g):
            return g * other.value, g * self.value
        return Variable(f'{self.name}*{other.name}', value, parents=[self, other], vjp=vjp)
    
    def dot(self, other):
        value = np.dot(self.value, other.value)
        def vjp(g):
            return g * other.value, g * self.value
        return Variable(f'{self.name}.dot({self.other})', value, parents=[self, other], vjp=vjp)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            value = self.value / other
        elif isinstance(other, Variable):
            value = self.value / other.value
        else:
            raise TypeError(f"Can't divide Variable by object of type {type(other)}")

        def vjp(g):
            other_value = other.value if isinstance(other, Variable) else other
            return g * (1/other_value), g * self.value

        other_name = other.name if isinstance(other, Variable) else other
        parents = [self]
        if isinstance(other, Variable):
            parents.append(other)

        return Variable(f'{self.name}/{other_name}', value, parents=parents, vjp=vjp)

    def __pow__(self, power):
        value = self.value ** power
        def vjp(g):
            return g * (power * (self.value**(power - 1))),
        return Variable(f'{self.name}^{power}', value, parents=[self], vjp=vjp)

    def mean(self):
        value = np.mean(self.value)
        def vjp(g):
            g_reshaped = np.ones_like(self.value)
            return g_reshaped / self.value.shape[0], 
        return Variable(f'Mean({self.name})', value, parents=[self], vjp=vjp)

    def sqrt(self):
        value = np.sqrt(self.value)
        def vjp(g):
            return 0.5 * (g**(-0.5)),
        return Variable(f'sqrt({self.name})', value, parents=[self], vjp=vjp)

    def backward(self, grad=1, verbose=False):
        if verbose:
            print(f"incoming grad to {self.name} = {grad}")

        if self.grad is not None:
            self.grad += grad
        else:
            self.grad = grad

        if self.vjp:
            grads = self.vjp(grad)
            for parent, gradient in zip(self.parents, grads):
                parent.backward(gradient, verbose=verbose)
    
    def update(self, step_size=1e-3):
        if self.trainable:
            self.value = self.value - (self.value * step_size)

        for parent in self.parents:
            parent.update(step_size)
            
    
    def __repr__(self):
        return f"Variable(value={self.value}, parents={self.parents})"

