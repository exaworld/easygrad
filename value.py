import math

class Value:
    """
    A class representing a scalar value with support for automatic differentiation.

    Methods:
    - __init__(self, data, _children=(), _op='', label=''): Initializes a Value object with data, gradient, and bookkeeping variables.
    - __repr__(self): Returns a string representation of the Value object.
    - __add__(self, other): Defines addition operation between two Value objects or a Value object and a scalar.
    - __mul__(self, other): Defines multiplication operation between two Value objects or a Value object and a scalar.
    - __pow__(self, other): Defines exponentiation operation of a Value object to a scalar power.
    - tanh(self): Applies the hyperbolic tangent activation function to the Value object.
    - exp(self): Applies the exponential function to the Value object.
    - relu(self): Applies the rectified linear unit (ReLU) activation function to the Value object.
    - sigmoid(self): Applies the sigmoid activation function to the Value object.
    - backward(self): Performs backpropagation to compute gradients of all connected Value objects.
    - __radd__(self, other): Defines reverse addition operation when the left operand is not a Value object.
    - __rmul__(self, other): Defines reverse multiplication operation when the left operand is not a Value object.
    - __neg__(self): Negates the Value object.
    - __sub__(self, other): Defines subtraction operation between two Value objects or a Value object and a scalar.
    - __rsub__(self, other): Defines reverse subtraction operation when the left operand is not a Value object.
    - __truediv__(self, other): Defines division operation between two Value objects or a Value object and a scalar.
    - __rtruediv__(self, other): Defines reverse division operation when the left operand is not a Value object.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        """
        Initializes a Value object.

        Args:
        - data: The scalar value of the object.
        - _children: A tuple containing the child Value objects that contributed to the current object's value.
        - _op: The operation symbol associated with the current object.
        - label: A label to identify the current object (optional).
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
   
    def __repr__(self):
        return f"Value(data={self.data})" 
    
    # The __add__() method defines the behavior of the + operator between two Value objects or a Value object and a scalar.
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # Using the chain rule, the derivative of the final output (dF) with respect to the current child
        # is equal to the local derivative (d of parent with respect to child) times the dF with respect to its parent.
        # For example, c = a + b; dO/dA = dC/dA * dO/dC.

        def _backward():
            self.grad += 1.0 * out.grad  # dC/dA = 1.0, dO/dC = out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        """
        Defines multiplication operation between two Value objects or a Value object and a scalar.

        Args:
        - other: The Value object or scalar to multiply with.

        Returns:
        - out: A new Value object representing the result of the multiplication.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Defines exponentiation operation of a Value object to a scalar power.

        Args:
        - other: The scalar power to raise the Value object to.

        Returns:
        - out: A new Value object representing the result of the exponentiation.
        """
        assert isinstance(other, (int, float))  # Should be int or float
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        """
        Applies the hyperbolic tangent activation function to the Value object.

        Returns:
        - out: A new Value object representing the result of the tanh operation.
        """
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
            
        return out

    def exp(self):
        """
        Applies the exponential function to the Value object.

        Returns:
        - out: A new Value object representing the result of the exponential function.
        """
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        """
        Calculates the natural logarithm of the Value object.

        Returns:
        - out: A new Value object representing the natural logarithm.
        """
        assert self.data > 0, "Value must be greater than 0 for logarithm operation"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """
        Applies the rectified linear unit (ReLU) activation function to the Value object.

        Returns:
        - out: A new Value object representing the result of the ReLU operation.
        """
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        """
        Applies the sigmoid activation function to the Value object.

        Returns:
        - out: A new Value object representing the result of the sigmoid operation.
        """
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        """
        Performs backpropagation to compute gradients of all connected Value objects.
        """
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    # For binary output with one neuron only
    def neg_log_loss(self, y):
        """
        Calculates the negative log loss (binary cross-entropy loss).

        Args:
        - y: The true label (0 or 1).
        - self: The predicted probability of the positive class.

        Returns:
        - out: A new Loss object representing the negative log loss.
        """
        assert y in [0, 1], "True label must be 0 or 1"
        assert 0 <= self.data <= 1, "Predicted probability must be between 0 and 1"

        loss = -(y * math.log(self.data) + (1 - y) * math.log(1 - self.data))
        out = Value(loss, (self,), 'neg_log_loss')

        def _backward():
            self.grad = (self.data - y)  # dLoss/dp
        out._backward = _backward

        return out

    def __radd__(self, other):
        """
        Defines reverse addition operation when the left operand is not a Value object.

        Args:
        - other: The Value object or scalar that is being added to the Value object.

        Returns:
        - out: A new Value object representing the result of the reverse addition.
        """
        return self + other

    def __rmul__(self, other):
        """
        Defines reverse multiplication operation when the left operand is not a Value object.

        Args:
        - other: The Value object or scalar that is being multiplied with the Value object.

        Returns:
        - out: A new Value object representing the result of the reverse multiplication.
        """
        return self * other
    
    def __neg__(self):
        """
        Negates the Value object.

        Returns:
        - out: A new Value object representing the negation of the original Value object.
        """
        return self * -1

    def __sub__(self, other):
        """
        Defines subtraction operation between two Value objects or a Value object and a scalar.

        Args:
        - other: The Value object or scalar to subtract from the Value object.

        Returns:
        - out: A new Value object representing the result of the subtraction.
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Defines reverse subtraction operation when the left operand is not a Value object.

        Args:
        - other: The Value object or scalar that is being subtracted from the Value object.

        Returns:
        - out: A new Value object representing the result of the reverse subtraction.
        """
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __truediv__(self, other):
        """
        Defines division operation between two Value objects or a Value object and a scalar.

        Args:
        - other: The Value object or scalar to divide the Value object by.

        Returns:
        - out: A new Value object representing the result of the division.
        """
        return self * other**(-1)
    
    def __rtruediv__(self, other):
        """
        Defines reverse division operation when the left operand is not a Value object.

        Args:
        - other: The Value object or scalar that is being divided by the Value object.

        Returns:
        - out: A new Value object representing the result of the reverse division.
        """
        other = other if isinstance(other, Value) else Value(other)
        return other * self**(-1)