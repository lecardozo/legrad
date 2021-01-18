# legrad

This is a very naive, non-optimized automatic differentiation library implemented in Python for educational purposes. Heavily inspired by [autograd](https://github.com/HIPS/autograd)/[jax](https://github.com/google/jax) and [pytorch](https://github.com/pytorch/pytorch) implementations.

## Example

```python
from legrad import Variable

# Don't mind this, is just used to make the output pretty
def ndigits(x):
    return int(np.log10(x)) + 1

# Data
x = Variable('x', np.array([1,2,3]), trainable=False)
y_true = Variable('y_true', np.array([2, 4, 6]), trainable=False)

# Parameters
W = Variable('w', 10)
b = Variable('b', 1)

n_epochs = 10000
losses = np.zeros(n_epochs)
for i in range(n_epochs):
    y_pred = model(x, W, b)
    loss = loss_fn(y_true, y_pred)
    losses[i] = loss.value
    if i % 100 == 0:
        padding = ndigits(n_epochs) - ndigits(i+1) - 1
        print(f"Epoch {str(i) + ' '*padding} | Loss: {loss.value}")

    # Compute gradients through reverse-mode differentiation (a.k.a. backpropagation)
    loss.backward()
    
    # Update parameters values
    loss.update(step_size=1e-3)
```