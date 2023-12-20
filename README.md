# Fractional Differential Equation Solver based on Pytorch
## how to use
```
from torchfde import fdeint
y = fdeint(f, y0, beta, t, step_size=0.1,method='predictor')
# f: function can be one neural network
# y0: initial value
# beta: order of fractional differential equation
# t: integration time
# step_size: step size of integration
# method: solver method, in ['predictor','corrector','implicitl1','gl','trap']

y = fdeint(f, y0, beta, t, step_size=0.1,method='predictor',options={'memory':399})
# options: options for solver
# memory: memory size of predictor method

y = fdeint(f, y0, beta, t, step_size=0.1,method='corrector',options={'corrector_step':2})
# corrector_step: correct times of corrector method

```
