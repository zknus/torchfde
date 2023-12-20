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


## cite us if you use this code in your research
```
@inproceedings{FROND2023,
    title={Unleashing the Potential of Fractional Calculus in Graph Neural Networks},
    author = {Qiyu Kang and Kai Zhao and Qinxu Ding and Feng Ji and Xuhao Li and Wenfei Liang and Yang Song and Wee Peng Tay},
    booktitle = {Adv. Neural Inform. Process. Syst. Workshop on Machine Learning and the Physical Sciences},
    year = {2023},
}
```
```
@INPROCEEDINGS{ZhaKanSon:C24,
author = {Qiyu Kang and Kai Zhao and Yang Song and Yihang Xie and Yanan Zhao and Sijie Wang and Rui She and Wee Peng Tay},
title = {Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: {A} Robustness Study},
booktitle = {Proc. AAAI Conference on Artificial Intelligence},
volume = {},
pages = {},
month = {Feb.},
year = {2024},
address = {Vancouver, Canada},
}
```

