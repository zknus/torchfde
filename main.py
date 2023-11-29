from torchfde import fdeint
import torch
import math

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def f(t, y):

        # y = (40320 / math.gamma(9-0.5)) * (t ** (8-0.5)) - 3 * ((math.gamma(5+0.5)/math.gamma(5-0.5)) * (t ** (4-0.5/2))) + 9/4 * math.gamma(1+0.5) + ( (3/2 * (t ** 0.5/2) )- t **4) **3 -(y ** (3/2))
        y = (2 / math.gamma(3-0.5)) * (t**(2-0.5)) - (1 / math.gamma(2-0.5)) * (t**(1-0.5)) - y + t**2 -t
        return torch.tensor(y, dtype=torch.float32)


    y0 = torch.tensor([0])
    t = 40
    beta = 0.5

    y = fdeint(f, y0, beta, t, step_size=0.1,method='predictor')
    print("Euler Predictor: ", y.item())
    y = fdeint(f, y0, beta, t, step_size=0.1,method='corrector',options={'corrector_step':2})
    print("Euler Corrector: ", y.item())

    y_gt = t ** 2 - t
    print("ground truth: ", y_gt)