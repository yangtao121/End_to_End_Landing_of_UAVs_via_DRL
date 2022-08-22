import numpy as np


def sat_gd(u, a):
    u_ = np.linalg.norm(u, ord=np.inf)

    if u_ > a:
        out = a * (u / u_)
    else:
        out = u

    return out


if __name__ == '__main__':
    a = 2

    u = np.eye(3)
    b = np.array([0.1, .2, .3])

    # out = sat_gd(u, a)
    c = u @ np.transpose(b)
    print(np.squeeze(c, axis=0))
