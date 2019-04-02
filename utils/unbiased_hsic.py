import torch


def _center_kern(K, m):
    e = K.new_ones(m, 1)
    K = K - e @ e.t() @ K / m
    K = K - K @ e @ e.t() / m
    return K


def _compute_h_vec(Ks, Ls, m):
    e = Ks.new_ones(m, 1)

    t1 = (m - 2)**2 * (Ks * Ls) @ e
    t2 = (m - 2) * ((torch.sum(Ks * Ls) * e) - (Ks @ (Ls @ e)) - (Ls @ (Ks @ e)))
    t3 = m * (Ks @ e) * (Ls @ e)
    t4 = Ls.sum() * (Ks @ e)
    t5 = Ks.sum() * (Ls @ e)
    t6 = (e.t() @ Ks) @ (Ls @ e) * e

    h_xy = t1 + t2 - t3 + t4 + t5 - t6

    return h_xy


def _HSIC_unbiased(K, L, m):
    e = K.new_ones(m, 1)

    Ks = _center_kern(K, m)
    Ls = _center_kern(L, m)

    Ks = Ks - Ks.diag().diag()
    Ls = Ls - Ls.diag().diag()

    t1 = torch.sum(Ks * Ls)
    t2 = Ks.sum() * Ls.sum() / ((m - 1) * (m - 2))
    t3 = (e.t() @ Ks) @ (Ls @ e) * 2 / (m - 2)

    return (t1 + t2 - t3) / (m * (m - 3))


def _rbf_kernel(x_gram, y_gram, sigma_1=1, sigma_2=0.5):
    x_sqnorms = x_gram.diag()
    y_sqnorms = y_gram.diag()

    gamma_first = sigma_1  # 1. / (2 * sigma_first**2) TODO
    gamma_second = sigma_2  # 1. / (2 * sigma_second**2) TODO

    kernel_x = torch.exp(-gamma_first * (-2 * x_gram + x_sqnorms.unsqueeze(0) + x_sqnorms.unsqueeze(1)))
    kernel_y = torch.exp(-gamma_second * (-2 * y_gram + y_sqnorms.unsqueeze(0) + y_sqnorms.unsqueeze(1)))

    return kernel_x, kernel_y


def variance_adjusted_unbiased_HSIC(x, y, sigma_1=1., sigma_2=0.5, use_rbf=False):
    x_gram = x @ x.t()
    y_gram = y @ y.t()

    if use_rbf:
        kernel_x, kernel_y = _rbf_kernel(x_gram, y_gram, sigma_1, sigma_2)
    else:
        kernel_x, kernel_y = x_gram, y_gram

    m = x.size(0)

    constant = (1 / (4 * m)) * (1 / ((m - 1) * (m - 2) * (m - 3))**2)

    K = _center_kern(kernel_x, m)
    L = _center_kern(kernel_y, m)

    Ks = K - K.diag().diag()
    Ls = L - L.diag().diag()

    h_xy = _compute_h_vec(Ks, Ls, m)

    R_xy = constant * (h_xy.t() @ h_xy)

    HSIC_xy = _HSIC_unbiased(K, L, m)

    variance = (16 / m) * (R_xy - (HSIC_xy**2))

    variance_adjusted_HSIC = HSIC_xy / variance

    return variance_adjusted_HSIC


def test():
    import pytest
    data_first = torch.FloatTensor([[1, 3], [2, 4], [1, 4], [10, 22], [113, 22]])
    data_second = torch.FloatTensor([[1], [98], [16], [98], [99]])
    sigma_first = torch.FloatTensor(1)
    sigma_second = torch.FloatTensor(1)
    hsic = variance_adjusted_unbiased_HSIC(data_first, data_second, sigma_first, sigma_second, use_rbf=False).item()
    assert pytest.approx(hsic, 1.e-6) == 4.05317e-6


if __name__ == "__main__":
    test()
