from polynomial_Reg_model import polynomial_rmse_ridge


def sweep(max_deg=10):
    best = (None, float('inf'))
    for d in range(1, max_deg + 1):
        rmse, alpha, _ = polynomial_rmse_ridge(d)
        print(f"deg={d}, rmse={rmse:.6f}, alpha={alpha}")
        if rmse < best[1]:
            best = (d, rmse)
    print(f"BEST: degree={best[0]}, rmse={best[1]:.6f}")


if __name__ == '__main__':
    sweep(10)
