"""Pure-stdlib statistics: OLS, t quantiles, prediction intervals.

Deliberately no numpy/scipy: every calculation here can be read line by line
and re-checked by hand. Matrix inversion is Gauss-Jordan on tiny matrices
(models have at most a handful of predictors).
"""
import math


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def variance(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def sd(xs):
    return math.sqrt(variance(xs))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def ols(X, y):
    """Ordinary least squares. X rows are predictor vectors (no intercept
    handling here — include a constant column explicitly). Returns beta,
    residual standard error, and the (X'X)^-1 matrix.
    """
    n, p = len(X), len(X[0])
    Xt = [[X[i][j] for i in range(n)] for j in range(p)]
    XtX = [[sum(Xt[a][k] * Xt[b][k] for k in range(n)) for b in range(p)]
           for a in range(p)]
    Xty = [sum(Xt[a][k] * y[k] for k in range(n)) for a in range(p)]
    inv = invert(XtX)
    beta = [sum(inv[a][b] * Xty[b] for b in range(p)) for a in range(p)]
    fitted = [sum(beta[j] * X[i][j] for j in range(p)) for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]
    dof = n - p
    rss = sum(r * r for r in resid)
    rse = math.sqrt(rss / dof) if dof > 0 else float("nan")
    tss = sum((v - mean(y)) ** 2 for v in y)
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    return {
        "beta": beta, "fitted": fitted, "resid": resid, "rse": rse,
        "r2": r2, "adj_r2": adj, "dof": dof, "n": n, "p": p,
        "cov": [[inv[a][b] * rse * rse for b in range(p)] for a in range(p)],
        "se_beta": [math.sqrt(inv[j][j]) * rse for j in range(p)],
    }


def invert(A):
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-14:
            raise ValueError("singular design matrix")
        M[col], M[pivot] = M[pivot], M[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            M[r] = [rv - f * cv for rv, cv in zip(M[r], M[col])]
    return [row[n:] for row in M]


# Exact t quantiles via the regularized incomplete beta function
# I_x(a,b) (Numerical Recipes Lentz continued fraction). The Cornish-Fisher
# expansion above is retained only as a cross-check/fallback.
def _betacf(x, a, b):
    eps = 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(x, a, b):
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1 - x, b, a) / b


def t_cdf(t, dof):
    """Two-sided-consistent CDF of Student's t."""
    x = dof / (dof + t * t)
    ib = betai(x, dof / 2.0, 0.5)
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def t_critical(dof, alpha=0.05):
    """Exact two-sided critical value by bisection of the t CDF."""
    if dof <= 0:
        return normal_quantile(1 - alpha / 2)
    target = 1 - alpha / 2
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_cdf(mid, dof) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def normal_quantile(p):
    """Inverse standard normal CDF, Acklam algorithm."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def prediction_se(fit, x_new, rse):
    """Standard error for a *new* prediction at x_new: model-coefficient
    uncertainty plus residual scatter of a fresh observation.
    """
    p = fit["p"]
    cov = fit["cov"]
    var_mean = sum(cov[a][b] * x_new[a] * x_new[b]
                   for a in range(p) for b in range(p))
    return math.sqrt(max(var_mean, 0.0) + rse * rse)


def rmse(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))
