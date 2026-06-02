import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

RUN_MAIN = True
RUN_EXTRAPOLATION = True
RUN_SWEEP = False
RUN_SMOOTHNESS = True
RUN_LOSS_CMP = True
TRAIN_STRATEGY = "interior_weighted"
LOSS_MODE = "normalized"
N_TRAIN = 600
N_BANDS = 4

SEED = 42
import random
random.seed(SEED); np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

try:
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

PARAMS = dict(a=1.0, b=1.0, eta=np.pi / 2.0,
              D=51.28e4, rho=2700.0, h=0.002, m_u=2.7, ks=9.593e6)
N_T_NUM = 10
N_T_PINN = 8
OMEGA_MAX = 5100.0
F_MAX = OMEGA_MAX / (2 * np.pi)

class Dispersion:
    def __init__(self, **p):
        self.__dict__.update(p)
        self.nu = np.sqrt(self.ks / self.m_u)

    def S_np(self, kappa_b, kx_b, ky_a, N_T=N_T_NUM):
        se = np.sin(self.eta); asb = self.a / self.b * se
        res = 0.0 + 0.0j
        with np.errstate(divide="ignore", invalid="ignore"):
            for n in range(-N_T, N_T + 1):
                kxn = kx_b + 2.0 * np.pi * n
                lam = np.sqrt((kxn / kappa_b) ** 2 - 1.0 + 0j)
                gam = np.sqrt((kxn / kappa_b) ** 2 + 1.0 + 0j)
                dl = np.cos(se * ky_a)
                res += (np.sinh(gam * kappa_b * asb) / (dl - np.cosh(gam * kappa_b * asb)) / gam
                        - np.sinh(lam * kappa_b * asb) / (dl - np.cosh(lam * kappa_b * asb)) / lam)
        return res

    def F_np(self, omega, kx, ky):
        kappa = (self.rho * self.h * omega ** 2 / self.D) ** 0.25
        mu = self.m_u * omega ** 2 / (self.D * (1.0 - omega ** 2 / self.nu ** 2))
        lhs = 4.0 * self.b * kappa ** 3 / mu
        return (lhs - self.S_np(kappa * self.b, kx * self.b, ky * self.a)).real

    def residual_torch(self, omega, kx, ky, mode="normalized", N_T=N_T_PINN):
        se = float(np.sin(self.eta)); asb = self.a / self.b * se
        kappa_b = self.b * (self.rho * self.h * omega ** 2 / self.D) ** 0.25
        mu = self.m_u * omega ** 2 / (self.D * (1.0 - omega ** 2 / self.nu ** 2))
        lhs = 4.0 * self.b * kappa_b ** 3 / (self.b ** 3 * mu)
        kx_b = kx * self.b; ky_a = ky * self.a
        dl = torch.cos(se * ky_a)
        S = torch.complex(torch.zeros_like(kappa_b), torch.zeros_like(kappa_b))
        eps = 1e-12
        for n in range(-N_T, N_T + 1):
            kxn = kx_b + 2.0 * np.pi * n
            ratio = (kxn / kappa_b) ** 2
            lam = torch.sqrt((ratio - 1.0).to(torch.complex128))
            gam = torch.sqrt((ratio + 1.0).to(torch.complex128))
            kb = kappa_b.to(torch.complex128)
            S = S + (torch.sinh(gam * kb * asb) / (dl - torch.cosh(gam * kb * asb)) / (gam + eps)
                     - torch.sinh(lam * kb * asb) / (dl - torch.cosh(lam * kb * asb)) / (lam + eps))
        S = S.real
        if mode == "raw":
            return (lhs - S)
        return (lhs - S) / (1.0 + torch.abs(lhs) + torch.abs(S))

def brillouin_path(n_seg):
    kx1 = np.linspace(0, np.pi, n_seg, endpoint=False); ky1 = np.zeros(n_seg)
    kx2 = np.full(n_seg, np.pi); ky2 = np.linspace(0, np.pi, n_seg, endpoint=False)
    t = np.linspace(np.pi, 0, n_seg, endpoint=True); kx3, ky3 = t, t
    KX = np.concatenate([kx1, kx2, kx3]); KY = np.concatenate([ky1, ky2, ky3])
    S = np.concatenate([np.linspace(0, 1, n_seg, endpoint=False),
                        np.linspace(1, 2, n_seg, endpoint=False),
                        np.linspace(2, 3, n_seg, endpoint=True)])
    return KX, KY, S

def sample_irreducible_bz(n, kx_lo=0.0, kx_hi=np.pi, weight=None, rng=RNG):
    out = []
    while len(out) < n:
        kx = rng.uniform(0, np.pi); ky = rng.uniform(0, np.pi)
        if not (ky <= kx and kx_lo <= kx <= kx_hi):
            continue
        if weight == "corners":
            dX = np.hypot(kx - np.pi, ky - 0.0)
            dM = np.hypot(kx - np.pi, ky - np.pi)
            p = np.exp(-min(dX, dM) / (0.35 * np.pi))
            if rng.uniform() > 0.3 + 0.7 * p:
                continue
        out.append((kx, ky))
    return np.array(out)

def sample_region(name, n, rng=RNG):
    cfg = dict(near_Gamma=((0.12 * np.pi, 0.0), 0.16 * np.pi),
               near_X=((np.pi, 0.0), 0.18 * np.pi),
               near_M=((np.pi, np.pi), 0.18 * np.pi))
    (cx, cy), r = cfg[name]
    out = []
    while len(out) < n:
        kx = rng.uniform(0, np.pi); ky = rng.uniform(0, np.pi)
        if ky <= kx and (kx - cx) ** 2 + (ky - cy) ** 2 <= r ** 2:
            out.append((kx, ky))
    return np.array(out)

def numerical_roots_per_k(disp, KX, KY, S=None, n_omega=4000):
    grid = np.linspace(5.0, OMEGA_MAX, n_omega)
    if S is None:
        S = np.arange(len(KX))
    out = []
    for kx, ky, s in zip(KX, KY, S):
        vals = np.array([disp.F_np(g, kx, ky) for g in grid]); rr = []
        for j in range(len(grid) - 1):
            if vals[j] * vals[j + 1] < 0.0:
                try:
                    r = brentq(disp.F_np, grid[j], grid[j + 1],
                               args=(kx, ky), xtol=1e-10, maxiter=400)
                    if abs(disp.F_np(r, kx, ky)) < 1e-2:
                        rr.append(r / (2 * np.pi))
                except Exception:
                    pass
        out.append((s, np.array(sorted(rr))))
    return out

def test_metric(pinn_f, roots_per_k, tol_hz=15.0):
    errs, errs_cov, covered, total = [], [], 0, 0
    for ph, (_, rr) in zip(pinn_f, roots_per_k):
        for r in rr:
            total += 1
            d = np.min(np.abs(ph - r)); errs.append(d ** 2)
            if d < tol_hz:
                covered += 1; errs_cov.append(d ** 2)
    if total == 0:
        return None
    return dict(rmse=float(np.sqrt(np.mean(errs))),
                rmse_cov=float(np.sqrt(np.mean(errs_cov))) if errs_cov else float("nan"),
                coverage=covered / total, n=total)

def path_roughness(pinn_f):
    f = pinn_f * (2 * np.pi)
    d2 = f[2:] - 2 * f[1:-1] + f[:-2]
    return float(np.mean((d2 / OMEGA_MAX) ** 2))

def find_complete_gap(fs, fmax=F_MAX, n_bins=400):
    if len(fs) == 0:
        return None
    grid = np.linspace(0, fmax, n_bins); win = 1.5 * fmax / n_bins
    occ = np.array([np.any(np.abs(fs - f) <= win) for f in grid])
    best = (0.0, None, None); j = 0
    while j < n_bins:
        if not occ[j]:
            k = j
            while k < n_bins and not occ[k]:
                k += 1
            lo, hi = grid[j], grid[k - 1]
            if (hi - lo) > best[0] and lo > 5:
                best = (hi - lo, lo, hi)
            j = k
        else:
            j += 1
    return None if best[1] is None else (best[1], best[2])

if HAVE_TORCH:
    class FourierFeatures(nn.Module):
        def __init__(self, n_modes=6):
            super().__init__()
            self.register_buffer("j", torch.arange(1, n_modes + 1, dtype=torch.float64))

        def forward(self, k):
            ang = 2 * np.pi * k.unsqueeze(-1) * self.j
            feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
            return torch.cat([k, feats.flatten(1)], dim=-1)

    class BandPINN(nn.Module):
        def __init__(self, n_bands=N_BANDS, n_modes=6, hidden=(96, 96, 96),
                     omega_min=2.0, dscale=1400.0):
            super().__init__()
            self.M = n_bands; self.omega_min = omega_min; self.dscale = dscale
            self.ff = FourierFeatures(n_modes)
            d = 2 + 4 * n_modes; layers = []
            for hsz in hidden:
                layers += [nn.Linear(d, hsz), nn.Tanh()]; d = hsz
            layers += [nn.Linear(d, n_bands)]
            self.net = nn.Sequential(*layers)

        def forward(self, k):
            inc = torch.nn.functional.softplus(self.net(self.ff(k)))
            return self.omega_min + torch.cumsum(inc, dim=-1) * self.dscale

    def _predict_hz(model, KX, KY):
        with torch.no_grad():
            k = torch.tensor(np.stack([KX / np.pi, KY / np.pi], 1), dtype=torch.float64)
            return model(k).cpu().numpy() / (2 * np.pi)

    def train_pinn(disp, KX, KY, n_bands=N_BANDS, n_modes=6, hidden=(96, 96, 96),
                   epochs_adam=4000, epochs_lbfgs=200, smooth_w=0.05, lr=2e-3,
                   loss_mode=LOSS_MODE, rar_rounds=0, rar_add=150,
                   knn_w=0.0, knn_k=4, jac_w=0.0, verbose=True):
        KX = np.asarray(KX, float).copy(); KY = np.asarray(KY, float).copy()
        model = BandPINN(n_bands, n_modes, hidden).double()
        history = []

        def tensors(KX, KY):
            return (torch.tensor(np.stack([KX/np.pi, KY/np.pi], 1), dtype=torch.float64),
                    torch.tensor(KX, dtype=torch.float64).unsqueeze(1),
                    torch.tensor(KY, dtype=torch.float64).unsqueeze(1))
        k_t, kx_t, ky_t = tensors(KX, KY)

        def build_knn(KX, KY, kk):
            P = np.stack([KX, KY], 1)
            d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            idx = np.argsort(d2, 1)[:, :kk]
            return torch.tensor(idx, dtype=torch.long)
        nbr = build_knn(KX, KY, knn_k) if knn_w > 0 else None

        target = 180.0 * 2 * np.pi

        def losses(need_jac=False):
            if jac_w > 0 or need_jac:
                k_in = k_t.clone().requires_grad_(True)
            else:
                k_in = k_t
            omega = model(k_in)
            r = disp.residual_torch(omega, kx_t, ky_t, mode=loss_mode)
            loss_res = (r ** 2).mean()
            gaps = omega[:, 1:] - omega[:, :-1]
            loss_sep = (target / (gaps + target)).pow(2).mean()
            total = loss_res + smooth_w * loss_sep
            if knn_w > 0:
                diff = omega.unsqueeze(1) - omega[nbr]
                loss_knn = (diff / OMEGA_MAX).pow(2).mean()
                total = total + knn_w * loss_knn
            if jac_w > 0:
                g = torch.autograd.grad(omega.sum(), k_in, create_graph=True)[0]
                loss_jac = (g / OMEGA_MAX).pow(2).mean()
                total = total + jac_w * loss_jac
            return total, loss_res

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rar_at = set(np.linspace(epochs_adam * 0.4, epochs_adam * 0.9,
                                 max(rar_rounds, 1)).astype(int)) if rar_rounds else set()
        for ep in range(epochs_adam):
            opt.zero_grad(); loss, lres = losses(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            history.append((ep, float(loss.detach()), float(lres.detach())))
            if verbose and ep % 500 == 0:
                print(f"  [adam] {ep:5d} loss={float(loss):.3e} res={float(lres):.3e}")
            if rar_rounds and ep in rar_at:
                pool = sample_irreducible_bz(2000, weight=None)
                pk, pkx, pky = tensors(pool[:, 0], pool[:, 1])
                with torch.no_grad():
                    rr = disp.residual_torch(model(pk), pkx, pky,
                                             mode=loss_mode).abs().max(1).values.numpy()
                add = pool[np.argsort(rr)[-rar_add:]]
                KX = np.concatenate([KX, add[:, 0]]); KY = np.concatenate([KY, add[:, 1]])
                k_t, kx_t, ky_t = tensors(KX, KY)
                if knn_w > 0:
                    nbr = build_knn(KX, KY, knn_k)
                if verbose:
                    print(f"  [RAR] +{rar_add} точек, всего {len(KX)}")

        if epochs_lbfgs > 0:
            opt2 = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=epochs_lbfgs,
                                     line_search_fn="strong_wolfe")
            it = {"n": epochs_adam}

            def closure():
                opt2.zero_grad(); l, lr_ = losses(); l.backward()
                history.append((it["n"], float(l.detach()), float(lr_.detach()))); it["n"] += 1
                return l
            try:
                opt2.step(closure)
            except Exception as e:
                print("  [lbfgs] пропущен:", e)
            if verbose:
                print(f"  [lbfgs] done loss={history[-1][1]:.3e} res={history[-1][2]:.3e}")

        return model, np.array(history), (KX, KY)

def evaluate_regions(disp, predict_fn, regions):
    res = {}
    for name, (KX, KY) in regions.items():
        roots = numerical_roots_per_k(disp, KX, KY)
        res[name] = test_metric(predict_fn(KX, KY), roots)
    return res

def plot_loss(history, fname="loss_history.png"):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ep = history[:, 0]; tot = history[:, 1]; rez = history[:, 2]
    ax.semilogy(ep, tot, label="total loss", lw=1.2)
    ax.semilogy(ep, rez, label="residual", lw=1.2, alpha=0.8)
    ax.set_xlabel("эпоха"); ax.set_ylabel("loss (log)")
    ax.set_title("Сходимость обучения PINN"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(); fig.tight_layout(); fig.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Сохранено: {fname}")

def plot_dispersion(S_test, num_x, num_f, pinn_f, S_path, gap, metric, fname):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    if gap is not None:
        ax.axhspan(gap[0], gap[1], color="violet", alpha=0.18,
                   label=f"Band gap {gap[0]:.0f}–{gap[1]:.0f} Гц")
    ax.scatter(num_x, num_f, s=14, color="0.35", zorder=3, label="Численно (эталон)")
    if pinn_f is not None:
        cmap = plt.cm.viridis(np.linspace(0, 0.85, pinn_f.shape[1]))
        for m in range(pinn_f.shape[1]):
            ax.scatter(S_path, pinn_f[:, m], s=12, marker="x", color=cmap[m],
                       zorder=4, label=f"PINN голова {m+1}")
    ax.set_xlim(0, 3); ax.set_ylim(0, F_MAX)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["Γ", "X", "M", "Γ"])
    ax.set_ylabel("Частота, Гц"); ax.set_xlabel("Волновой вектор")
    ttl = "PINN (обучен на 2D-точках) vs численный метод на пути (held-out)"
    if metric:
        ttl += f"\nTest RMSE = {metric['rmse']:.1f} Гц, покрытие = {metric['coverage']*100:.0f}%"
    ax.set_title(ttl, fontsize=10.5)
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Сохранено: {fname}")

def plot_region_rmse(region_metrics, fname="region_rmse.png"):
    names = list(region_metrics.keys())
    rmse = [region_metrics[n]["rmse"] if region_metrics[n] else np.nan for n in names]
    cov = [region_metrics[n]["coverage"] * 100 if region_metrics[n] else 0 for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, rmse, color="steelblue")
    for b, c in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{c:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Test RMSE, Гц"); ax.set_title("Ошибка PINN по регионам (над столбцом — покрытие)")
    ax.grid(True, axis="y", alpha=0.3); plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Сохранено: {fname}")

def plot_2d_band_comparison(disp, predict_fn, n_grid=22, fname="band2d.png"):
    import matplotlib.tri as mtri
    ax_kx = np.linspace(0.03, np.pi, n_grid)
    ax_ky = np.linspace(0.03, np.pi, n_grid)
    pkx, pky = [], []
    for kx in ax_kx:
        for ky in ax_ky:
            if ky <= kx:
                pkx.append(kx); pky.append(ky)
    pkx, pky = np.array(pkx), np.array(pky)
    roots = numerical_roots_per_k(disp, pkx, pky, n_omega=2500)
    num = np.array([rr[0] if len(rr) else np.nan for _, rr in roots])
    pf = predict_fn(pkx, pky)
    pinn = pf[:, 0]
    err = np.abs(pinn - num)
    ok = ~np.isnan(num)
    tri = mtri.Triangulation(pkx[ok], pky[ok])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, val, ttl in zip(axs, [num[ok], pinn[ok], err[ok]],
                            ["Численно (низшая ветвь)", "PINN (голова 1)",
                             "|PINN − численно|, Гц"]):
        c = ax.tricontourf(tri, val, levels=18, cmap="viridis")
        fig.colorbar(c, ax=ax, fraction=0.046)
        ax.plot([0, np.pi, np.pi, 0], [0, 0, np.pi, 0], "w-", lw=0.8)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("kx"); ax.set_ylabel("ky")
        ax.text(0.05, -0.18, "Γ"); ax.text(np.pi-0.2, -0.18, "X"); ax.text(np.pi-0.2, np.pi-0.1, "M")
    fig.suptitle("Акустическая ветвь по зоне Бриллюэна: PINN vs численный метод",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(fname, dpi=180, bbox_inches="tight")
    print(f"Сохранено: {fname}")

def plot_region_dispersion_grid(disp, predict_fn, region_names, n=45,
                                fname="region_dispersion.png"):
    fig, axs = plt.subplots(1, len(region_names), figsize=(5 * len(region_names), 4.5))
    if len(region_names) == 1:
        axs = [axs]
    for ax, name in zip(axs, region_names):
        p = sample_region(name, n)
        dist = np.hypot(p[:, 0], p[:, 1])
        roots = numerical_roots_per_k(disp, p[:, 0], p[:, 1], n_omega=2800)
        pf = predict_fn(p[:, 0], p[:, 1])
        for (d, (_, rr)) in zip(dist, roots):
            ax.scatter(np.full(len(rr), d), rr, s=20, color="0.35", zorder=2)
        cmap = plt.cm.viridis(np.linspace(0, 0.85, pf.shape[1]))
        for m in range(pf.shape[1]):
            ax.scatter(dist, pf[:, m], s=22, marker="x", color=cmap[m], zorder=3)
        ax.axhspan(275, 350, color="violet", alpha=0.15)
        ax.set_title(name, fontsize=11); ax.set_xlabel("|k| от Γ")
        ax.set_ylabel("Частота, Гц"); ax.set_ylim(0, F_MAX); ax.grid(True, alpha=0.3)
    fig.suptitle("По регионам: серое — численно (факт), крестики — PINN", fontsize=11)
    fig.tight_layout(); fig.savefig(fname, dpi=180, bbox_inches="tight")
    print(f"Сохранено: {fname}")

def plot_extrapolation_path(disp, predict_fn, kx_split=0.6 * np.pi,
                            fname="extrapolation_path.png"):
    KX, KY, S = brillouin_path(n_seg=70)
    roots = numerical_roots_per_k(disp, KX, KY, S)
    num_x = np.concatenate([np.full(len(rr), s) for s, rr in roots])
    num_f = np.concatenate([rr for _, rr in roots])
    pf = predict_fn(KX, KY)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    out = KX > kx_split
    in_seg = False
    for i in range(len(S)):
        if out[i] and not in_seg:
            x0 = S[i]; in_seg = True
        if (not out[i] or i == len(S) - 1) and in_seg:
            ax.axvspan(x0, S[i], color="orange", alpha=0.12); in_seg = False
    ax.scatter(num_x, num_f, s=12, color="0.35", zorder=3, label="Численно (факт)")
    cmap = plt.cm.viridis(np.linspace(0, 0.85, pf.shape[1]))
    for m in range(pf.shape[1]):
        ax.scatter(S, pf[:, m], s=14, marker="x", color=cmap[m], zorder=4,
                   label=f"PINN голова {m+1}")
    ax.set_xlim(0, 3); ax.set_ylim(0, F_MAX)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["Γ", "X", "M", "Γ"])
    ax.set_ylabel("Частота, Гц"); ax.set_xlabel("Волновой вектор")
    ax.set_title("Экстраполяция: оранжевая зона (kx>0.6π) НЕ была в обучении",
                 fontsize=10.5)
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(fname, dpi=180, bbox_inches="tight")
    print(f"Сохранено: {fname}")

def loss_comparison(disp, train_xy, test_path, fname="loss_comparison.png"):
    KXtr, KYtr = train_xy
    KXp, KYp, Sp = test_path
    roots = numerical_roots_per_k(disp, KXp, KYp, Sp)
    num_x = np.concatenate([np.full(len(rr), s) for s, rr in roots])
    num_f = np.concatenate([rr for _, rr in roots])

    results = {}
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for col, mode in enumerate(["raw", "normalized"]):
        model, hist, _ = train_pinn(disp, KXtr, KYtr, loss_mode=mode,
                                    epochs_adam=4000, epochs_lbfgs=200, verbose=False)
        pf = _predict_hz(model, KXp, KYp)
        m = test_metric(pf, roots)
        results[mode] = dict(m=m, hist=hist, pf=pf)

        ax = axs[0, col]
        ax.semilogy(hist[:, 0], hist[:, 1], lw=1.1, label="полный лосс")
        ax.semilogy(hist[:, 0], hist[:, 2], lw=1.1, alpha=0.8, label="невязка")
        ax.set_title(f"Лосс: {mode}", fontsize=11)
        ax.set_xlabel("эпоха"); ax.set_ylabel("лосс (лог)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

        ax = axs[1, col]
        ax.scatter(num_x, num_f, s=10, color="0.35", zorder=2, label="численно")
        cmap = plt.cm.viridis(np.linspace(0, 0.85, pf.shape[1]))
        for mm in range(pf.shape[1]):
            ax.scatter(Sp, pf[:, mm], s=10, marker="x", color=cmap[mm], zorder=3)
        ax.set_xlim(0, 3); ax.set_ylim(0, F_MAX)
        ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["Γ", "X", "M", "Γ"])
        ax.set_ylabel("Частота, Гц"); ax.set_xlabel("Волновой вектор")
        ttl = f"{mode}: RMSE={m['rmse']:.1f} Гц, покрытие={m['coverage']*100:.0f}%"
        ax.set_title(ttl, fontsize=10); ax.grid(True, alpha=0.3)

    fig.suptitle("Сравнение функций потерь: сырая (из гайда) vs нормированная", fontsize=12)
    fig.tight_layout(); fig.savefig(fname, dpi=180, bbox_inches="tight")
    print(f"Сохранено: {fname}")

    print("\n=== СРАВНЕНИЕ ЛОССОВ (raw vs normalized) ===")
    print(f"{'метрика':<18s} {'raw':>10s} {'normalized':>12s}")
    print("-" * 42)
    for key in ["rmse", "rmse_cov", "coverage"]:
        rv = results["raw"]["m"][key]
        nv = results["normalized"]["m"][key]
        if key == "coverage":
            print(f"{'покрытие':<18s} {rv*100:>9.0f}% {nv*100:>11.0f}%")
        else:
            print(f"{key:<18s} {rv:>10.1f} {nv:>12.1f}")
    print(f"{'финальный лосс':<18s} {results['raw']['hist'][-1,1]:>10.2e}"
          f" {results['normalized']['hist'][-1,1]:>12.2e}")

    return results

def smoothness_experiment(disp, train_xy, kx_split=0.6 * np.pi):
    KXtr, KYtr = train_xy
    KXp, KYp, Sp = brillouin_path(n_seg=70)
    roots_path = numerical_roots_per_k(disp, KXp, KYp, Sp)
    num_x = np.concatenate([np.full(len(rr), s) for s, rr in roots_path])
    num_f = np.concatenate([rr for _, rr in roots_path])
    out = KXp > kx_split

    configs = [
        dict(tag="baseline", knn_w=0.0, jac_w=0.0),
        dict(tag="+knn", knn_w=0.3, jac_w=0.0),
        dict(tag="+knn+jac", knn_w=0.3, jac_w=0.1),
    ]
    print("\n=== ЭКСПЕРИМЕНТ: регуляризаторы гладкости против хаоса ===")
    print("(train kx<=0.6π; смотрим экстраполяцию kx>0.6π)")
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, c in zip(axs, configs):
        tr = sample_irreducible_bz(len(KXtr), kx_hi=kx_split, weight="corners")
        model, _, _ = train_pinn(disp, tr[:, 0], tr[:, 1], knn_w=c["knn_w"],
                                 jac_w=c["jac_w"], epochs_adam=3000, epochs_lbfgs=120,
                                 verbose=False)
        pf = _predict_hz(model, KXp, KYp)
        ex_idx = np.where(out)[0]
        roots_ex = [roots_path[i] for i in ex_idx]
        m = test_metric(pf[ex_idx], roots_ex)
        rough = path_roughness(pf[ex_idx]) if len(ex_idx) > 2 else float("nan")
        print(f"  {c['tag']:10s}: экстра RMSE={m['rmse']:6.1f}  покрытие={m['coverage']*100:3.0f}%"
              f"  roughness={rough:.2e}")
        x0 = Sp[out].min() if out.any() else 0
        x1 = Sp[out].max() if out.any() else 3
        ax.axvspan(x0, x1, color="orange", alpha=0.12)
        ax.scatter(num_x, num_f, s=10, color="0.35", zorder=2)
        cmap = plt.cm.viridis(np.linspace(0, 0.85, pf.shape[1]))
        for mm in range(pf.shape[1]):
            ax.scatter(Sp, pf[:, mm], s=12, marker="x", color=cmap[mm], zorder=3)
        ax.set_title(f"{c['tag']}\nэкстра roughness={rough:.1e}", fontsize=10)
        ax.set_xlim(0, 3); ax.set_ylim(0, F_MAX)
        ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["Γ", "X", "M", "Γ"])
        ax.set_xlabel("Волновой вектор"); ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("Частота, Гц")
    fig.suptitle("Гладкостная регуляризация против хаоса в экстраполяции "
                 "(оранжевое — вне обучения)", fontsize=11)
    fig.tight_layout(); fig.savefig("smoothness_experiment.png", dpi=170, bbox_inches="tight")
    print("Сохранено: smoothness_experiment.png")

def hyperparameter_sweep(disp, train_xy, test_path):
    KXtr, KYtr = train_xy
    KXp, KYp, Sp = test_path
    roots_path = numerical_roots_per_k(disp, KXp, KYp, Sp)
    configs = [
        dict(n_modes=0, hidden=(96, 96, 96), smooth_w=0.05, tag="без Fourier"),
        dict(n_modes=6, hidden=(96, 96, 96), smooth_w=0.05, tag="Fourier-6"),
        dict(n_modes=10, hidden=(128, 128), smooth_w=0.05, tag="Fourier-10/шире"),
        dict(n_modes=6, hidden=(96, 96, 96), smooth_w=0.0, tag="без сглаживания"),
    ]
    print("\n=== SWEEP ===")
    for c in configs:
        model, hist, _ = train_pinn(disp, KXtr, KYtr, n_modes=c["n_modes"],
                                    hidden=c["hidden"], smooth_w=c["smooth_w"],
                                    epochs_adam=2500, epochs_lbfgs=100, verbose=False)
        pf = _predict_hz(model, KXp, KYp)
        m = test_metric(pf, roots_path)
        print(f"  {c['tag']:18s}: final_loss={hist[-1,1]:.2e}  "
              f"test_RMSE={m['rmse']:.1f} Гц  покрытие={m['coverage']*100:.0f}%")

def main():
    disp = Dispersion(**PARAMS)
    print(f"Резонанс: ν = {disp.nu:.1f} рад/с = {disp.nu/2/np.pi:.1f} Гц")
    print(f"loss_mode={LOSS_MODE}, train_strategy={TRAIN_STRATEGY}, N_train={N_TRAIN}")

    weight = "corners" if TRAIN_STRATEGY == "interior_weighted" else None
    train = sample_irreducible_bz(N_TRAIN, weight=weight)
    KXtr, KYtr = train[:, 0], train[:, 1]

    KXp, KYp, Sp = brillouin_path(n_seg=70)
    print("Численный эталон на ПУТИ (test)...")
    roots_path = numerical_roots_per_k(disp, KXp, KYp, Sp)
    num_x = np.concatenate([np.full(len(rr), s) for s, rr in roots_path])
    num_f = np.concatenate([rr for _, rr in roots_path])
    gap = find_complete_gap(num_f)
    if gap:
        print(f"Полный band gap: {gap[0]:.1f}–{gap[1]:.1f} Гц")

    if not HAVE_TORCH:
        print("torch не найден — только численный эталон. pip install torch")
        plot_dispersion(Sp, num_x, num_f, None, Sp, gap, None, "wave_spectra_v4.png")
        plt.show(); return

    print("Обучение PINN на 2D-сетке...")
    model, history, _ = train_pinn(disp, KXtr, KYtr,
                                   loss_mode=LOSS_MODE,
                                   rar_rounds=(2 if TRAIN_STRATEGY != "path" else 0))
    plot_loss(history)
    predict = lambda KX, KY: _predict_hz(model, KX, KY)

    pf_path = predict(KXp, KYp)
    m_path = test_metric(pf_path, roots_path)
    print(f"TEST (путь): RMSE_все={m_path['rmse']:.1f} Гц, "
          f"RMSE_покрытых={m_path['rmse_cov']:.1f} Гц, покрытие={m_path['coverage']*100:.0f}%")
    plot_dispersion(Sp, num_x, num_f, pf_path, Sp, gap, m_path, "wave_spectra_v4.png")

    regions = {n: (lambda p: (p[:, 0], p[:, 1]))(sample_region(n, 40))
               for n in ["near_Gamma", "near_X", "near_M"]}
    reg_metrics = evaluate_regions(disp, predict, regions)
    print("\nТесты по регионам:")
    for n, mt in reg_metrics.items():
        if mt:
            print(f"  {n:11s}: RMSE_все={mt['rmse']:6.1f}  RMSE_покр={mt['rmse_cov']:5.1f}  "
                  f"покрытие={mt['coverage']*100:3.0f}% ({mt['n']} корней)")
    plot_region_rmse(reg_metrics)

    print("\nГенерация доп. визуализаций по областям...")
    plot_2d_band_comparison(disp, predict, n_grid=22)
    plot_region_dispersion_grid(disp, predict,
                                ["near_Gamma", "near_X", "near_M"])

    if RUN_EXTRAPOLATION:
        print("\n=== ЭКСТРАПОЛЯЦИЯ (train kx<=0.6π, test kx>0.6π) ===")
        tr2 = sample_irreducible_bz(N_TRAIN, kx_hi=0.6 * np.pi, weight=weight)
        m2, h2, _ = train_pinn(disp, tr2[:, 0], tr2[:, 1],
                               loss_mode=LOSS_MODE, verbose=False)
        ex = sample_irreducible_bz(120, kx_lo=0.6 * np.pi)
        roots_ex = numerical_roots_per_k(disp, ex[:, 0], ex[:, 1])
        m_ex = test_metric(_predict_hz(m2, ex[:, 0], ex[:, 1]), roots_ex)
        inb = sample_irreducible_bz(120, kx_hi=0.6 * np.pi)
        roots_in = numerical_roots_per_k(disp, inb[:, 0], inb[:, 1])
        m_in = test_metric(_predict_hz(m2, inb[:, 0], inb[:, 1]), roots_in)
        print(f"  интерполяция (kx<=0.6π): RMSE={m_in['rmse']:.1f} Гц, "
              f"покрытие={m_in['coverage']*100:.0f}%")
        print(f"  экстраполяция (kx>0.6π): RMSE={m_ex['rmse']:.1f} Гц, "
              f"покрытие={m_ex['coverage']*100:.0f}%")
        print("  (рост RMSE и падение покрытия на экстраполяции — ожидаемо)")
        plot_extrapolation_path(disp, lambda KX, KY: _predict_hz(m2, KX, KY))

    if RUN_LOSS_CMP:
        loss_comparison(disp, (KXtr, KYtr), (KXp, KYp, Sp))

    if RUN_SWEEP:
        hyperparameter_sweep(disp, (KXtr, KYtr), (KXp, KYp, Sp))

    if RUN_SMOOTHNESS:
        smoothness_experiment(disp, (KXtr, KYtr))

    plt.show()

if __name__ == "__main__":
    if RUN_MAIN:
        main()