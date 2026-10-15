"""GAN condicional (pix2pix) para FC + anatomia → SC individual (plano v2 §3.2).

- Entrada (canais N×N): FC (arctanh), GD/100 mm, MPC/2, prior do grupo de treino (μ),
  máscara inter-hemisférica. Nada derivado da difusão do sujeito de teste entra.
- Saída: DESVIO individual Δ = y − μ (y = log1p(SC/média)); previsão final = μ + Δ, simetrizada.
  Prever y inteiro faria a rede copiar μ, que sozinha já dá r ≈ 0,94.
- Gerador: U-Net 2D pequena (poucos sujeitos → poucos parâmetros), dropout no gargalo.
- Discriminador: PatchGAN sobre o par (condição, Δ) — guia §4, Modelo B.
- Perda: L = λ_adv·L_adv + λ_L1·L1(Δ). Com λ_adv = 0 vira a ablação "U-Net só L1".
- Seleção de época por r do desvio em sujeitos de VALIDAÇÃO internos ao treino.
"""

import copy

import numpy as np
import torch
from torch import nn

from .sc_models import interhemispheric


def _sym(M):
    return (M + M.transpose(-1, -2)) / 2


def to_matrix(v, n):
    M = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    M[iu] = v
    return M + M.T


def build_tensors(mats, subs, mu_vec, n, base=None):
    """Empilha canais de entrada, alvo e deslocamento para os sujeitos `subs`.

    Sem `base`: alvo = y − μ (desvio do grupo); deslocamento = 0.
    Com `base` ({sujeito: previsão do B2}): entra como canal extra, alvo = y − base (resíduo)
    e deslocamento = base − μ, usado para medir o desvio final (base + resíduo) em relação a μ.
    """
    mu = to_matrix(mu_vec, n)
    labels = mats[subs[0]]["labels"]
    inter = to_matrix(interhemispheric(labels), n)
    X, Y, O = [], [], []
    for s in subs:
        d = mats[s]
        fc = np.arctanh(np.clip(d["fc"], -0.999, 0.999))
        np.fill_diagonal(fc, 0.0)
        ch = [fc, d["gd"] / 100.0, d["mpc"] / 2.0, mu, inter]
        if base is None:
            off = np.zeros((n, n))
            Y.append(to_matrix(d["y"], n) - mu)
        else:
            b = to_matrix(base[s], n)
            off = b - mu
            np.fill_diagonal(off, 0.0)
            ch.append(off)
            Y.append(to_matrix(d["y"], n) - b)
        X.append(np.stack(ch))
        O.append(off)

    def t(a):
        return torch.tensor(np.array(a), dtype=torch.float32)

    return t(X), t(Y)[:, None], t(O)[:, None]


def block(cin, cout, down=True, drop=0.0):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1) if down else nn.ConvTranspose2d(cin, cout, 4, 2, 1),
              nn.GroupNorm(min(8, cout), cout), nn.LeakyReLU(0.2) if down else nn.ReLU()]
    if drop:
        layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, cin=5, base=16, drop=0.3, zero_init=False):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.d1, self.d2, self.d3, self.d4 = block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3], drop=drop)
        self.u4 = block(c[3], c[2], down=False, drop=drop)
        self.u3 = block(c[2] * 2, c[1], down=False)
        self.u2 = block(c[1] * 2, c[0], down=False)
        self.u1 = nn.ConvTranspose2d(c[0] * 2, 1, 4, 2, 1)
        if zero_init:  # saída inicial = 0 → no modo residual, a época 0 reproduz exatamente o B2
            nn.init.zeros_(self.u1.weight)
            nn.init.zeros_(self.u1.bias)

    def forward(self, x):
        n = x.shape[-1]
        p = (-n) % 16
        x = nn.functional.pad(x, (0, p, 0, p))
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        y = self.u4(e4)
        y = self.u3(torch.cat([y, e3], 1))
        y = self.u2(torch.cat([y, e2], 1))
        y = self.u1(torch.cat([y, e1], 1))[..., :n, :n]
        y = _sym(y)
        eye = torch.eye(n, dtype=y.dtype, device=y.device)
        return y * (1 - eye)


class PatchD(nn.Module):
    def __init__(self, cin=6, base=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(base, base * 2, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(base * 2, base * 4, 4, 2, 1)), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 4, 1, 3, 1, 1))

    def forward(self, cond, dev):
        return self.net(torch.cat([cond, dev], 1))


def _r_dev(pred, true, n):
    iu = np.triu_indices(n, 1)
    out = []
    for p, t in zip(pred, true):
        a, b = p[0][iu], t[0][iu]
        out.append(np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 else 0.0)
    return float(np.mean(out))


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(Xtr, Ytr, Xva, Yva, lam_adv=1.0, lam_l1=100.0, epochs=300, batch=4, lr_g=2e-4, lr_d=2e-4,
          eval_every=10, seed=0, log=None, Ova=None, zero_init=False):
    """Treina G (e D se lam_adv > 0). Retorna o gerador da melhor época na validação.

    `Ova` (deslocamento base − μ) faz a seleção usar o desvio FINAL (base + resíduo) × verdade.
    Com `zero_init`, a época 0 (resíduo nulo = B2) também concorre na seleção.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    Xtr, Ytr, Xva = Xtr.to(dev), Ytr.to(dev), Xva.to(dev)
    n = Xtr.shape[-1]
    G = UNet(cin=Xtr.shape[1], zero_init=zero_init).to(dev)
    D = PatchD(cin=Xtr.shape[1] + 1).to(dev) if lam_adv > 0 else None
    og = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999)) if D else None
    bce = nn.BCEWithLogitsLoss()
    best, best_r, hist = None, -np.inf, []
    gen = torch.Generator().manual_seed(seed)
    off = np.zeros(tuple(Yva.shape)) if Ova is None else Ova.numpy()

    def evaluate(ep):
        nonlocal best, best_r
        G.eval()
        with torch.no_grad():
            r = _r_dev(G(Xva).cpu().numpy() + off, Yva.numpy() + off, n)
            ltr = float((G(Xtr) - Ytr).abs().mean())
        hist.append({"epoch": ep, "r_dev_val": r, "l1_train": ltr})
        if log:
            log(f"    ep {ep:4d}  L1_treino={ltr:.4f}  r_desvio_val={r:+.4f}")
        if r > best_r:
            best_r, best = r, copy.deepcopy(G.state_dict())

    if zero_init:
        evaluate(0)
    for ep in range(1, epochs + 1):
        G.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(dev)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            x, y = Xtr[idx], Ytr[idx]
            fake = G(x)
            if D is not None:
                od.zero_grad()
                pr, pf = D(x, y), D(x, fake.detach())
                ld = 0.5 * (bce(pr, torch.full_like(pr, 0.9)) + bce(pf, torch.zeros_like(pf)))
                ld.backward()
                od.step()
            og.zero_grad()
            l1 = (fake - y).abs().mean()
            lg = lam_l1 * l1
            if D is not None:
                pf = D(x, fake)
                lg = lg + lam_adv * bce(pf, torch.ones_like(pf))
            lg.backward()
            og.step()
        if ep % eval_every == 0 or ep == epochs:
            evaluate(ep)
    G.load_state_dict(best)
    G.eval()
    return G, {"best_r_dev_val": best_r, "history": hist}


def predict(G, X):
    with torch.no_grad():
        return G(X.to(next(G.parameters()).device)).cpu().numpy()[:, 0]