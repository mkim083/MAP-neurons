#!/usr/bin/env python3
"""
Train a 1-layer MNIST SNN with real-valued 3-state MAP hidden dynamics.

Hidden state per neuron: (phi_t, theta_t, v_t)
  theta_eff_t = theta_t + theta_v_couple * v_t
  s_t         = phi_t + dt * (omega_u_scale * u_t) - dt * phi_leak * phi_t - v_inhib * v_t
  z_t         = spike(s_t - theta_eff_t)
  phi_{t+1}   = clip(s_t - 2*phi_cap*z_t, -phi_cap, +phi_cap)
  v_{t+1}     = clip(v_decay*v_t + v_gain_z*z_t + v_gain_u*u_t, v_min, v_max)
  theta_{t+1} = clip(theta_decay*theta_t + (1-theta_decay)*theta_base + theta_gain_z*z_t + theta_gain_v*v_{t+1},
                     theta_min, theta_max)

Command example:
for dense FC:
python -u train_map_1layer_mnist_bptt.py \
  --device cpu --epochs 20 --batch 128 --num_workers 0 \
  --T 12 --H 256 \
  --fc_mode dense \
  --train_input_mode bernoulli --eval_input_mode bernoulli \
  --ckpt_out /(location of output file)/your_file.pt

 for sparse FC:
 python -u train_map_1layer_mnist_bptt.py \
  --device cpu --epochs 20 --batch 128 --num_workers 0 \
  --T 12 --H 256 \
  --fc_mode shared \
  --fc_mask_beta_start 4 --fc_mask_beta_end 12 --fc_mask_l1_lam 1e-5 \
  --train_input_mode bernoulli --eval_input_mode bernoulli \
  --ckpt_out /(location of output file)/your_file.pt




This is the continuous (non-FSM-table) training path.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train_phase_ring_snn_1layer_mnist_tfa_projected_phase_pwsec_connmask_intq import SharedMaskLinear


def _subset(ds, limit: int):
    if int(limit) <= 0 or int(limit) >= len(ds):
        return ds
    return Subset(ds, list(range(int(limit))))

#Input encoding
def _make_temporal_input(x_img: torch.Tensor, T: int, mode: str, rate_scale: float) -> torch.Tensor:
    x = x_img.view(x_img.size(0), -1).clamp(0.0, 1.0) * float(rate_scale)
    x = x.clamp(0.0, 1.0)
    if str(mode) == "bernoulli":
        p = x.unsqueeze(1).expand(-1, int(T), -1).contiguous()
        return torch.bernoulli(p)
    return x.unsqueeze(1).expand(-1, int(T), -1).contiguous()


@dataclass
class CfgMAPReal3State1L:
    din: int = 784
    h: int = 256
    dout: int = 10
    T: int = 12

    dt: float = 1.0
    phi_cap: float = 1.0
    phi_leak: float = 0.0
    omega_u_scale: float = 1.0
    v_inhib: float = 0.1
    theta_v_couple: float = 0.1

    theta_min: float = 0.6
    theta_max: float = 1.6
    theta_base: float = 1.0
    theta_decay: float = 0.95
    theta_gain_z: float = 0.20
    theta_gain_v: float = 0.05

    v_min: float = -1.0
    v_max: float = 1.0
    v_decay: float = 0.90
    v_gain_z: float = 0.25
    v_gain_u: float = 0.05

    phi_init_frac: float = -1.0
    theta_init_frac: float = 0.4
    v_init_frac: float = 0.0

    beta: float = 8.0
    spike_mode: str = "hard_ste"  # soft, hard_ste, hard
    readout_time_norm: str = "mean"  # sum, mean

    # FC parameterization
    fc_mode: str = "shared"  # shared or dense
    fc_shared_w_init: float = 0.03
    fc_shared_w_min: float = -2.0
    fc_shared_w_max: float = 2.0
    fc_mask_logit_init: float = 0.0
    fc_mask_hard_eval: int = 1


class MAPReal3State1L(nn.Module):
    def __init__(self, cfg: CfgMAPReal3State1L):
        super().__init__()
        self.cfg = cfg
        self.fc_mode = str(cfg.fc_mode).lower()

        if self.fc_mode == "dense":
            self.fc = nn.Linear(int(cfg.din), int(cfg.h), bias=True)
            with torch.no_grad():
                self.fc.weight.normal_(mean=0.0, std=float(max(1.0e-5, cfg.fc_shared_w_init)))
                if self.fc.bias is not None:
                    self.fc.bias.zero_()
        else:
            self.fc = SharedMaskLinear(
                int(cfg.din),
                int(cfg.h),
                bias=True,
                shared_w_init=float(cfg.fc_shared_w_init),
                shared_w_min=float(cfg.fc_shared_w_min),
                shared_w_max=float(cfg.fc_shared_w_max),
                mask_logit_init=float(cfg.fc_mask_logit_init),
                mask_hard_eval=int(cfg.fc_mask_hard_eval),
            )
        self.out = nn.Linear(int(cfg.h), int(cfg.dout), bias=True)

    @staticmethod
    def _ste_binary(z_soft: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z_hard = cond.to(z_soft.dtype)
        return z_soft + (z_hard - z_soft).detach()

    def set_mask_beta(self, beta: float) -> None:
        if self.fc_mode == "shared":
            self.fc.set_mask_beta(float(beta))

    def mask_stats(self) -> tuple[float, float]:
        if self.fc_mode != "shared":
            return 1.0, 1.0
        p, h = self.fc.mask_stats()
        return float(p.item()), float(h.item())

    def mask_prob_mean(self) -> torch.Tensor:
        if self.fc_mode != "shared":
            return torch.zeros((), device=self.out.weight.device, dtype=self.out.weight.dtype)
        p, _ = self.fc.mask_stats()
        return p

    def _init_states(self, bsz: int, dev: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = int(self.cfg.h)
        phi = torch.full((bsz, h), float(self.cfg.phi_init_frac) * float(self.cfg.phi_cap), device=dev, dtype=dtype)

        th0 = float(self.cfg.theta_min) + float(max(0.0, min(1.0, float(self.cfg.theta_init_frac)))) * (
            float(self.cfg.theta_max) - float(self.cfg.theta_min)
        )
        theta = torch.full((bsz, h), th0, device=dev, dtype=dtype)

        v0 = float(self.cfg.v_min) + float(max(0.0, min(1.0, float(self.cfg.v_init_frac)))) * (
            float(self.cfg.v_max) - float(self.cfg.v_min)
        )
        v = torch.full((bsz, h), v0, device=dev, dtype=dtype)
        return phi, theta, v

    def _spike(self, margin: torch.Tensor) -> torch.Tensor:
        beta = float(max(1.0e-3, float(self.cfg.beta)))
        z_soft = torch.sigmoid(beta * margin)
        cond = margin >= 0.0
        mode = str(self.cfg.spike_mode)
        if mode == "hard":
            return cond.to(margin.dtype)
        if mode == "hard_ste":
            return self._ste_binary(z_soft, cond)
        return z_soft

    def _layer_step(
        self,
        phi: torch.Tensor,
        theta: torch.Tensor,
        v: torch.Tensor,
        u_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        cap = float(cfg.phi_cap)
        period = 2.0 * cap

        theta_eff = theta + float(cfg.theta_v_couple) * v
        omega = float(cfg.omega_u_scale) * u_t
        s = (
            phi
            + float(cfg.dt) * omega
            - float(cfg.dt) * float(cfg.phi_leak) * phi
            - float(cfg.v_inhib) * v
        )
        margin = s - theta_eff
        z = self._spike(margin)

        phi_next = torch.clamp(s - period * z, min=-cap, max=cap)
        v_next = torch.clamp(
            float(cfg.v_decay) * v + float(cfg.v_gain_z) * z + float(cfg.v_gain_u) * u_t,
            min=float(cfg.v_min),
            max=float(cfg.v_max),
        )
        theta_next = torch.clamp(
            float(cfg.theta_decay) * theta
            + (1.0 - float(cfg.theta_decay)) * float(cfg.theta_base)
            + float(cfg.theta_gain_z) * z
            + float(cfg.theta_gain_v) * v_next,
            min=float(cfg.theta_min),
            max=float(cfg.theta_max),
        )
        return phi_next, theta_next, v_next, z, omega

    def forward_with_metrics(
        self, x_raster: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        if x_raster.dim() != 3 or int(x_raster.size(2)) != int(self.cfg.din):
            raise ValueError(f"expected [B,T,{int(self.cfg.din)}]")
        if int(x_raster.size(1)) != int(self.cfg.T):
            raise ValueError(f"expected T={int(self.cfg.T)}")

        bsz = int(x_raster.size(0))
        dev = x_raster.device
        dtype = torch.float32

        phi, theta, v = self._init_states(bsz, dev, dtype)
        z_sum = torch.zeros((bsz, int(self.cfg.h)), device=dev, dtype=dtype)
        z_acc = torch.zeros((), device=dev, dtype=dtype)

        phi_min = torch.full((), float("inf"), device=dev, dtype=dtype)
        phi_max = torch.full((), float("-inf"), device=dev, dtype=dtype)
        th_min = torch.full((), float("inf"), device=dev, dtype=dtype)
        th_max = torch.full((), float("-inf"), device=dev, dtype=dtype)
        v_min = torch.full((), float("inf"), device=dev, dtype=dtype)
        v_max = torch.full((), float("-inf"), device=dev, dtype=dtype)
        om_min = torch.full((), float("inf"), device=dev, dtype=dtype)
        om_max = torch.full((), float("-inf"), device=dev, dtype=dtype)

        for t in range(int(self.cfg.T)):
            x_t = x_raster[:, t, :]
            u_t = self.fc(x_t)
            phi, theta, v, z_t, omega = self._layer_step(phi, theta, v, u_t)

            z_sum = z_sum + z_t
            z_acc = z_acc + torch.mean(z_t)

            phi_min = torch.minimum(phi_min, torch.min(phi))
            phi_max = torch.maximum(phi_max, torch.max(phi))
            th_min = torch.minimum(th_min, torch.min(theta))
            th_max = torch.maximum(th_max, torch.max(theta))
            v_min = torch.minimum(v_min, torch.min(v))
            v_max = torch.maximum(v_max, torch.max(v))
            om_min = torch.minimum(om_min, torch.min(omega))
            om_max = torch.maximum(om_max, torch.max(omega))

        if str(self.cfg.readout_time_norm) == "mean":
            z_sum = z_sum / float(max(1, int(self.cfg.T)))
        logits = self.out(z_sum)
        z_rate = z_acc / float(max(1, int(self.cfg.T)))
        return (
            logits,
            z_rate,
            (float(phi_min.item()), float(phi_max.item())),
            (float(th_min.item()), float(th_max.item())),
            (float(v_min.item()), float(v_max.item())),
            (float(om_min.item()), float(om_max.item())),
        )

    def forward(self, x_raster: torch.Tensor):
        logits, z_rate, _, _, _, _ = self.forward_with_metrics(x_raster)
        return logits, z_rate


@torch.no_grad()
def _eval(model: MAPReal3State1L, loader: DataLoader, device: str, input_mode: str, rate_scale: float):
    model.eval()
    ok = 0
    total = 0
    z_acc = 0.0
    nb = 0
    ph_min, ph_max = float("inf"), float("-inf")
    th_min, th_max = float("inf"), float("-inf")
    vv_min, vv_max = float("inf"), float("-inf")
    om_min, om_max = float("inf"), float("-inf")

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        xr = _make_temporal_input(x, int(model.cfg.T), str(input_mode), float(rate_scale))
        lg, zr, ph, th, vv, om = model.forward_with_metrics(xr)
        ok += int((lg.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
        z_acc += float(zr.item())
        nb += 1
        ph_min = min(ph_min, float(ph[0]))
        ph_max = max(ph_max, float(ph[1]))
        th_min = min(th_min, float(th[0]))
        th_max = max(th_max, float(th[1]))
        vv_min = min(vv_min, float(vv[0]))
        vv_max = max(vv_max, float(vv[1]))
        om_min = min(om_min, float(om[0]))
        om_max = max(om_max, float(om[1]))

    acc = 100.0 * ok / max(1, total)
    return acc, (z_acc / max(1, nb)), (ph_min, ph_max), (th_min, th_max), (vv_min, vv_max), (om_min, om_max)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="/Users/moon/Documents/snn/mnist_pdr/MAP/paper/model/data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--test_limit", type=int, default=0)

    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--H", type=int, default=256)

    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--phi_cap", type=float, default=1.0)
    ap.add_argument("--phi_leak", type=float, default=0.0)
    ap.add_argument("--omega_u_scale", type=float, default=1.0)
    ap.add_argument("--v_inhib", type=float, default=0.1)
    ap.add_argument("--theta_v_couple", type=float, default=0.1)

    ap.add_argument("--theta_min", type=float, default=0.6)
    ap.add_argument("--theta_max", type=float, default=1.6)
    ap.add_argument("--theta_base", type=float, default=1.0)
    ap.add_argument("--theta_decay", type=float, default=0.95)
    ap.add_argument("--theta_gain_z", type=float, default=0.20)
    ap.add_argument("--theta_gain_v", type=float, default=0.05)

    ap.add_argument("--v_min", type=float, default=-1.0)
    ap.add_argument("--v_max", type=float, default=1.0)
    ap.add_argument("--v_decay", type=float, default=0.90)
    ap.add_argument("--v_gain_z", type=float, default=0.25)
    ap.add_argument("--v_gain_u", type=float, default=0.05)

    ap.add_argument("--phi_init_frac", type=float, default=-1.0)
    ap.add_argument("--theta_init_frac", type=float, default=0.4)
    ap.add_argument("--v_init_frac", type=float, default=0.0)

    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--spike_mode", choices=["soft", "hard_ste", "hard"], default="hard_ste")
    ap.add_argument("--readout_time_norm", choices=["sum", "mean"], default="mean")

    ap.add_argument("--fc_mode", choices=["shared", "dense"], default="shared")
    ap.add_argument("--fc_shared_w_init", type=float, default=0.03)
    ap.add_argument("--fc_shared_w_min", type=float, default=-2.0)
    ap.add_argument("--fc_shared_w_max", type=float, default=2.0)
    ap.add_argument("--fc_mask_logit_init", type=float, default=0.0)
    ap.add_argument("--fc_mask_hard_eval", type=int, choices=[0, 1], default=1)
    ap.add_argument("--fc_mask_beta_start", type=float, default=4.0)
    ap.add_argument("--fc_mask_beta_end", type=float, default=12.0)
    ap.add_argument("--fc_mask_l1_lam", type=float, default=0.0)

    ap.add_argument("--train_input_mode", choices=["repeat", "bernoulli"], default="bernoulli")
    ap.add_argument("--eval_input_mode", choices=["repeat", "bernoulli"], default="bernoulli")
    ap.add_argument("--rate_scale", type=float, default=1.0)

    ap.add_argument(
        "--ckpt_out",
        default="/Users/moon/Documents/snn/mnist_pdr/MAP/paper/model/out_ckpt/mnist_map_real_3state_1l_bptt.pt",
    )
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    dev = str(args.device)

    cfg = CfgMAPReal3State1L(
        din=784,
        h=int(args.H),
        dout=10,
        T=int(args.T),
        dt=float(args.dt),
        phi_cap=float(args.phi_cap),
        phi_leak=float(args.phi_leak),
        omega_u_scale=float(args.omega_u_scale),
        v_inhib=float(args.v_inhib),
        theta_v_couple=float(args.theta_v_couple),
        theta_min=float(args.theta_min),
        theta_max=float(args.theta_max),
        theta_base=float(args.theta_base),
        theta_decay=float(args.theta_decay),
        theta_gain_z=float(args.theta_gain_z),
        theta_gain_v=float(args.theta_gain_v),
        v_min=float(args.v_min),
        v_max=float(args.v_max),
        v_decay=float(args.v_decay),
        v_gain_z=float(args.v_gain_z),
        v_gain_u=float(args.v_gain_u),
        phi_init_frac=float(args.phi_init_frac),
        theta_init_frac=float(args.theta_init_frac),
        v_init_frac=float(args.v_init_frac),
        beta=float(args.beta),
        spike_mode=str(args.spike_mode),
        readout_time_norm=str(args.readout_time_norm),
        fc_mode=str(args.fc_mode),
        fc_shared_w_init=float(args.fc_shared_w_init),
        fc_shared_w_min=float(args.fc_shared_w_min),
        fc_shared_w_max=float(args.fc_shared_w_max),
        fc_mask_logit_init=float(args.fc_mask_logit_init),
        fc_mask_hard_eval=int(args.fc_mask_hard_eval),
    )

    model = MAPReal3State1L(cfg).to(dev)

    tr = transforms.ToTensor()
    ds_tr = datasets.MNIST(root=str(args.data_dir), train=True, download=True, transform=tr)
    ds_te = datasets.MNIST(root=str(args.data_dir), train=False, download=True, transform=tr)
    ds_tr = _subset(ds_tr, int(args.train_limit))
    ds_te = _subset(ds_te, int(args.test_limit))

    ld_tr = DataLoader(ds_tr, batch_size=int(args.batch), shuffle=True, num_workers=int(args.num_workers), pin_memory=False)
    ld_te = DataLoader(ds_te, batch_size=max(256, int(args.batch)), shuffle=False, num_workers=int(args.num_workers), pin_memory=False)

    if str(args.optimizer) == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(args.lr),
            momentum=float(args.momentum),
            nesterov=True,
            weight_decay=float(args.weight_decay),
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_acc = -1.0
    best_ep = -1
    best_sd = None

    for ep in range(1, int(args.epochs) + 1):
        if model.fc_mode == "shared":
            if int(args.epochs) <= 1:
                m_beta = float(args.fc_mask_beta_end)
            else:
                r = float(ep - 1) / float(int(args.epochs) - 1)
                m_beta = float(args.fc_mask_beta_start) + r * (float(args.fc_mask_beta_end) - float(args.fc_mask_beta_start))
            model.set_mask_beta(float(m_beta))
        else:
            m_beta = 0.0

        model.train()
        ls = 0.0
        n = 0
        z_tr = 0.0
        nb = 0

        for x, y in ld_tr:
            x = x.to(dev)
            y = y.to(dev)
            xr = _make_temporal_input(x, int(cfg.T), str(args.train_input_mode), float(args.rate_scale))
            lg, zr, _, _, _, _ = model.forward_with_metrics(xr)
            lsmooth = float(max(0.0, min(0.2, float(args.label_smoothing))))
            ce = F.cross_entropy(lg, y, label_smoothing=lsmooth)
            loss = ce + float(args.fc_mask_l1_lam) * model.mask_prob_mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(y.numel())
            ls += float(loss.item()) * bs
            n += bs
            z_tr += float(zr.item())
            nb += 1

        tr_loss = ls / float(max(1, n))
        z_tr = z_tr / float(max(1, nb))
        acc, z_ev, ph, th, vv, om = _eval(model, ld_te, dev, str(args.eval_input_mode), float(args.rate_scale))
        mp, mh = model.mask_stats()

        if model.fc_mode == "dense":
            w = model.fc.weight.detach()
            sw_min = float(w.min().item())
            sw_max = float(w.max().item())
        else:
            sw = torch.clamp(model.fc.shared_w.detach(), min=float(cfg.fc_shared_w_min), max=float(cfg.fc_shared_w_max))
            sw_min = float(sw.min().item())
            sw_max = float(sw.max().item())

        if acc > best_acc:
            best_acc = float(acc)
            best_ep = int(ep)
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[ep {ep:02d}] loss={tr_loss:.4f} test={acc:.2f}% "
            f"T={int(cfg.T)} H={int(cfg.h)} fc={str(cfg.fc_mode)} "
            f"mask(beta={m_beta:.2f},p={mp:.3f},h={mh:.3f},l1={float(args.fc_mask_l1_lam):.1e}) sw=({sw_min:.3f},{sw_max:.3f}) "
            f"z_tr={z_tr:.3f} z_ev={z_ev:.3f} "
            f"phi=({ph[0]:.3f},{ph[1]:.3f}) th=({th[0]:.3f},{th[1]:.3f}) v=({vv[0]:.3f},{vv[1]:.3f}) om=({om[0]:.3f},{om[1]:.3f}) "
            f"in=({str(args.train_input_mode)},{str(args.eval_input_mode)}) spike={str(cfg.spike_mode)}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_out)), exist_ok=True)
    ckpt = {
        "arch": "mnist_map_real_3state_1layer_bptt",
        "dataset": "mnist",
        "state_dict": best_sd if best_sd is not None else model.state_dict(),
        "config": {
            "Din": int(cfg.din),
            "H": int(cfg.h),
            "Dout": int(cfg.dout),
            "T": int(cfg.T),
            "dt": float(cfg.dt),
            "phi_cap": float(cfg.phi_cap),
            "phi_leak": float(cfg.phi_leak),
            "omega_u_scale": float(cfg.omega_u_scale),
            "v_inhib": float(cfg.v_inhib),
            "theta_v_couple": float(cfg.theta_v_couple),
            "theta_min": float(cfg.theta_min),
            "theta_max": float(cfg.theta_max),
            "theta_base": float(cfg.theta_base),
            "theta_decay": float(cfg.theta_decay),
            "theta_gain_z": float(cfg.theta_gain_z),
            "theta_gain_v": float(cfg.theta_gain_v),
            "v_min": float(cfg.v_min),
            "v_max": float(cfg.v_max),
            "v_decay": float(cfg.v_decay),
            "v_gain_z": float(cfg.v_gain_z),
            "v_gain_u": float(cfg.v_gain_u),
            "phi_init_frac": float(cfg.phi_init_frac),
            "theta_init_frac": float(cfg.theta_init_frac),
            "v_init_frac": float(cfg.v_init_frac),
            "beta": float(cfg.beta),
            "spike_mode": str(cfg.spike_mode),
            "readout_time_norm": str(cfg.readout_time_norm),
            "fc_mode": str(cfg.fc_mode),
            "fc_shared_w_init": float(cfg.fc_shared_w_init),
            "fc_shared_w_min": float(cfg.fc_shared_w_min),
            "fc_shared_w_max": float(cfg.fc_shared_w_max),
            "fc_mask_logit_init": float(cfg.fc_mask_logit_init),
            "fc_mask_hard_eval": int(cfg.fc_mask_hard_eval),
            "fc_mask_beta_start": float(args.fc_mask_beta_start),
            "fc_mask_beta_end": float(args.fc_mask_beta_end),
            "fc_mask_l1_lam": float(args.fc_mask_l1_lam),
            "train_input_mode": str(args.train_input_mode),
            "eval_input_mode": str(args.eval_input_mode),
            "rate_scale": float(args.rate_scale),
            "best_test_acc": float(best_acc),
            "best_epoch": int(best_ep),
        },
        "best_test_acc": float(best_acc),
        "best_epoch": int(best_ep),
    }
    torch.save(ckpt, str(args.ckpt_out))
    print(f"[ok] wrote ckpt={args.ckpt_out} best={best_acc:.2f}% best_epoch={best_ep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
