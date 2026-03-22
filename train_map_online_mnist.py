#!/usr/bin/env python3
"""
Online Local Target Learning (LTL) on MNIST.

Motivation (fixes common online failures):
  1) noisy instantaneous errors -> low-pass error trace
  2) updates when neuron is insensitive -> post-sensitivity gate psi(z)
  3) trace feedback destabilizes spikes -> trace only in learning, not forward
  4) firing collapse -> mild adaptive threshold (optional)

Neuron dynamics:
  u[t+1] = alpha*u[t] + W x[t] + b - beta*s[t]
  s[t] = H( u[t] - (theta0 + theta_a*a[t]) )
  a[t+1] = rho_a*a[t] + gamma_a*(s[t] - r_target)   (optional homeostasis)

Pre-trace and sensitivity:
  p[t+1] = lambda_p * p[t] + x[t]
  psi[t] = d sigma(z/tau)  or triangular surrogate of z

Output:
  o[t+1] = rho_out * o[t] + V s[t] + b_out
  delta_out[t] = softmax(o[t]) - y*
  delta_bar[t] = lambda_d * delta_bar[t-1] + delta_out[t]

Local target update:
  c = (-delta_bar) @ Bfb^T
  W += lr * ( (c*psi)^T @ p ) / B
  b += lr * mean(c*psi)

Output update:
  W2 -= lr_out * (delta_bar^T @ s) / B
  b2 -= lr_out * mean(delta_bar)
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _subset(ds, limit: int):
    if int(limit) <= 0 or int(limit) >= len(ds):
        return ds
    return Subset(ds, list(range(int(limit))))


def _make_temporal_input(x_img: torch.Tensor, T: int, mode: str, rate_scale: float) -> torch.Tensor:
    x = x_img.view(x_img.size(0), -1).clamp(0.0, 1.0) * float(rate_scale)
    x = x.clamp(0.0, 1.0)
    if str(mode) == "bernoulli":
        p = x.unsqueeze(1).expand(-1, int(T), -1).contiguous()
        return torch.bernoulli(p)
    return x.unsqueeze(1).expand(-1, int(T), -1).contiguous()


def _psi(z: torch.Tensor, mode: str, tau: float) -> torch.Tensor:
    if mode == "tri":
        return (1.0 - (z / max(1.0e-6, tau)).abs()).clamp(min=0.0, max=1.0)
    if mode == "sigmoid":
        s = torch.sigmoid(z / max(1.0e-6, tau))
        return s * (1.0 - s)
    # hard sigmoid derivative proxy
    return (0.5 + 0.5 * z / max(1.0e-6, tau)).clamp(0.0, 1.0) * (1.0 / max(1.0e-6, tau))


@dataclass
class Cfg:
    din: int = 784
    h: int = 256
    dout: int = 10
    T: int = 10
    alpha: float = 0.95
    beta: float = 1.0
    theta0: float = 1.0
    theta_a: float = 0.1
    rho_a: float = 0.98
    gamma_a: float = 0.02
    r_target: float = 0.05
    lambda_p: float = 0.9
    psi_mode: str = "tri"
    psi_tau: float = 1.0
    osc_enable: int = 0
    osc_mode: str = "learn_only"  # learn_only | forward_bias | off
    osc_amp: float = 0.2
    osc_omega: float = 0.3
    osc_kappa_u: float = 0.0
    rho_out: float = 0.0
    lambda_d: float = 0.9
    lr: float = 1e-3
    lr_out: float = 1e-3
    fb_scale: float = 1.0
    input_mode: str = "bernoulli"
    rate_scale: float = 1.0
    readout_time_norm: str = "mean"


class SGTN_LTL_1L(nn.Module):
    def __init__(self, cfg: Cfg, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.W1 = nn.Parameter(torch.empty(cfg.h, cfg.din, device=device))
        self.b1 = nn.Parameter(torch.zeros(cfg.h, device=device))
        self.W2 = nn.Parameter(torch.empty(cfg.dout, cfg.h, device=device))
        self.b2 = nn.Parameter(torch.zeros(cfg.dout, device=device))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.register_buffer("Bfb", torch.randn(cfg.h, cfg.dout, device=device) * float(cfg.fb_scale))

    def forward_online(self, x_seq: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        B = x_seq.size(0)
        u = torch.zeros(B, cfg.h, device=self.device)
        s = torch.zeros(B, cfg.h, device=self.device)
        a = torch.zeros(B, cfg.h, device=self.device)
        p = torch.zeros(B, cfg.din, device=self.device)
        o = torch.zeros(B, cfg.dout, device=self.device)
        d_bar = torch.zeros(B, cfg.dout, device=self.device)
        logits_sum = torch.zeros(B, cfg.dout, device=self.device)
        phi = torch.zeros(B, cfg.h, device=self.device)

        for t in range(cfg.T):
            x_t = x_seq[:, t, :]
            p = cfg.lambda_p * p + x_t
            u = cfg.alpha * u + (x_t @ self.W1.T) + self.b1 - cfg.beta * s
            if int(cfg.osc_enable) == 1 and str(cfg.osc_mode) in ("learn_only", "forward_bias"):
                phi = phi + cfg.osc_omega + cfg.osc_kappa_u * u
                phi = torch.remainder(phi, 2.0 * math.pi)
                osc = cfg.osc_amp * torch.sin(phi)
            else:
                osc = 0.0
            z = u - (cfg.theta0 + cfg.theta_a * a)
            if str(cfg.osc_mode) == "forward_bias":
                z = z + osc
            s = (z >= 0.0).float()
            psi = _psi(z, cfg.psi_mode, cfg.psi_tau)

            o = cfg.rho_out * o + (s @ self.W2.T) + self.b2
            logits_sum = logits_sum + o
            prob = torch.softmax(o, dim=1)
            y_onehot = torch.zeros_like(prob).scatter_(1, y.view(-1, 1), 1.0)
            delta = (prob - y_onehot)
            d_bar = cfg.lambda_d * d_bar + delta

            c = (-d_bar) @ self.Bfb.T
            with torch.no_grad():
                self.W2 -= cfg.lr_out * (d_bar.T @ s) / B
                self.b2 -= cfg.lr_out * d_bar.mean(dim=0)
                if int(cfg.osc_enable) == 1 and str(cfg.osc_mode) == "learn_only":
                    g_osc = 1.0 + osc
                    cg = c * psi * g_osc
                else:
                    cg = c * psi
                self.W1 += cfg.lr * (cg.T @ p) / B
                self.b1 += cfg.lr * cg.mean(dim=0)

            # homeostatic threshold (optional)
            a = cfg.rho_a * a + cfg.gamma_a * (s - cfg.r_target)

        logits_avg = logits_sum / float(cfg.T) if cfg.readout_time_norm == "mean" else logits_sum
        loss = nn.CrossEntropyLoss()(logits_avg, y)
        return logits_avg, loss

    @torch.no_grad()
    def forward_eval(self, x_seq: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = x_seq.size(0)
        u = torch.zeros(B, cfg.h, device=self.device)
        s = torch.zeros(B, cfg.h, device=self.device)
        a = torch.zeros(B, cfg.h, device=self.device)
        p = torch.zeros(B, cfg.din, device=self.device)
        o = torch.zeros(B, cfg.dout, device=self.device)
        logits_sum = torch.zeros(B, cfg.dout, device=self.device)
        phi = torch.zeros(B, cfg.h, device=self.device)
        for t in range(cfg.T):
            x_t = x_seq[:, t, :]
            p = cfg.lambda_p * p + x_t
            u = cfg.alpha * u + (x_t @ self.W1.T) + self.b1 - cfg.beta * s
            if int(cfg.osc_enable) == 1 and str(cfg.osc_mode) in ("learn_only", "forward_bias"):
                phi = phi + cfg.osc_omega + cfg.osc_kappa_u * u
                phi = torch.remainder(phi, 2.0 * math.pi)
                osc = cfg.osc_amp * torch.sin(phi)
            else:
                osc = 0.0
            z = u - (cfg.theta0 + cfg.theta_a * a)
            if str(cfg.osc_mode) == "forward_bias":
                z = z + osc
            s = (z >= 0.0).float()
            o = cfg.rho_out * o + (s @ self.W2.T) + self.b2
            logits_sum = logits_sum + o
            a = cfg.rho_a * a + cfg.gamma_a * (s - cfg.r_target)
        return logits_sum / float(cfg.T) if cfg.readout_time_norm == "mean" else logits_sum


def _eval(model: SGTN_LTL_1L, loader: DataLoader, cfg: Cfg, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for x_img, y in loader:
        x_img = x_img.to(device)
        y = y.to(device)
        x_seq = _make_temporal_input(x_img, cfg.T, cfg.input_mode, cfg.rate_scale)
        logits = model.forward_eval(x_seq)
        loss = nn.CrossEntropyLoss()(logits, y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item()
    acc = correct / max(1, total)
    return acc, loss_sum / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser(description="Online MNIST with SGTN + LTL")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--test_limit", type=int, default=0)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--theta0", type=float, default=1.0)
    ap.add_argument("--theta_a", type=float, default=0.1)
    ap.add_argument("--rho_a", type=float, default=0.98)
    ap.add_argument("--gamma_a", type=float, default=0.02)
    ap.add_argument("--r_target", type=float, default=0.05)
    ap.add_argument("--lambda_p", type=float, default=0.9)
    ap.add_argument("--psi_mode", choices=["tri", "sigmoid", "hard_sigmoid"], default="tri")
    ap.add_argument("--psi_tau", type=float, default=1.0)
    ap.add_argument("--osc_enable", type=int, default=0, choices=[0, 1])
    ap.add_argument("--osc_mode", choices=["learn_only", "forward_bias", "off"], default="learn_only")
    ap.add_argument("--osc_amp", type=float, default=0.2)
    ap.add_argument("--osc_omega", type=float, default=0.3)
    ap.add_argument("--osc_kappa_u", type=float, default=0.0)
    ap.add_argument("--rho_out", type=float, default=0.0)
    ap.add_argument("--lambda_d", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_out", type=float, default=1e-3)
    ap.add_argument("--fb_scale", type=float, default=1.0)
    ap.add_argument("--input_mode", choices=["bernoulli", "repeat"], default="bernoulli")
    ap.add_argument("--rate_scale", type=float, default=1.0)
    ap.add_argument("--readout_time_norm", choices=["sum", "mean"], default="mean")
    ap.add_argument("--ckpt_out", required=True)
    args = ap.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device)

    cfg = Cfg(
        h=int(args.H),
        T=int(args.T),
        alpha=float(args.alpha),
        beta=float(args.beta),
        theta0=float(args.theta0),
        theta_a=float(args.theta_a),
        rho_a=float(args.rho_a),
        gamma_a=float(args.gamma_a),
        r_target=float(args.r_target),
        lambda_p=float(args.lambda_p),
        psi_mode=str(args.psi_mode),
        psi_tau=float(args.psi_tau),
        osc_enable=int(args.osc_enable),
        osc_mode=str(args.osc_mode),
        osc_amp=float(args.osc_amp),
        osc_omega=float(args.osc_omega),
        osc_kappa_u=float(args.osc_kappa_u),
        rho_out=float(args.rho_out),
        lambda_d=float(args.lambda_d),
        lr=float(args.lr),
        lr_out=float(args.lr_out),
        fb_scale=float(args.fb_scale),
        input_mode=str(args.input_mode),
        rate_scale=float(args.rate_scale),
        readout_time_norm=str(args.readout_time_norm),
    )

    tr = transforms.Compose([transforms.ToTensor()])
    ds_tr = datasets.MNIST(root=os.path.expanduser("~/data"), train=True, download=True, transform=tr)
    ds_te = datasets.MNIST(root=os.path.expanduser("~/data"), train=False, download=True, transform=tr)
    ds_tr = _subset(ds_tr, int(args.train_limit))
    ds_te = _subset(ds_te, int(args.test_limit))

    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch), shuffle=True, num_workers=int(args.num_workers))
    dl_te = DataLoader(ds_te, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers))

    model = SGTN_LTL_1L(cfg, device).to(device)

    best_acc = 0.0
    for ep in range(int(args.epochs)):
        model.train()
        loss_sum = 0.0
        for x_img, y in dl_tr:
            x_img = x_img.to(device)
            y = y.to(device)
            x_seq = _make_temporal_input(x_img, cfg.T, cfg.input_mode, cfg.rate_scale)
            _, loss = model.forward_online(x_seq, y)
            loss_sum += loss.item()
        acc_te, loss_te = _eval(model, dl_te, cfg, device)
        print(f"[ep {ep+1:02d}] train_loss={loss_sum/max(1,len(dl_tr)):.4f} test_acc={acc_te*100:.2f}% test_loss={loss_te:.4f}")
        if acc_te > best_acc:
            best_acc = acc_te
            os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_out)), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, args.ckpt_out)
            print(f"[ok] wrote ckpt={args.ckpt_out} best={best_acc*100:.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())
