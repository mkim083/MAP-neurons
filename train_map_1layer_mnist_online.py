#!/usr/bin/env python3
"""
Online local learning (e-prop style) for 1-layer MNIST real 3-state MAP model.

Model dynamics are imported from:
  train_map_real_snn_1layer_mnist_bptt.py

MAP eligibility options:
- post replacement with phi signal:    --eprop_post_src phi
- optional phi gating on credit:       --phi_gate_enable 1
- phi signal shape control:            --phi_signal_mode {linear,legacy}


python -u /train_map_1layer_mnist_online.py \
  --device cpu --epochs 20 --batch 128 --num_workers 0 \
  --T 12 --H 256 \
  --fc_mode shared \
  --train_input_mode bernoulli --eval_input_mode bernoulli \
  --ckpt_out /your_file.pt


For phi-post signal:
python -u /train_map_1layer_mnist_online.py \
  --device cpu --epochs 20 --batch 128 --num_workers 0 \
  --T 12 --H 256 \
  --fc_mode shared \
  --eprop_post_src phi \
  --phi_gate_enable 0 --phi_signal_mode linear --phi_gate_power 1.0 \
  --train_input_mode bernoulli --eval_input_mode bernoulli \
  --ckpt_out /your_file.pt


"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_map_real_snn_1layer_mnist_bptt import (
    CfgMAPReal3State1L,
    MAPReal3State1L,
    _eval,
    _make_temporal_input,
    _subset,
)


def _norm_update(x: Optional[torch.Tensor], mode: str, eps: float) -> Optional[torch.Tensor]:
    if x is None:
        return None
    m = str(mode).lower()
    if m == "none":
        return x
    if m == "rms":
        d = torch.sqrt(torch.mean(x * x) + float(max(1.0e-12, eps)))
        return x / d
    if m == "meanabs":
        d = torch.mean(torch.abs(x)) + float(max(1.0e-12, eps))
        return x / d
    raise ValueError(f"unsupported grad norm mode: {mode}")


def _assign_rule_grad(param: Optional[torch.Tensor], grad: Optional[torch.Tensor]) -> None:
    if param is None or grad is None:
        return
    if tuple(grad.shape) != tuple(param.shape):
        if int(grad.numel()) == 1:
            grad = grad.expand_as(param)
        elif int(grad.numel()) == int(param.numel()):
            grad = grad.reshape_as(param)
        else:
            grad = torch.zeros_like(param)
    if param.grad is None:
        param.grad = grad.detach().clone()
    else:
        param.grad.copy_(grad.detach())


def _phi_signal(phi: torch.Tensor, cap: float, mode: str, power: float = 1.0) -> torch.Tensor:
    cap_c = float(max(1.0e-6, abs(cap)))
    if str(mode).lower() == "legacy":
        # Prior MAP script style: bowl-shaped around center.
        phi_norm = torch.clamp(phi / cap_c, min=-1.0, max=1.0)
        s = torch.clamp(0.5 + 0.5 * (1.0 - phi_norm * phi_norm), min=0.0)
    else:
        # Linear distance-to-cap gate.
        s = torch.clamp(1.0 - torch.abs(phi) / cap_c, min=0.0)
    p = float(max(0.0, power))
    if p > 0.0 and abs(p - 1.0) > 1.0e-8:
        s = torch.pow(s, p)
    return s


@torch.no_grad()
def _run_online_epoch(
    *,
    model: MAPReal3State1L,
    loader: DataLoader,
    device: str,
    input_mode: str,
    rate_scale: float,
    lr_out: float,
    lr_fc: float,
    lr_sw: float,
    lr_mask: float,
    lr_bias: float,
    trace_decay_pre: float,
    trace_decay_post: float,
    global_mod_scale: float,
    hidden_update_scale: float,
    eprop_post_src: str,
    post_trace_mix_alpha: float,
    phi_signal_mode: str,
    phi_gate_enable: int,
    phi_gate_power: float,
    homeo_enable: int,
    homeo_eta: float,
    homeo_target: float,
    mask_logit_clip: float,
    mask_logit_bound: float,
    optimizer: torch.optim.Optimizer | None,
    rule_use_optimizer: int,
    grad_norm_mode: str,
    grad_norm_eps: float,
    rule_grad_clip: float,
) -> tuple[float, float]:
    model.train()
    cfg = model.cfg
    t_steps = int(cfg.T)
    din = int(cfg.din)
    h = int(cfg.h)
    dout = int(cfg.dout)

    post_src = str(eprop_post_src).lower()
    if post_src not in {"sens", "phi", "mix"}:
        raise ValueError("eprop_post_src must be one of: sens, phi, mix")
    mix_a = float(max(0.0, min(1.0, float(post_trace_mix_alpha))))
    phi_sig_mode_l = str(phi_signal_mode).lower()
    if phi_sig_mode_l not in {"linear", "legacy"}:
        raise ValueError("phi_signal_mode must be one of: linear, legacy")

    ce_sum = 0.0
    z_sum = 0.0
    n_terms = 0
    n_steps = 0

    rate_ema = torch.zeros((h,), device=device, dtype=torch.float32)
    rate_beta = 0.05

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        bs = int(y.numel())
        xr = _make_temporal_input(x, t_steps, str(input_mode), float(rate_scale))

        phi, theta, vv = model._init_states(bs, torch.device(device), torch.float32)
        z_mem = torch.zeros((bs, h), device=device, dtype=torch.float32)

        pre_trace = torch.zeros((bs, din), device=device, dtype=torch.float32)
        post_trace = torch.zeros((bs, h), device=device, dtype=torch.float32)

        for t in range(t_steps):
            x_t = xr[:, t, :]
            u_t = model.fc(x_t)

            theta_eff = theta + float(cfg.theta_v_couple) * vv
            omega = float(cfg.omega_u_scale) * u_t
            s = (
                phi
                + float(cfg.dt) * omega
                - float(cfg.dt) * float(cfg.phi_leak) * phi
                - float(cfg.v_inhib) * vv
            )
            margin = s - theta_eff

            beta = float(max(1.0e-3, float(cfg.beta)))
            z_soft = torch.sigmoid(beta * margin)
            z_hard = (margin >= 0.0).to(torch.float32)
            mode = str(cfg.spike_mode)
            if mode == "hard":
                z_t = z_hard
            elif mode == "hard_ste":
                z_t = z_soft + (z_hard - z_soft).detach()
            else:
                z_t = z_soft

            cap = float(cfg.phi_cap)
            period = 2.0 * cap
            phi_next = torch.clamp(s - period * z_t, min=-cap, max=cap)
            v_next = torch.clamp(
                float(cfg.v_decay) * vv + float(cfg.v_gain_z) * z_t + float(cfg.v_gain_u) * u_t,
                min=float(cfg.v_min),
                max=float(cfg.v_max),
            )
            theta_next = torch.clamp(
                float(cfg.theta_decay) * theta
                + (1.0 - float(cfg.theta_decay)) * float(cfg.theta_base)
                + float(cfg.theta_gain_z) * z_t
                + float(cfg.theta_gain_v) * v_next,
                min=float(cfg.theta_min),
                max=float(cfg.theta_max),
            )

            z_mem = z_mem + z_t
            z_feat = z_mem / float(t + 1) if str(cfg.readout_time_norm) == "mean" else z_mem
            logits_t = model.out(z_feat)
            ce_t = F.cross_entropy(logits_t, y)

            probs = F.softmax(logits_t, dim=1)
            y_oh = F.one_hot(y, num_classes=dout).to(probs.dtype)
            e_out = probs - y_oh

            dW_out = (e_out.t() @ z_feat) / float(max(1, bs))
            db_out = torch.mean(e_out, dim=0)

            c_hidden = float(global_mod_scale) * (e_out @ model.out.weight.detach())

            pre_trace = float(trace_decay_pre) * pre_trace + x_t.detach()

            psi = beta * z_soft * (1.0 - z_soft)
            sens = psi * abs(float(cfg.omega_u_scale))
            phi_sig = _phi_signal(phi, float(cfg.phi_cap), mode=phi_sig_mode_l, power=float(phi_gate_power))

            if post_src == "sens":
                post_in = sens
            elif post_src == "phi":
                post_in = phi_sig
            else:
                post_in = mix_a * sens + (1.0 - mix_a) * phi_sig

            post_trace = float(trace_decay_post) * post_trace + post_in.detach()
            post_credit = post_trace
            if int(phi_gate_enable) == 1:
                post_credit = post_credit * phi_sig

            if model.fc_mode == "shared":
                beta_mask = float(model.fc._mask_beta.item())
                p_mask = torch.sigmoid(beta_mask * model.fc.mask_logits.detach())
                pre_eff = pre_trace @ p_mask.t()  # [B,H]

                ge = float(hidden_update_scale) * c_hidden * (pre_eff * post_credit)
                dsw = torch.mean(ge, dim=0)
                sw = torch.clamp(
                    model.fc.shared_w.detach(),
                    min=float(cfg.fc_shared_w_min),
                    max=float(cfg.fc_shared_w_max),
                )
                dlog = torch.einsum(
                    "bh,bd->hd",
                    float(hidden_update_scale) * c_hidden * post_credit * sw.view(1, -1),
                    pre_trace,
                ) / float(max(1, bs))
                dlog = torch.clamp(dlog, min=-float(mask_logit_clip), max=float(mask_logit_clip))

                db_h = torch.mean(float(hidden_update_scale) * c_hidden * post_credit, dim=0) if model.fc.bias is not None else None

                if int(homeo_enable) == 1 and db_h is not None:
                    z_mean = torch.mean(z_t.detach(), dim=0)
                    rate_ema = (1.0 - rate_beta) * rate_ema + rate_beta * z_mean
                    db_h = db_h + (-(float(homeo_eta)) * (float(homeo_target) - rate_ema))

                if int(rule_use_optimizer) == 1 and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    _assign_rule_grad(model.out.weight, _norm_update(dW_out, grad_norm_mode, grad_norm_eps))
                    _assign_rule_grad(model.out.bias, _norm_update(db_out, grad_norm_mode, grad_norm_eps))
                    _assign_rule_grad(model.fc.shared_w, _norm_update(dsw, grad_norm_mode, grad_norm_eps))
                    _assign_rule_grad(model.fc.mask_logits, _norm_update(dlog, grad_norm_mode, grad_norm_eps))
                    _assign_rule_grad(model.fc.bias, _norm_update(db_h, grad_norm_mode, grad_norm_eps) if db_h is not None else None)

                    if float(rule_grad_clip) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(rule_grad_clip))
                    optimizer.step()
                else:
                    model.out.weight.data -= float(lr_out) * dW_out
                    model.out.bias.data -= float(lr_out) * db_out
                    model.fc.shared_w.data -= float(lr_sw) * dsw
                    model.fc.mask_logits.data -= float(lr_mask) * dlog
                    if db_h is not None:
                        model.fc.bias.data -= float(lr_bias) * db_h

                model.fc.shared_w.data.clamp_(min=float(cfg.fc_shared_w_min), max=float(cfg.fc_shared_w_max))
                model.fc.mask_logits.data.clamp_(min=-float(mask_logit_bound), max=float(mask_logit_bound))
            else:
                # Dense FC mode
                ge = float(hidden_update_scale) * c_hidden * post_credit
                dW_h = torch.einsum("bh,bd->hd", ge, pre_trace) / float(max(1, bs))
                db_h = torch.mean(ge, dim=0) if model.fc.bias is not None else None

                if int(homeo_enable) == 1 and db_h is not None:
                    z_mean = torch.mean(z_t.detach(), dim=0)
                    rate_ema = (1.0 - rate_beta) * rate_ema + rate_beta * z_mean
                    db_h = db_h + (-(float(homeo_eta)) * (float(homeo_target) - rate_ema))

                if int(rule_use_optimizer) == 1 and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    _assign_rule_grad(model.out.weight, _norm_update(dW_out, grad_norm_mode, grad_norm_eps))
                    _assign_rule_grad(model.out.bias, _norm_update(db_out, grad_norm_mode, grad_norm_eps))
                    _assign_rule_grad(model.fc.weight, _norm_update(dW_h, grad_norm_mode, grad_norm_eps))
                    _assign_rule_grad(model.fc.bias, _norm_update(db_h, grad_norm_mode, grad_norm_eps) if db_h is not None else None)

                    if float(rule_grad_clip) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(rule_grad_clip))
                    optimizer.step()
                else:
                    model.out.weight.data -= float(lr_out) * dW_out
                    model.out.bias.data -= float(lr_out) * db_out
                    model.fc.weight.data -= float(lr_fc) * dW_h
                    if db_h is not None:
                        model.fc.bias.data -= float(lr_bias) * db_h

            phi = phi_next.detach()
            theta = theta_next.detach()
            vv = v_next.detach()

            ce_sum += float(ce_t.item()) * bs
            z_sum += float(torch.mean(z_t).item())
            n_terms += bs
            n_steps += 1

    return ce_sum / max(1, n_terms), z_sum / max(1, n_steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="/Users/moon/Documents/snn/mnist_pdr/MAP/paper/model/data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
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

    # Online e-prop style rule knobs.
    ap.add_argument("--lr_out", type=float, default=1e-2)
    ap.add_argument("--lr_fc", type=float, default=2e-4)
    ap.add_argument("--lr_sw", type=float, default=2e-4)
    ap.add_argument("--lr_mask", type=float, default=1e-4)
    ap.add_argument("--lr_bias", type=float, default=2e-4)
    ap.add_argument("--trace_decay_pre", type=float, default=0.95)
    ap.add_argument("--trace_decay_post", type=float, default=0.95)
    ap.add_argument("--global_mod_scale", type=float, default=1.0)
    ap.add_argument("--hidden_update_scale", type=float, default=0.2)

    # MAP-specific eligibility options requested.
    ap.add_argument("--eprop_post_src", choices=["sens", "phi", "mix"], default="sens")
    ap.add_argument("--post_trace_mix_alpha", type=float, default=0.5)
    ap.add_argument("--phi_signal_mode", choices=["linear", "legacy"], default="linear")
    ap.add_argument("--phi_gate_enable", type=int, choices=[0, 1], default=0)
    ap.add_argument("--phi_gate_power", type=float, default=1.0)

    ap.add_argument("--homeo_enable", type=int, choices=[0, 1], default=0)
    ap.add_argument("--homeo_eta", type=float, default=0.01)
    ap.add_argument("--homeo_target", type=float, default=0.15)
    ap.add_argument("--mask_logit_clip", type=float, default=2e-3)
    ap.add_argument("--mask_logit_bound", type=float, default=8.0)

    ap.add_argument("--rule_use_optimizer", type=int, choices=[0, 1], default=1)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--sgd_momentum", type=float, default=0.0)
    ap.add_argument("--grad_norm_mode", choices=["none", "rms", "meanabs"], default="rms")
    ap.add_argument("--grad_norm_eps", type=float, default=1e-3)
    ap.add_argument("--rule_grad_clip", type=float, default=0.0)

    ap.add_argument(
        "--ckpt_out",
        default="/Users/moon/Documents/snn/mnist_pdr/MAP/paper/model/out_ckpt/mnist_map_real_3state_1l_online_eprop.pt",
    )
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    device = str(args.device)

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

    model = MAPReal3State1L(cfg).to(device)

    tr = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(args.data_dir), train=True, download=True, transform=tr)
    test_ds = datasets.MNIST(root=str(args.data_dir), train=False, download=True, transform=tr)
    train_ds = _subset(train_ds, int(args.train_limit))
    test_ds = _subset(test_ds, int(args.test_limit))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )

    optim_groups = []
    if float(args.lr_out) > 0.0:
        optim_groups.append({"params": [model.out.weight, model.out.bias], "lr": float(args.lr_out)})

    if model.fc_mode == "shared":
        if float(args.lr_sw) > 0.0:
            optim_groups.append({"params": [model.fc.shared_w], "lr": float(args.lr_sw)})
        if float(args.lr_mask) > 0.0:
            optim_groups.append({"params": [model.fc.mask_logits], "lr": float(args.lr_mask)})
        if float(args.lr_bias) > 0.0 and model.fc.bias is not None:
            optim_groups.append({"params": [model.fc.bias], "lr": float(args.lr_bias)})
    else:
        if float(args.lr_fc) > 0.0:
            optim_groups.append({"params": [model.fc.weight], "lr": float(args.lr_fc)})
        if float(args.lr_bias) > 0.0 and model.fc.bias is not None:
            optim_groups.append({"params": [model.fc.bias], "lr": float(args.lr_bias)})

    optimizer = None
    if int(args.rule_use_optimizer) == 1 and optim_groups:
        if str(args.optimizer) == "sgd":
            optimizer = torch.optim.SGD(
                optim_groups,
                momentum=float(args.sgd_momentum),
                weight_decay=float(args.weight_decay),
            )
        else:
            optimizer = torch.optim.AdamW(
                optim_groups,
                betas=(float(args.adam_beta1), float(args.adam_beta2)),
                weight_decay=float(args.weight_decay),
            )

    best_acc = -1.0
    best_ep = -1

    for ep in range(1, int(args.epochs) + 1):
        if model.fc_mode == "shared":
            if int(args.epochs) <= 1:
                beta = float(args.fc_mask_beta_end)
            else:
                r = float(ep - 1) / float(int(args.epochs) - 1)
                beta = float(args.fc_mask_beta_start) + (float(args.fc_mask_beta_end) - float(args.fc_mask_beta_start)) * r
            model.set_mask_beta(beta)
        else:
            beta = 0.0

        ce_tr, z_tr = _run_online_epoch(
            model=model,
            loader=train_loader,
            device=device,
            input_mode=str(args.train_input_mode),
            rate_scale=float(args.rate_scale),
            lr_out=float(args.lr_out),
            lr_fc=float(args.lr_fc),
            lr_sw=float(args.lr_sw),
            lr_mask=float(args.lr_mask),
            lr_bias=float(args.lr_bias),
            trace_decay_pre=float(args.trace_decay_pre),
            trace_decay_post=float(args.trace_decay_post),
            global_mod_scale=float(args.global_mod_scale),
            hidden_update_scale=float(args.hidden_update_scale),
            eprop_post_src=str(args.eprop_post_src),
            post_trace_mix_alpha=float(args.post_trace_mix_alpha),
            phi_signal_mode=str(args.phi_signal_mode),
            phi_gate_enable=int(args.phi_gate_enable),
            phi_gate_power=float(args.phi_gate_power),
            homeo_enable=int(args.homeo_enable),
            homeo_eta=float(args.homeo_eta),
            homeo_target=float(args.homeo_target),
            mask_logit_clip=float(args.mask_logit_clip),
            mask_logit_bound=float(args.mask_logit_bound),
            optimizer=optimizer,
            rule_use_optimizer=int(args.rule_use_optimizer),
            grad_norm_mode=str(args.grad_norm_mode),
            grad_norm_eps=float(args.grad_norm_eps),
            rule_grad_clip=float(args.rule_grad_clip),
        )

        acc, z_ev, ph, th, vv, om = _eval(
            model=model,
            loader=test_loader,
            device=device,
            input_mode=str(args.eval_input_mode),
            rate_scale=float(args.rate_scale),
        )

        if acc > best_acc:
            best_acc = float(acc)
            best_ep = int(ep)

        mp, mh = model.mask_stats()
        if model.fc_mode == "dense":
            sw_min = float(model.fc.weight.detach().min().item())
            sw_max = float(model.fc.weight.detach().max().item())
        else:
            sw = torch.clamp(model.fc.shared_w.detach(), min=float(cfg.fc_shared_w_min), max=float(cfg.fc_shared_w_max))
            sw_min = float(sw.min().item())
            sw_max = float(sw.max().item())

        print(
            f"[ep {ep:02d}] ce={ce_tr:.4f} test={acc:.2f}% "
            f"T={int(cfg.T)} H={int(cfg.h)} fc={str(cfg.fc_mode)} "
            f"mask(beta={beta:.2f},p={mp:.3f},h={mh:.3f}) sw=({sw_min:.3f},{sw_max:.3f}) "
            f"rule(post={str(args.eprop_post_src)},phiGate={int(args.phi_gate_enable)},phiSig={str(args.phi_signal_mode)},phiPow={float(args.phi_gate_power):.2f}) "
            f"z_tr={z_tr:.3f} z_ev={z_ev:.3f} "
            f"phi=({ph[0]:.3f},{ph[1]:.3f}) th=({th[0]:.3f},{th[1]:.3f}) v=({vv[0]:.3f},{vv[1]:.3f}) om=({om[0]:.3f},{om[1]:.3f})"
        )

    ckpt = {
        "arch": "mnist_map_real_3state_1layer_online_eprop",
        "state_dict": model.state_dict(),
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
            "learning_rule": "online_eprop_local",
            "eprop_post_src": str(args.eprop_post_src),
            "post_trace_mix_alpha": float(args.post_trace_mix_alpha),
            "phi_signal_mode": str(args.phi_signal_mode),
            "phi_gate_enable": int(args.phi_gate_enable),
            "phi_gate_power": float(args.phi_gate_power),
            "best_test_acc": float(best_acc),
            "best_epoch": int(best_ep),
        },
        "best_test_acc": float(best_acc),
        "best_epoch": int(best_ep),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt_out)), exist_ok=True)
    torch.save(ckpt, str(args.ckpt_out))
    print(f"[ok] wrote ckpt={args.ckpt_out} best={best_acc:.2f}% best_epoch={best_ep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
