import torch
import torch.nn as nn

from constants import *


# ----------------------------
# Losses (GPU)
# ----------------------------
def criterion_pde(net, xyz):
    # xyz: (B,3) on GPU
    xyz = xyz.detach().clone().float().requires_grad_(True)

    uvwp = net(xyz)  # (B,4)
    u = uvwp[:, 0:1]
    v = uvwp[:, 1:2]
    w = uvwp[:, 2:3]
    p = uvwp[:, 3:4]

    # ---------- first derivatives ----------
    ones = torch.ones_like(u)

    grad_u = torch.autograd.grad(u, xyz, grad_outputs=ones, create_graph=True)[0]  # (B,3)
    grad_v = torch.autograd.grad(v, xyz, grad_outputs=ones, create_graph=True)[0]
    grad_w = torch.autograd.grad(w, xyz, grad_outputs=ones, create_graph=True)[0]
    grad_p = torch.autograd.grad(p, xyz, grad_outputs=ones, create_graph=True)[0]

    u_x, u_y, u_z = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    v_x, v_y, v_z = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
    w_x, w_y, w_z = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]
    P_x, P_y, P_z = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

    # ---------- second derivatives (diagonal Hessian terms) ----------
    u_xx = torch.autograd.grad(u_x, xyz, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyz, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    u_zz = torch.autograd.grad(u_z, xyz, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, 2:3]

    v_xx = torch.autograd.grad(v_x, xyz, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, xyz, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
    v_zz = torch.autograd.grad(v_z, xyz, grad_outputs=torch.ones_like(v_z), create_graph=True)[0][:, 2:3]

    w_xx = torch.autograd.grad(w_x, xyz, grad_outputs=torch.ones_like(w_x), create_graph=True)[0][:, 0:1]
    w_yy = torch.autograd.grad(w_y, xyz, grad_outputs=torch.ones_like(w_y), create_graph=True)[0][:, 1:2]
    w_zz = torch.autograd.grad(w_z, xyz, grad_outputs=torch.ones_like(w_z), create_graph=True)[0][:, 2:3]

    # XX_scale = (X_scale ** 2)
    # YY_scale = (Y_scale ** 2)
    # ZZ_scale = (Z_scale ** 2)
    # UU_scale = U_scale ** 2
    XX_scale = 1
    YY_scale = 1
    ZZ_scale = 1
    UU_scale = 1
    # X_scale = Y_scale = Z_scale = 1
    # ===== inspect term scales using YOUR variables =====
    """"
    with torch.no_grad():
        # momentum-x blocks (match your formula)
        conv_x = (u * u_x / X_scale + v * u_y / Y_scale + w * u_z / Z_scale)
        pres_x = (1.0 / rho) * (P_x / (X_scale * UU_scale))
        diff_x = Diff * (u_xx / XX_scale + u_yy / YY_scale + u_zz / ZZ_scale)

        # momentum-y
        conv_y = (u * v_x / X_scale + v * v_y / Y_scale + w * v_z / Z_scale)
        pres_y = (1.0 / rho) * (P_y / (Y_scale * UU_scale))
        diff_y = Diff * (v_xx / XX_scale + v_yy / YY_scale + v_zz / ZZ_scale)

        # momentum-z
        conv_z = (u * w_x / X_scale + v * w_y / Y_scale + w * w_z / Z_scale)
        pres_z = (1.0 / rho) * (P_z / (Z_scale * UU_scale))
        diff_z = Diff * (w_xx / XX_scale + w_yy / YY_scale + w_zz / ZZ_scale)

        # continuity parts (as in your form)
        cont_x = (u_x / X_scale)
        cont_y = (v_y / Y_scale)
        cont_z = (w_z / Z_scale)

        # helpers
        mag = lambda t: t.abs().mean().item()

        StringUtils.print(
            f"[Mx] |conv|:{mag(conv_x):.2e} |pres|:{mag(pres_x):.2e} |diff|:{mag(diff_x):.2e}   "
            f"[My] |conv|:{mag(conv_y):.2e} |pres|:{mag(pres_y):.2e} |diff|:{mag(diff_y):.2e}   "
            f"[Mz] |conv|:{mag(conv_z):.2e} |pres|:{mag(pres_z):.2e} |diff|:{mag(diff_z):.2e}   ", color="yellow"
        )
        StringUtils.print(
            f"[Cont] |ux|:{mag(cont_x):.2e} |vy|:{mag(cont_y):.2e} |wz|:{mag(cont_z):.2e}  "
            f"|sum|:{mag(cont_x + cont_y + cont_z):.2e}", color="yellow"
        )
    """

    # Dimensionless PDE residuals
    loss_x = (
            (u * u_x + v * u_y + w * u_z)
            + (P_x)
            - Diff * (u_xx / (U_scale * X_scale) + u_yy / (U_scale * Y_scale) + u_zz / (U_scale * Z_scale))
    )

    loss_y = (
            (u * v_x + v * v_y + w * v_z)
            + (P_y)
            - Diff * (v_xx / (V_scale * X_scale) + v_yy / (V_scale * Y_scale) + v_zz / (V_scale * Z_scale))
    )

    loss_z = (
            (u * w_x + v * w_y + w * w_z)
            + (P_z)
            - Diff * (w_xx / (W_scale * X_scale) + w_yy / (W_scale * Y_scale) + w_zz / (W_scale * Z_scale))
    )
    # Continuity
    loss_divergence = (u_x + v_y + w_z)

    # --- pressure gauge (choose ONE) ---
    gauge_weight = 1e-5

    # Option A: zero-mean gauge (recommended for minibatches)
    p_gauge = (p.mean()).pow(2)  # P is your dimensionless pressure prediction (N,1) or (N,)

    mse = nn.MSELoss()

    mse_loss = (mse(loss_x, torch.zeros_like(loss_x)) +
                mse(loss_y, torch.zeros_like(loss_y)) +
                mse(loss_z, torch.zeros_like(loss_z)) +
                mse(loss_divergence, torch.zeros_like(loss_divergence)) * Lambda_div)

    if add_gauge_loss:
        loss = mse_loss + (gauge_weight * p_gauge)
    else:
        loss = mse_loss

    # Gauge (pick exactly one)
    # zero-mean gauge (works with minibatches)

    if add_l1_loss:
        l1 = nn.L1Loss()
        l1_loss = (l1(loss_x, torch.zeros_like(loss_x)) +
                   l1(loss_y, torch.zeros_like(loss_y)) +
                   l1(loss_z, torch.zeros_like(loss_z)) +
                   l1(loss_divergence, torch.zeros_like(loss_divergence)) * Lambda_div)
        l1_loss = l1_loss.mean()
        loss += l1_loss
    # loss+=gauge_weight * p_gauge

    # loss+=gauge_weight * u_gauge
    # loss+=gauge_weight * v_gauge
    # loss+=gauge_weight * w_gauge
    def chk(name, t):
        if not torch.isfinite(t).all():
            print(f"[NaN/Inf] {name}:",
                  "min", t.min().item(),
                  "max", t.max().item(),
                  "mean", t.mean().item())
            raise RuntimeError(f"Non-finite in {name}")

    chk("uvwp", uvwp)
    chk("u", u);
    chk("v", v);
    chk("w", w);
    chk("p", p)
    # after grads:
    chk("u_x", u_x);
    chk("u_xx", u_xx)  # etc...
    chk("loss_x", loss_x)
    chk("loss_div", loss_divergence)
    return loss


def criterion_pde_single(net, x, y, z):
    # x,y,z,geom_k,inlet_k are small batch tensors on GPU
    # xyz
    x = xyz[:, 0]
    x = xyz[:, 1]
    x = xyz[:, 2]
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    net_internal_mesh_in = torch.cat((x, y, z), 1)

    net_in = net_internal_mesh_in
    uvwp = net(net_in)
    u, v, w, p = uvwp[:, 0:1], uvwp[:, 1:2], uvwp[:, 2:3], uvwp[:, 3:4]
    # u = net_u(net_in).view(-1, 1)
    # print("U",u[:,:10])
    # v = net_v(net_in).view(-1, 1)
    # print("V",v[:,:10])
    # w = net_w(net_in).view(-1, 1)
    # print("W",w[:,:10])
    # P = net_p(net_in).view(-1, 1)
    # print("P",P[:,:10])

    ones_x = torch.ones_like(x)
    ones_y = torch.ones_like(y)
    ones_z = torch.ones_like(z)

    u_x = torch.autograd.grad(u, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]  # fixed

    P_x = torch.autograd.grad(p, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    P_y = torch.autograd.grad(p, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    P_z = torch.autograd.grad(p, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    # Dimensionless PDE residuals
    loss_x = (
            (u * u_x + v * u_y + w * u_z)
            + (P_x)
            - Diff * (u_xx / (U_scale * X_scale) + u_yy / (U_scale * Y_scale) + u_zz / (U_scale * Z_scale))
    )

    loss_y = (
            (u * v_x + v * v_y + w * v_z)
            + (P_y)
            - Diff * (v_xx / (V_scale * X_scale) + v_yy / (V_scale * Y_scale) + v_zz / (V_scale * Z_scale))
    )

    loss_z = (
            (u * w_x + v * w_y + w * w_z)
            + (P_z)
            - Diff * (w_xx / (W_scale * X_scale) + w_yy / (W_scale * Y_scale) + w_zz / (W_scale * Z_scale))
    )
    # Continuity
    loss_divergence = (u_x + v_y + w_z)

    # --- pressure gauge (choose ONE) ---
    gauge_weight = 1e-5

    # Option A: zero-mean gauge (recommended for minibatches)
    p_gauge = (p.mean()).pow(2)  # P is your dimensionless pressure prediction (N,1) or (N,)

    mse = nn.MSELoss()

    mse_loss = (mse(loss_x, torch.zeros_like(loss_x)) +
                mse(loss_y, torch.zeros_like(loss_y)) +
                mse(loss_z, torch.zeros_like(loss_z)) +
                mse(loss_divergence, torch.zeros_like(loss_divergence)) * Lambda_div)

    if add_gauge_loss:
        loss = mse_loss + (gauge_weight * p_gauge)
    else:
        loss = mse_loss

    # Gauge (pick exactly one)
    # zero-mean gauge (works with minibatches)

    if add_l1_loss:
        l1 = nn.L1Loss()
        l1_loss = (l1(loss_x, torch.zeros_like(loss_x)) +
                   l1(loss_y, torch.zeros_like(loss_y)) +
                   l1(loss_z, torch.zeros_like(loss_z)) +
                   l1(loss_divergence, torch.zeros_like(loss_divergence)) * Lambda_div)
        l1_loss = l1_loss.mean()
        loss += l1_loss
    # loss+=gauge_weight * p_gauge

    # loss+=gauge_weight * u_gauge
    # loss+=gauge_weight * v_gauge
    # loss+=gauge_weight * w_gauge

    return loss


def criterion_pde_single_concat_sparse_latent(net, x, y, z, sparse_latent):
    # x,y,z,geom_k,inlet_k are small batch tensors on GPU
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    net_in = torch.cat((x, y, z), 1)

    uvwp = net(net_in, sparse_latent)
    u, v, w, p = uvwp[:, 0:1], uvwp[:, 1:2], uvwp[:, 2:3], uvwp[:, 3:4]
    # u = net_u(net_in).view(-1, 1)
    # print("U",u[:,:10])
    # v = net_v(net_in).view(-1, 1)
    # print("V",v[:,:10])
    # w = net_w(net_in).view(-1, 1)
    # print("W",w[:,:10])
    # P = net_p(net_in).view(-1, 1)
    # print("P",P[:,:10])

    ones_x = torch.ones_like(x)
    ones_y = torch.ones_like(y)
    ones_z = torch.ones_like(z)

    u_x = torch.autograd.grad(u, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]  # fixed

    P_x = torch.autograd.grad(p, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    P_y = torch.autograd.grad(p, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    P_z = torch.autograd.grad(p, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    # X_scale = Y_scale = Z_scale = 1
    # ===== inspect term scales using YOUR variables =====
    """
    with torch.no_grad():
        # momentum-x blocks (match your formula)
        conv_x = (u * u_x / X_scale + v * u_y / Y_scale + w * u_z / Z_scale)
        pres_x = (1.0 / rho) * (P_x / (X_scale * UU_scale))
        diff_x = Diff * (u_xx / XX_scale + u_yy / YY_scale + u_zz / ZZ_scale)

        # momentum-y
        conv_y = (u * v_x / X_scale + v * v_y / Y_scale + w * v_z / Z_scale)
        pres_y = (1.0 / rho) * (P_y / (Y_scale * UU_scale))
        diff_y = Diff * (v_xx / XX_scale + v_yy / YY_scale + v_zz / ZZ_scale)

        # momentum-z
        conv_z = (u * w_x / X_scale + v * w_y / Y_scale + w * w_z / Z_scale)
        pres_z = (1.0 / rho) * (P_z / (Z_scale * UU_scale))
        diff_z = Diff * (w_xx / XX_scale + w_yy / YY_scale + w_zz / ZZ_scale)

        # continuity parts (as in your form)
        cont_x = (u_x / X_scale)
        cont_y = (v_y / Y_scale)
        cont_z = (w_z / Z_scale)

        # helpers
        mag = lambda t: t.abs().mean().item()

        StringUtils.print(
            f"[Mx] |conv|:{mag(conv_x):.2e} |pres|:{mag(pres_x):.2e} |diff|:{mag(diff_x):.2e}   "
            f"[My] |conv|:{mag(conv_y):.2e} |pres|:{mag(pres_y):.2e} |diff|:{mag(diff_y):.2e}   "
            f"[Mz] |conv|:{mag(conv_z):.2e} |pres|:{mag(pres_z):.2e} |diff|:{mag(diff_z):.2e}   ", color="yellow"
        )
        StringUtils.print(
            f"[Cont] |ux|:{mag(cont_x):.2e} |vy|:{mag(cont_y):.2e} |wz|:{mag(cont_z):.2e}  "
            f"|sum|:{mag(cont_x + cont_y + cont_z):.2e}", color="yellow"
        )
    """
    # Dimensionless PDE residuals
    loss_x = (
            (u * u_x + v * u_y + w * u_z)
            + (P_x)
            - Diff * (u_xx / (U_scale * X_scale) + u_yy / (U_scale * Y_scale) + u_zz / (U_scale * Z_scale))
    )

    loss_y = (
            (u * v_x + v * v_y + w * v_z)
            + (P_y)
            - Diff * (v_xx / (V_scale * X_scale) + v_yy / (V_scale * Y_scale) + v_zz / (V_scale * Z_scale))
    )

    loss_z = (
            (u * w_x + v * w_y + w * w_z)
            + (P_z)
            - Diff * (w_xx / (W_scale * X_scale) + w_yy / (W_scale * Y_scale) + w_zz / (W_scale * Z_scale))
    )
    # Continuity
    loss_divergence = (u_x + v_y + w_z)

    # --- pressure gauge (choose ONE) ---
    gauge_weight = 1e-5

    # Option A: zero-mean gauge (recommended for minibatches)
    p_gauge = (p.mean()).pow(2)  # P is your dimensionless pressure prediction (N,1) or (N,)

    mse = nn.MSELoss()

    mse_loss = (mse(loss_x, torch.zeros_like(loss_x)) +
                mse(loss_y, torch.zeros_like(loss_y)) +
                mse(loss_z, torch.zeros_like(loss_z)) +
                mse(loss_divergence, torch.zeros_like(loss_divergence)) * Lambda_div)

    if add_gauge_loss:
        loss = mse_loss + (gauge_weight * p_gauge)
    else:
        loss = mse_loss

    # Gauge (pick exactly one)
    # zero-mean gauge (works with minibatches)

    if add_l1_loss:
        l1 = nn.L1Loss()
        l1_loss = (l1(loss_x, torch.zeros_like(loss_x)) +
                   l1(loss_y, torch.zeros_like(loss_y)) +
                   l1(loss_z, torch.zeros_like(loss_z)) +
                   l1(loss_divergence, torch.zeros_like(loss_divergence)) * Lambda_div)
        l1_loss = l1_loss.mean()
        loss += l1_loss
    # loss+=gauge_weight * p_gauge

    # loss+=gauge_weight * u_gauge
    # loss+=gauge_weight * v_gauge
    # loss+=gauge_weight * w_gauge

    return loss




def criterion_pde_single_concat_sparse_latent_embed_coord(net, xyz, sparse_latent,Re):
    coords = xyz.clone().detach().requires_grad_(True)  # [N,3] leaf
    uvwp = net(coords, sparse_latent)
    u, v, w, p = uvwp[:, 0:1], uvwp[:, 1:2], uvwp[:, 2:3], uvwp[:, 3:4]


    ones_u = torch.ones_like(u)

    # First derivatives
    grads_u = torch.autograd.grad(u, coords, grad_outputs=ones_u, create_graph=True)[0]  # [N,3]
    u_x, u_y, u_z = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]

    grads_v = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x, v_y, v_z = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]

    grads_w = torch.autograd.grad(w, coords, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_x, w_y, w_z = grads_w[:, 0:1], grads_w[:, 1:2], grads_w[:, 2:3]

    grads_p = torch.autograd.grad(p, coords, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    P_x, P_y, P_z = grads_p[:, 0:1], grads_p[:, 1:2], grads_p[:, 2:3]

    # Second derivatives (diagonal Hessian terms)
    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    u_zz = torch.autograd.grad(u_z, coords, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, 2:3]

    v_xx = torch.autograd.grad(v_x, coords, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, coords, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
    v_zz = torch.autograd.grad(v_z, coords, grad_outputs=torch.ones_like(v_z), create_graph=True)[0][:, 2:3]

    w_xx = torch.autograd.grad(w_x, coords, grad_outputs=torch.ones_like(w_x), create_graph=True)[0][:, 0:1]
    w_yy = torch.autograd.grad(w_y, coords, grad_outputs=torch.ones_like(w_y), create_graph=True)[0][:, 1:2]
    w_zz = torch.autograd.grad(w_z, coords, grad_outputs=torch.ones_like(w_z), create_graph=True)[0][:, 2:3]


    # Dimensionless PDE residuals
    loss_x =  u * u_x + v * u_y + w * u_z - (1 / Re) * (u_xx + u_yy + u_zz) + P_x
    loss_y =  u * v_x + v * v_y + w * v_z - (1 / Re) * (v_xx + v_yy + v_zz) + P_y
    loss_z =  u * w_x + v * w_y + w * w_z - (1 / Re) * (w_xx + w_yy + w_zz) + P_z



    """ours:
    loss_x = (
            (u * u_x + v * u_y + w * u_z)
            + (P_x)
            - Diff * (u_xx / (U_scale * X_scale) + u_yy / (U_scale * Y_scale) + u_zz / (U_scale * Z_scale))
    )

    loss_y = (
            (u * v_x + v * v_y + w * v_z)
            + (P_y)
            - Diff * (v_xx / (V_scale * X_scale) + v_yy / (V_scale * Y_scale) + v_zz / (V_scale * Z_scale))
    )

    loss_z = (
            (u * w_x + v * w_y + w * w_z)
            + (P_z)
            - Diff * (w_xx / (W_scale * X_scale) + w_yy / (W_scale * Y_scale) + w_zz / (W_scale * Z_scale))
    )"""

    # Continuity
    loss_divergence = (u_x + v_y + w_z)

    # --- pressure gauge (choose ONE) ---
    gauge_weight = 1e-5

    # Option A: zero-mean gauge (recommended for minibatches)
    p_gauge = (p.mean()).pow(2)  # P is your dimensionless pressure prediction (N,1) or (N,)

    mse = nn.MSELoss()

    mse_loss = (mse(loss_x, torch.zeros_like(loss_x)) +
                mse(loss_y, torch.zeros_like(loss_y)) +
                mse(loss_z, torch.zeros_like(loss_z)) +
                mse(loss_divergence, torch.zeros_like(loss_divergence)) * Lambda_div)

    if add_gauge_loss:
        loss = mse_loss + (gauge_weight * p_gauge)
    else:
        loss = mse_loss

    # Gauge (pick exactly one)
    # zero-mean gauge (works with minibatches)

    if add_l1_loss:
        l1 = nn.L1Loss()
        l1_loss = (l1(loss_x, torch.zeros_like(loss_x)) +
                   l1(loss_y, torch.zeros_like(loss_y)) +
                   l1(loss_z, torch.zeros_like(loss_z)) +
                   l1(loss_divergence, torch.zeros_like(loss_divergence)) * Lambda_div)
        l1_loss = l1_loss.mean()
        loss += l1_loss
    # loss+=gauge_weight * p_gauge

    # loss+=gauge_weight * u_gauge
    # loss+=gauge_weight * v_gauge
    # loss+=gauge_weight * w_gauge

    return loss


def loss_boundary_condition(net,
                            # xb, yb, zb,
                            xb_in, yb_in, zb_in, ub_in, vb_in, wb_in):
    """
    Wall, inlet, outlet(We don't have yet)
    :param net_u:
    :param net_v:
    :param net_w:
    :param xb:
    :param yb:
    :param zb:
    :param xb_in:
    :param yb_in:
    :param zb_in:
    :param ub_in:
    :param vb_in:
    :param wb_in:
    :return:
    """
    # nin_wall = torch.cat((xb, yb, zb), 1)
    # prediction = net(nin_wall)
    # uw = net_u(nin_wall)
    # vw = net_v(nin_wall)
    # ww = net_w(nin_wall)
    # uw, vw, ww = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    nin_in = torch.cat((xb_in, yb_in, zb_in), 1)
    prediction = net(nin_in)
    # uw = net_u(nin_wall)
    # vw = net_v(nin_wall)
    # ww = net_w(nin_wall)
    ui, vi, wi = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    mse = nn.MSELoss(reduction='mean')

    # loss = (mse(uw, ub) + mse(vw, vb) + mse(ww, wb) * W_LAMBDA +
    #       mse(ui, ub_in) + mse(vi, vb_in) + mse(wi, wb_in) * W_LAMBDA)

    # wall terms
    # wall_mse = mse(uw, torch.zeros_like(uw, device=uw.device).float()) + mse(vw, torch.zeros_like(vw,
    #                                                                                               device=vw.device).float()) + mse(
    #     ww, torch.zeros_like(ww, device=ww.device).float())
    # if add_l1_loss:

    # inlet terms
    inlet_mse = mse(ui, ub_in) + mse(vi, vb_in) + mse(wi, wb_in)

    loss_bc = inlet_mse
    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        inlet_l1 = l1(ui, ub_in) + l1(vi, vb_in) + l1(wi, wb_in)
        # wall_l1 = l1(uw, torch.zeros_like(uw, device=uw.device).float()) + l1(vw, torch.zeros_like(vw,
        #                                                                                            device=vw.device).float()) + l1(
        #     ww, torch.zeros_like(ww, device=ww.device).float())
        loss_bc += inlet_l1
    return loss_bc


def loss_boundary_condition_parabolic_noslip(net, xb, yb, zb, xb_in, yb_in, zb_in
                                             ):
    """
    inlet
    :param net_u:
    :param net_v:
    :param net_w:
    :param xb:
    :param yb:
    :param zb:
    :param xb_in:
    :param yb_in:
    :param zb_in:
    :return:
    """
    nin_wall = torch.cat((xb, yb, zb), 1)
    prediction = net(nin_wall)
    uw, vw, ww = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
    mse = nn.MSELoss(reduction='mean')

    wall_mse = (mse(uw, torch.zeros_like(uw, device=uw.device).float())
                + mse(vw, torch.zeros_like(vw, device=vw.device).float())
                + mse(ww, torch.zeros_like(ww, device=ww.device).float()))

    loss_bc = wall_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        # inlet_l1 = l1(ui, ub_in) + l1(vi, vb_in) + l1(wi, wb_in)
        wall_l1 = (l1(uw, torch.zeros_like(uw, device=uw.device).float()) +
                   l1(vw, torch.zeros_like(vw, device=vw.device).float()) +
                   l1(ww, torch.zeros_like(ww, device=ww.device).float()))
        # loss_bc += (wall_l1 + inlet_l1)
        loss_bc += wall_l1

    """
    nin_in = torch.cat((xb_in, yb_in, zb_in), 1)
    prediction = net(nin_in)
    ui, vi, wi = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    r = torch.sqrt( (xb_in - x0)**2 + (yb_in - y0)**2 )

    # parabolic target profile
    w_target = w_max * (1.0 - (r / R) ** 2)
    w_target[r > R] = 0.0  # just in case of tiny numerical noise
    w_target = torch.clamp(w_target, min=0.0)

    mse = nn.MSELoss(reduction='mean')

    # inlet terms
    inlet_mse =mse(wi, w_target)

    loss_bc += inlet_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        inlet_l1 = l1(wi, w_target)

        loss_bc += inlet_l1
    """
    return loss_bc


def loss_boundary_condition_parabolic_noslip_concat_latent_sparse(net, xb, yb, zb, xb_in, yb_in, zb_in, sparse_wall
                                                                  ):
    """
    inlet
    :param net_u:
    :param net_v:
    :param net_w:
    :param xb:
    :param yb:
    :param zb:
    :param xb_in:
    :param yb_in:
    :param zb_in:
    :return:
    """
    nin_wall = torch.cat((xb, yb, zb, sparse_wall), 1)
    prediction = net(nin_wall)
    uw, vw, ww = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
    mse = nn.MSELoss(reduction='mean')

    wall_mse = (mse(uw, torch.zeros_like(uw, device=uw.device).float())
                + mse(vw, torch.zeros_like(vw, device=vw.device).float())
                + mse(ww, torch.zeros_like(ww, device=ww.device).float()))

    loss_bc = wall_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        # inlet_l1 = l1(ui, ub_in) + l1(vi, vb_in) + l1(wi, wb_in)
        wall_l1 = (l1(uw, torch.zeros_like(uw, device=uw.device).float()) +
                   l1(vw, torch.zeros_like(vw, device=vw.device).float()) +
                   l1(ww, torch.zeros_like(ww, device=ww.device).float()))
        # loss_bc += (wall_l1 + inlet_l1)
        loss_bc += wall_l1

    """
    nin_in = torch.cat((xb_in, yb_in, zb_in), 1)
    prediction = net(nin_in)
    ui, vi, wi = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    r = torch.sqrt( (xb_in - x0)**2 + (yb_in - y0)**2 )

    # parabolic target profile
    w_target = w_max * (1.0 - (r / R) ** 2)
    w_target[r > R] = 0.0  # just in case of tiny numerical noise
    w_target = torch.clamp(w_target, min=0.0)

    mse = nn.MSELoss(reduction='mean')

    # inlet terms
    inlet_mse =mse(wi, w_target)

    loss_bc += inlet_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        inlet_l1 = l1(wi, w_target)

        loss_bc += inlet_l1
    """
    return loss_bc


def loss_boundary_condition_parabolic_noslip_concat_latent_sparse_embed_coord(net, xb, yb, zb, xb_in, yb_in, zb_in,
                                                                              sparse_latent):
    """
    inlet
    :param net_u:
    :param net_v:
    :param net_w:
    :param xb:
    :param yb:
    :param zb:
    :param xb_in:
    :param yb_in:
    :param zb_in:
    :return:
    """
    coords = torch.cat([xb, yb, zb], dim=1)  # [B,3]

    prediction = net(coords, sparse_latent)
    uw, vw, ww = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
    mse = nn.MSELoss(reduction='mean')

    wall_mse = (mse(uw, torch.zeros_like(uw, device=uw.device).float())
                + mse(vw, torch.zeros_like(vw, device=vw.device).float())
                + mse(ww, torch.zeros_like(ww, device=ww.device).float()))

    loss_bc = wall_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        # inlet_l1 = l1(ui, ub_in) + l1(vi, vb_in) + l1(wi, wb_in)
        wall_l1 = (l1(uw, torch.zeros_like(uw, device=uw.device).float()) +
                   l1(vw, torch.zeros_like(vw, device=vw.device).float()) +
                   l1(ww, torch.zeros_like(ww, device=ww.device).float()))
        # loss_bc += (wall_l1 + inlet_l1)
        loss_bc += wall_l1
    return loss_bc


# for train_validation
def loss_boundary_condition_parabolic_noslip_concat_latent_sparse_embed_coord_wall(net, xyz_bc,
                                                                                   sparse_wall
                                                                                   ):
    prediction = net(xyz_bc, sparse_wall)
    uw, vw, ww = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
    mse = nn.MSELoss(reduction='mean')

    wall_mse = (mse(uw, torch.zeros_like(uw, device=uw.device).float())
                + mse(vw, torch.zeros_like(vw, device=vw.device).float())
                + mse(ww, torch.zeros_like(ww, device=ww.device).float()))

    loss_bc = wall_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        # inlet_l1 = l1(ui, ub_in) + l1(vi, vb_in) + l1(wi, wb_in)
        wall_l1 = (l1(uw, torch.zeros_like(uw, device=uw.device).float()) +
                   l1(vw, torch.zeros_like(vw, device=vw.device).float()) +
                   l1(ww, torch.zeros_like(ww, device=ww.device).float()))

        loss_bc += wall_l1

    """
    mse = nn.MSELoss(reduction='mean')
    prediction_ = net(outlet1, sparse_outlet1)
    p = prediction_[:, 3:4]  # p is pressure
    prediction__ = net(outlet2, sparse_outlet2)
    p_ = prediction__[:, 3:4]

    outlet_mse = (mse(p, torch.zeros_like(p, device=p.device).float())
                  + mse(p_, torch.zeros_like(p_, device=p_.device).float()))

    loss_bc += outlet_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')

        outlet_l1 = (l1(p, torch.zeros_like(p, device=p.device).float()) +
                     l1(p_, torch.zeros_like(p_, device=p_.device).float()))

        loss_bc += outlet_l1
    """

    return loss_bc


def loss_boundary_condition_parabolic_noslip_concat_latent_sparse_embed_coord_wall_trial(net, xyz_wall,
                                                                                         sparse_wall, inlet_xyz,
                                                                                         inlet_normals, inlet_areas,
                                                                                         sparse_inlet, outlet1, outlet2,
                                                                                         sparse_outlet1, sparse_outlet2,
                                                                                         q_target
                                                                                         ):
    prediction = net(xyz_wall, sparse_wall)
    uw, vw, ww = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
    mse = nn.MSELoss(reduction='mean')

    wall_mse = (mse(uw, torch.zeros_like(uw, device=uw.device).float())
                + mse(vw, torch.zeros_like(vw, device=vw.device).float())
                + mse(ww, torch.zeros_like(ww, device=ww.device).float()))

    loss_bc = wall_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')

        wall_l1 = (l1(uw, torch.zeros_like(uw, device=uw.device).float()) +
                   l1(vw, torch.zeros_like(vw, device=vw.device).float()) +
                   l1(ww, torch.zeros_like(ww, device=ww.device).float()))

        loss_bc += wall_l1

    mse = nn.MSELoss(reduction='mean')
    pred = net(inlet_xyz, sparse_inlet)
    u = pred[:, 0:3]  # (N,3)

    # u·n
    un = (u * inlet_normals).sum(dim=1, keepdim=True)  # (N,1)

    # Discrete surface integral
    Q_pred = torch.sum(un * inlet_areas)  # scalar

    # Target flow rate
    if not torch.is_tensor(q_target):
        Q_target = torch.tensor(q_target, device=Q_pred.device)

    # Loss
    if use_abs:
        loss = (torch.abs(Q_pred) - torch.abs(Q_target)) ** 2
    else:
        loss = (Q_pred - Q_target) ** 2

    loss_bc += loss

    return loss_bc


def loss_boundary_condition_parabolic_noslip_concat_latent_sparse_embed_coord_wall_inlet(net, xyz_bc, xyz_inlet,
                                                                                         uvw_inlet,
                                                                                         sparse_wall,
                                                                                         sparse_inlet
                                                                                         ):
    prediction = net(xyz_bc, sparse_wall)
    uw, vw, ww = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
    mse = nn.MSELoss(reduction='mean')

    wall_mse = (mse(uw, torch.zeros_like(uw, device=uw.device).float())
                + mse(vw, torch.zeros_like(vw, device=vw.device).float())
                + mse(ww, torch.zeros_like(ww, device=ww.device).float()))

    loss_bc = wall_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        # inlet_l1 = l1(ui, ub_in) + l1(vi, vb_in) + l1(wi, wb_in)
        wall_l1 = (l1(uw, torch.zeros_like(uw, device=uw.device).float()) +
                   l1(vw, torch.zeros_like(vw, device=vw.device).float()) +
                   l1(ww, torch.zeros_like(ww, device=ww.device).float()))
        # loss_bc += (wall_l1 + inlet_l1)
        loss_bc += wall_l1

    u_inlet = uvw_inlet[:, 0:1]
    v_inlet = uvw_inlet[:, 1:2]
    w_inlet = uvw_inlet[:, 2:3]

    prediction = net(xyz_inlet, sparse_inlet)
    ui, vi, wi = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    mse = nn.MSELoss(reduction='mean')

    # inlet terms
    inlet_mse = (mse(ui, u_inlet) +
                 mse(vi, v_inlet) +
                 mse(wi, w_inlet))

    loss_bc += inlet_mse

    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        inlet_l1 = (l1(ui, u_inlet) +
                    l1(vi, v_inlet) +
                    l1(wi, w_inlet))

        loss_bc += inlet_l1

    return loss_bc


def loss_boundary_condition_single(net,
                                   xb, yb, zb,
                                   # xb_in, yb_in, zb_in,
                                   # ub_in, vb_in, wb_in
                                   ):
    """
    Wall, inlet, outlet(We don't have yet)
    :param net_u:
    :param net_v:
    :param net_w:
    :param xb:
    :param yb:
    :param zb:
    :param xb_in:
    :param yb_in:
    :param zb_in:
    :param ub_in:
    :param vb_in:
    :param wb_in:
    :return:
    """
    nin_wall = torch.cat((xb, yb, zb), 1)
    prediction = net(nin_wall)
    # uw = net_u(nin_wall)
    # vw = net_v(nin_wall)
    # ww = net_w(nin_wall)
    uw, vw, ww = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    # nin_in = torch.cat((xb_in, yb_in, zb_in), 1)
    # ui = net_u(nin_in)
    # vi = net_v(nin_in)
    # wi = net_w(nin_in)
    mse = nn.MSELoss(reduction='mean')

    # loss = (mse(uw, ub) + mse(vw, vb) + mse(ww, wb) * W_LAMBDA +
    #       mse(ui, ub_in) + mse(vi, vb_in) + mse(wi, wb_in) * W_LAMBDA)

    # wall terms
    wall_mse = (mse(uw, torch.zeros_like(uw, device=uw.device).float()) +
                mse(vw, torch.zeros_like(vw, device=vw.device).float()) +
                mse(ww, torch.zeros_like(ww, device=ww.device).float()))
    # if add_l1_loss:

    # inlet terms
    # inlet_mse = mse(ui, ub_in) + mse(vi, vb_in) + mse(wi, wb_in)

    loss_bc = wall_mse
    # loss_bc = (wall_mse + inlet_mse)
    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        # inlet_l1 = l1(ui, ub_in) + l1(vi, vb_in) + l1(wi, wb_in)
        wall_l1 = (l1(uw, torch.zeros_like(uw, device=uw.device).float()) +
                   l1(vw, torch.zeros_like(vw, device=vw.device).float()) +
                   l1(ww, torch.zeros_like(ww, device=ww.device).float()))
        # loss_bc += (wall_l1 + inlet_l1)
        loss_bc += wall_l1
    return loss_bc


def loss_data_single(net, xzy_data,
                     uvw_data,
                     ):
    u_data = uvw_data[:, 0:1]
    v_data = uvw_data[:, 1:2]
    w_data = uvw_data[:, 2:3]

    prediction = net(xzy_data)
    u_prediction, v_prediction, w_prediction = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    mse = nn.MSELoss(reduction='mean')

    loss_data_ = (mse(u_prediction.squeeze(1), u_data) +
                  mse(v_prediction.squeeze(1), v_data) +
                  mse(w_prediction.squeeze(1), w_data))
    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        loss_data_ += (l1(u_prediction.squeeze(1), u_data) +
                       l1(v_prediction.squeeze(1), v_data) +
                       l1(w_prediction.squeeze(1), w_data))
        # loss_data_ = loss_data_

    return loss_data_


def loss_data_single_concat_sparse_latent(net, x_data, y_data, z_data,
                                          u_data, v_data, w_data, sparse_data
                                          ):
    nin_data_ = torch.cat((x_data, y_data, z_data, sparse_data), 1)

    prediction = net(nin_data_)
    u_prediction, v_prediction, w_prediction = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
    # v_prediction = net_v(nin_data_)
    # w_prediction = net_w(nin_data_)
    # mse = nn.MSELoss()

    # loss = (mse(u_prediction, u_data) + mse(v_prediction, v_data) + mse(w_prediction, w_data) * W_LAMBDA)

    # if add_l1_loss:
    #   l1 = nn.L1Loss()
    #  l1_loss = (l1(u_prediction, u_data) + l1(v_prediction, v_data) + l1(w_prediction, w_data) * W_LAMBDA)
    mse = nn.MSELoss(reduction='mean')

    loss_data_ = (mse(u_prediction.squeeze(1), u_data) +
                  mse(v_prediction.squeeze(1), v_data) +
                  mse(w_prediction.squeeze(1), w_data))
    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        loss_data_ += (l1(u_prediction.squeeze(1), u_data) +
                       l1(v_prediction.squeeze(1), v_data) +
                       l1(w_prediction.squeeze(1), w_data))
        # loss_data_ = loss_data_

    return loss_data_


def loss_data_single_concat_sparse_latent_embed_coord(net, xyz,
                                                      uvw, sparse_latent
                                                      ):
    # coords = torch.cat([x_data, y_data, z_data], dim=1)  # [B,3]
    u_data = uvw[:, 0]
    v_data = uvw[:, 1]
    w_data = uvw[:, 2]
    prediction = net(xyz, sparse_latent)
    u_prediction, v_prediction, w_prediction = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

    # loss = (mse(u_prediction, u_data) + mse(v_prediction, v_data) + mse(w_prediction, w_data) * W_LAMBDA)

    # if add_l1_loss:
    #   l1 = nn.L1Loss()
    #  l1_loss = (l1(u_prediction, u_data) + l1(v_prediction, v_data) + l1(w_prediction, w_data) * W_LAMBDA)
    mse = nn.MSELoss(reduction='mean')

    loss_data_ = (mse(u_prediction.squeeze(1), u_data) +
                  mse(v_prediction.squeeze(1), v_data) +
                  mse(w_prediction.squeeze(1), w_data))
    if add_l1_loss:
        l1 = nn.L1Loss(reduction='mean')
        loss_data_ += (l1(u_prediction.squeeze(1), u_data) +
                       l1(v_prediction.squeeze(1), v_data) +
                       l1(w_prediction.squeeze(1), w_data))
        # loss_data_ = loss_data_
    return loss_data_


# for MR case
import math


# convert  velocity to phase to tackle aliasing
def loss_data_single_concat_sparse_latent_embed_coord_MRI(net, xyz, uvw, sparse_latent):
    # Ensure shapes [B,1]
    u_data = uvw[:, 0:1]
    v_data = uvw[:, 1:2]
    w_data = uvw[:, 2:3]

    pred = net(xyz, sparse_latent)
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]
    w_pred = pred[:, 2:3]

    # Phase (ASSUMES velocities already normalized appropriately)
    phi_u_pred = 2 * math.pi * u_pred
    phi_v_pred = 2 * math.pi * v_pred
    phi_w_pred = 2 * math.pi * w_pred

    phi_u_data = 2 * math.pi * u_data
    phi_v_data = 2 * math.pi * v_data
    phi_w_data = 2 * math.pi * w_data

    # Complex mapping exp(-i phi) = cos(phi) - i sin(phi)
    Su_pred = torch.cos(phi_u_pred) - 1j * torch.sin(phi_u_pred)
    Sv_pred = torch.cos(phi_v_pred) - 1j * torch.sin(phi_v_pred)
    Sw_pred = torch.cos(phi_w_pred) - 1j * torch.sin(phi_w_pred)

    Su_data = torch.cos(phi_u_data) - 1j * torch.sin(phi_u_data)
    Sv_data = torch.cos(phi_v_data) - 1j * torch.sin(phi_v_data)
    Sw_data = torch.cos(phi_w_data) - 1j * torch.sin(phi_w_data)

    # Complex residuals
    du = Su_pred - Su_data
    dv = Sv_pred - Sv_data
    dw = Sw_pred - Sw_data

    # L2 on complex: mean |diff|^2
    loss = (
            torch.mean(torch.abs(du) ** 2) +
            torch.mean(torch.abs(dv) ** 2) +
            torch.mean(torch.abs(dw) ** 2)
    )

    # Optional L1 on complex: mean |diff|
    if add_l1_loss:
        loss = loss + (
                torch.mean(torch.abs(du)) +
                torch.mean(torch.abs(dv)) +
                torch.mean(torch.abs(dw))
        )

    return loss


def loss_data(net_u, net_v, net_w,
              x_data, y_data, z_data,
              u_data, v_data, w_data
              ):
    nin_data_ = torch.cat((x_data, y_data, z_data), 1)

    u_prediction = net_u(nin_data_)
    v_prediction = net_v(nin_data_)
    w_prediction = net_w(nin_data_)
    # mse = nn.MSELoss()

    # loss = (mse(u_prediction, u_data) + mse(v_prediction, v_data) + mse(w_prediction, w_data) * W_LAMBDA)

    # if add_l1_loss:
    #   l1 = nn.L1Loss()
    #  l1_loss = (l1(u_prediction, u_data) + l1(v_prediction, v_data) + l1(w_prediction, w_data) * W_LAMBDA)
    mse = nn.MSELoss(reduction='mean')
    l1 = nn.L1Loss(reduction='mean')
    loss_data_ = (mse(u_prediction.squeeze(1), u_data) +
                  mse(v_prediction.squeeze(1), v_data) +
                  mse(w_prediction.squeeze(1), w_data))
    if add_l1_loss:
        loss_data_ += (l1(u_prediction.squeeze(1), u_data) +
                       l1(v_prediction.squeeze(1), v_data) +
                       l1(w_prediction.squeeze(1), w_data))
        # loss_data_ = loss_data_

    return loss_data_

# def loss_data(net_u, net_v, net_w,
#               x_data, y_data, z_data, u_data, v_data, w_data
#
#               ):
#     nin_data_ = torch.cat((x_data, y_data, z_data), 1)
#
#     u_prediction = net_u(nin_data_)
#     v_prediction = net_v(nin_data_)
#     w_prediction = net_w(nin_data_)
#     # mse = nn.MSELoss()
#
#     # loss = (mse(u_prediction, u_data) + mse(v_prediction, v_data) + mse(w_prediction, w_data) * W_LAMBDA)
#
#     # if add_l1_loss:
#     #   l1 = nn.L1Loss()
#     #  l1_loss = (l1(u_prediction, u_data) + l1(v_prediction, v_data) + l1(w_prediction, w_data) * W_LAMBDA)
#     mse = nn.MSELoss(reduction='mean')
#     l1 = nn.L1Loss(reduction='mean')
#     loss_data_ = (mse(u_prediction, u_data) +
#                   mse(v_prediction, v_data) +
#                   mse(w_prediction, w_data))
#     if add_l1_loss:
#         loss_data_ += (l1(u_prediction, u_data) +
#                        l1(v_prediction, v_data) +
#                        l1(w_prediction, w_data))
#         # loss_data_ = W_LAMBDA * loss_data_
#
#     return loss_data_
