import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import tenpy
import pickle

# outputfile_name = f"data/results_U_-4.0_t_p_0.075.pkl"

# with open(outputfile_name, "rb") as f:
#     data = pickle.load(f)

# alpha_list = data['alpha_list']
# alpha_list = np.stack(alpha_list, axis=0)

# beta_list = data['beta_list']
# beta_list = np.stack(beta_list, axis=0)
# beta_list0 = beta_list[:,0,:,:]
# beta_list1 = beta_list[:,1,:,:]
# # alist = alpha_list
# alist = beta_list0

outputfile_name = f"stateless_data/results_L_64_U_8.0_t0_1.0_t_p_0.1_chi_1000_cpu.h5"

with h5py.File(outputfile_name, "r") as f:
    alpha_list = f['alpha_list'][:]
    beta_list = f['beta_list'][:]

alpha_list = np.transpose(alpha_list) # Match ordering in julia
beta_list = np.transpose(beta_list)

alpha_list[abs(alpha_list) < 1e-4] = 0.0
beta_list[abs(beta_list) < 1e-4] = 0.0

alpha_list = alpha_list[:,:,0,0,:]
beta_list0 = beta_list[0,:,:,0,0,:]
beta_list1 = beta_list[1,:,:,0,0,:]
alist = alpha_list

eps = 1e-12
threshold = 1e-3

idx0 = 1
num_times = alpha_list.shape[-1]

# --- Old slider plot (commented out) ---
fig, ax = plt.subplots()
img = ax.imshow(np.abs(alist[:,:,idx0] - alist[:,:,idx0-1])/np.abs(alist[:,:,idx0-1] + eps) > threshold)
# img = ax.imshow(np.abs(alist[:,:,idx0] - alist[:,:,idx0-1])/np.abs(alist[:,:,idx0-1] + eps), vmin=0, vmax=1, cmap='viridis')
# img = ax.imshow(np.abs(alist[:,:,idx0] - alist[:,:,idx0-1]), vmin=0, vmax=1e-3, cmap='viridis')
fig.colorbar(img, ax=ax, label="Relative Change")
ax_slider = fig.add_axes([0.25, 0.05, 0.50, 0.03])
slider = widgets.Slider(
    ax_slider,            # the axes to draw the slider in
    "Slice",              # slider label
    1,                  # minimum value
    num_times - 1,                    # maximum value
    valinit=idx0,         # initial value
    valfmt="%0.0f"        # format as integer
)

def update(val):
    i = int(round(val))
    img.set_data(np.abs(alist[:,:,i] - alist[:,:,i-1])/np.abs(alist[:,:,i-1] + eps) > threshold)        # update image data
    # img.set_data(np.abs(alist[:,:,i] - alist[:,:,i-1])/np.abs(alist[:,:,i-1] + eps))        # update image data
    # img.set_data(np.abs(alist[:,:,i] - alist[:,:,i-1]))        # update image data
    ax.set_title(f"Slice {i} of {num_times}")     # update title (optional)
    fig.canvas.draw_idle()             # redraw

# connect the slider to the update function
slider.on_changed(update)

# --- Old std-dev plot (commented out) ---
# std_map = np.std(alist[:, :, -20:-1], axis=-1) / (alist[:,:,-1] + eps)
# fig, ax = plt.subplots()
# im = ax.imshow(std_map, aspect='auto', cmap='viridis')
# ax.set_title("Standard Deviation of alist over time")
# ax.set_xlabel("Dimension 1")
# ax.set_ylabel("Dimension 0")
# fig.colorbar(im, ax=ax, label="Std Dev")

# --- close_ab-style convergence plot (RMS relative diagonal errors) ---
# def close_ab_metrics(alpha_prev, alpha_cur, beta_prev, beta_cur, r_range, eps=1e-12):
#     alpha_rms = []
#     beta_rms = []

#     # alpha: shape [L, L, 2, 2]
#     for j in range(2):
#         for jp in range(2):
#             for r in range(r_range + 1):
#                 n_diag = alpha_prev.shape[0] - r
#                 if n_diag <= 0:
#                     continue
#                 idx = np.arange(n_diag)
#                 a = alpha_prev[idx, idx + r, j, jp]
#                 am = alpha_cur[idx, idx + r, j, jp]
#                 errs = (am - a) / (np.abs(a) + eps)
#                 rms_err = np.sqrt(np.mean(errs ** 2))
#                 alpha_rms.append(rms_err)

#     # beta: shape [2, L, L, 2, 2]
#     for sigma in range(2):
#         for j in range(2):
#             for jp in range(2):
#                 for r in range(r_range + 1):
#                     n_diag = beta_prev.shape[1] - r
#                     if n_diag <= 0:
#                         continue
#                     idx = np.arange(n_diag)
#                     b = beta_prev[sigma, idx, idx + r, j, jp]
#                     bm = beta_cur[sigma, idx, idx + r, j, jp]
#                     errs = (bm - b) / (np.abs(b) + eps)
#                     rms_err = np.sqrt(np.mean(errs ** 2))
#                     beta_rms.append(rms_err)

#     alpha_max = np.max(alpha_rms) if alpha_rms else 0.0
#     beta_max = np.max(beta_rms) if beta_rms else 0.0
#     return alpha_max, beta_max, max(alpha_max, beta_max)


# r_range = 4
# iters = np.arange(1, num_times)
# alpha_max_errs = np.zeros(num_times - 1)
# beta_max_errs = np.zeros(num_times - 1)
# overall_max_errs = np.zeros(num_times - 1)

# for t in range(1, num_times):
#     a_prev = alpha_list[:, :, :, :, t - 1]
#     a_cur = alpha_list[:, :, :, :, t]
#     b_prev = beta_list[:, :, :, :, :, t - 1]
#     b_cur = beta_list[:, :, :, :, :, t]
#     alpha_max, beta_max, overall_max = close_ab_metrics(
#         a_prev, a_cur, b_prev, b_cur, r_range, eps=eps
#     )
#     alpha_max_errs[t - 1] = alpha_max
#     beta_max_errs[t - 1] = beta_max
#     overall_max_errs[t - 1] = overall_max

# converged = overall_max_errs <= threshold

# fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
# axes[0].plot(iters, alpha_max_errs, label="alpha max RMS rel err", linewidth=1.5)
# axes[0].plot(iters, beta_max_errs, label="beta max RMS rel err", linewidth=1.5)
# axes[0].axhline(threshold, color="red", linestyle="--", linewidth=1.0, label=f"threshold={threshold}")
# axes[0].set_yscale("log")
# axes[0].set_ylabel("Max RMS Relative Error")
# axes[0].set_title("close_ab-style Convergence (between successive saved iterations)")
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)

# axes[1].plot(iters, converged.astype(int), drawstyle="steps-mid", color="black")
# axes[1].set_xlabel("Iteration index")
# axes[1].set_ylabel("Converged")
# axes[1].set_yticks([0, 1])
# axes[1].set_yticklabels(["No", "Yes"])
# axes[1].grid(True, alpha=0.3)

# plt.tight_layout()

plt.show()
