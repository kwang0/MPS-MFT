import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import tenpy
import pickle

outputfile_name = f"data/results_U_-4.0_t_p_0.075.pkl"

with open(outputfile_name, "rb") as f:
    data = pickle.load(f)

alpha_list = data['alpha_list']
alpha_list = np.stack(alpha_list, axis=0)

beta_list = data['beta_list']
beta_list = np.stack(beta_list, axis=0)
beta_list0 = beta_list[:,0,:,:]
beta_list1 = beta_list[:,1,:,:]
# alist = alpha_list
alist = beta_list0

eps = 1e-12
threshold = 1e-4

idx0 = 1
num_times = beta_list.shape[0]

fig, ax = plt.subplots()
img = ax.imshow(np.abs(alist[idx0,:,:] - alist[idx0-1,:,:])/np.abs(alist[idx0-1,:,:] + eps) > threshold)
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
    img.set_data(np.abs(alist[i,:,:] - alist[i-1,:,:])/np.abs(alist[i-1,:,:] + eps) > threshold)        # update image data
    ax.set_title(f"Slice {i} of {num_times}")     # update title (optional)
    fig.canvas.draw_idle()             # redraw

# connect the slider to the update function
slider.on_changed(update)

plt.show()