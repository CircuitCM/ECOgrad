from pathlib import Path
import numpy as np
import pyvista as pv
import math as mt
import datetime
import os

"""To run this script pyvista needs to be installed."""

l2=np.linalg.norm

def print_camera_position(p: pv.Plotter):
    cam = p.camera
    posi=(
        cam.position,
        cam.focal_point,
        cam.up,
        cam.clipping_range,
        cam.view_angle,
        cam.parallel_scale,
        cam.parallel_projection,
    )
    print("Camera snapshot:", posi)

def restore_camera(plotter: pv.Plotter, snapshot):
    """Restore a camera snapshot onto a plotter."""
    (position, focal_point, view_up,
     clipping_range, view_angle,
     parallel_scale, parallel_projection) = snapshot

    plotter.camera_position = (position, focal_point, view_up)
    plotter.camera.clipping_range = clipping_range
    plotter.camera.view_angle = view_angle
    plotter.camera.parallel_scale = parallel_scale
    plotter.camera.parallel_projection = parallel_projection
    plotter.render()

def save_screenshot(p: pv.Plotter,dir):
    # timestamped filename so you don’t overwrite
    fname = Path(os.path.join(dir,"screenshot_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"))
    p.screenshot(fname)
    print(f"Saved screenshot to {fname.resolve()}")
    
def zoom_in(pl):
    pl.camera.Zoom(1.01)   # 1% closer
    pl.render()

def zoom_out(pl):
    pl.camera.Zoom(0.99)   # 1% farther
    pl.render()

def plot_ecg_scene(
    gradient=np.array([.5, .95]),
    sample_dir=.7,
    z_circle: float = 0.5,
    plane_half_width: float = 6.0,
    circle_resolution: int = 200,
    plane_resolution: int = 1000,
    sigm=0.,
    ver=2,
    title='Marks Ratio Expectation Bounds',
    window_dims=(1050,1050),
    pic_dir=None,
    camera_default=None
):
    """
    Visualise a tilted plane, a floating unit-circle, the gradient arrow, and the
    coloured (uᵀg)u ellipse drawn on that plane.

    Parameters
    ----------
    gradient : (2,) np.ndarray
        2-D gradient [g_x, g_y].
    sample_dir : (2,) np.ndarray|float
        Initial direction to sample gradient from. If array it will be scaled to norm(1). If float it will mix the true gradient and random orthogonal direction
    z_circle : float
        Height of the floating unit circle.
    plane_half_width : float
        Half the edge length of the square plane (total width = 2*plane_half_width).
    circle_resolution : int
        Number of sample points for circle / ellipse.
    plane_resolution : int
        Grid resolution (per axis) for the plane surface.
    """
    g = gradient
    g_len = l2(g)
    #print('GNORM',g_len)
    g_normed = g / g_len                           # unit vector along gradient
    perp_hat = np.array([-g_normed[1], g_normed[0]])  # unit vector ⟂ to gradient
    if not isinstance(sample_dir,np.ndarray):
        # choose u (≈ ⟂ to g) & compute ĝ
        u_vec = ((1-sample_dir)*perp_hat + sample_dir * g_normed)
        u_vec /= l2(u_vec)
        sample_dir = (u_vec @ g) * u_vec
    sd=sample_dir/l2(sample_dir)

    # --- square plane ---
    u = np.linspace(-plane_half_width, plane_half_width, plane_resolution)
    v = np.linspace(-plane_half_width, plane_half_width, plane_resolution)
    U, V = np.meshgrid(u, v)

    X = U * perp_hat[0] + V * g_normed[0]
    Y = U * perp_hat[1] + V * g_normed[1]
    Z = g[0] * X + g[1] * Y                    # z = g·(x,y)
    #Z[:]=-1

    # --- floating unit circle ---
    t = np.linspace(0.0, 2.0 * np.pi, circle_resolution)
    circ_x, circ_y = np.cos(t), np.sin(t)
    circ_z = np.full_like(circ_x, z_circle)

    # --- (uᵀg)u ellipse on plane ---
    u_vecs = np.column_stack((np.cos(t), np.sin(t)))  # (N,2)
    s = - u_vecs @ g #for descent direction   
    n_vecs=u_vecs.copy()
    ell_x = n_vecs[:,0] = s * u_vecs[:, 0]
    ell_y = n_vecs[:,1] = s * u_vecs[:, 1]
    ell_z = g[0] * ell_x + g[1] * ell_y
    
    nr=(np.sum(np.abs(s)))/circle_resolution
    print(nr,g_len,'avg d g',nr/g_len)

    # colour map: blue(0) → red(1)
    ratio = np.abs(s) / g_len
    colors = np.column_stack((ratio, np.zeros_like(ratio), 1.0 - ratio))

    # --- PyVista scene ---
    pl = pv.Plotter(window_size=window_dims)
    if camera_default is not None:
        restore_camera(pl,camera_default)
    #pl.camera_position=camera_default

    # unit circle
    circle_pts = np.column_stack((circ_x, circ_y, circ_z))
    pl.add_lines(circle_pts, color="black", width=4)
    pl.add_lines(np.vstack((circle_pts[-1], circle_pts[0])),
                 color="black", width=4)      # close circle

    # gradient arrow (scaled)
    arrow_end = -np.array([*g, g.dot(g)]) #* arrow_scale
    arrow = pv.Arrow(start=(0, 0, 0),
                     direction=arrow_end,
                     tip_length=0.15,
                     tip_radius=0.022,
                     shaft_radius=0.011,
                     scale='auto')
    pl.add_mesh(arrow, color="blue")
    sdg=sd.dot(g)*sd
    sarrow_end = - np.array([*sdg,np.sum(sdg*sdg)])
    sarrow = pv.Arrow(start=(0, 0, 0),
                     direction=sarrow_end,
                     tip_length=0.15,
                     tip_radius=0.022,
                     shaft_radius=0.011,
                     scale='auto')
    pl.add_mesh(sarrow, color="orange")
    

    # coloured ellipse segments
    for i in range(circle_resolution - 1):
        seg = np.array([[ell_x[i],   ell_y[i],   ell_z[i]],
                        [ell_x[i + 1], ell_y[i + 1], ell_z[i + 1]]])
        pl.add_lines(seg, color=tuple(colors[i]), width=5)

    #Calculate true relative norm, cosine, sine
    gd_norm=l2(sdg)
    cosr = sdg.dot(g) / (gd_norm*g_len) #cosine similarity
    magr=gd_norm/g_len #norm ratio = cosine similarity
    nrmse=l2(g-sdg)/g_len # normalized root mean sqr error, also sin(theta) when sdg is a directional derivative.
    sinr=mt.sqrt(1 - (sdg.dot(g) ** 2.) / (sdg.dot(sdg) * g.dot(g)))
    rmc=mt.sqrt(1-(nrmse**2)) #cosine similarity
    print(f"g to Dg: sin {sinr:.3f}, n-rmse {nrmse:.3f}, cos {cosr:.3f}, rel-norm {magr:.3f}, nrmse-cos {rmc:.3f}, ") #demo that calculated from the true gradient they are all the same.

    ### Creating tangent ellipse bounds for directional derivative (only meaningful if current directional derivative happens to be the true gradient).
    s = - u_vecs @ sdg
    ell_x = s * u_vecs[:, 0]
    ell_y = s * u_vecs[:, 1]
    ell_z = (sdg[0] * ell_x + sdg[1] * ell_y)

    # colour map: blue(0) → red(1)
    ratio = np.abs(s) / gd_norm
    colors = np.column_stack((ratio, np.zeros_like(ratio), 1.0 - ratio))
    # coloured ellipse segments
    for i in range(circle_resolution - 1):
        seg = np.array([[ell_x[i],   ell_y[i],   ell_z[i]],
                        [ell_x[i + 1], ell_y[i + 1], ell_z[i + 1]]])
        pl.add_lines(seg, color=colors[i], width=3)
    
    ### Plane Surface with green representing the valid surface area for directional derivatives.
    plane = pv.StructuredGrid(X, Y, Z)
    
    #if ver==1:
    #OLD METHOD
    # # Error Correcting Signal and boundary calculation:
    # X2d = np.column_stack([X.ravel(), Y.ravel()])  # .reshape(-1, 2)  # (n², 2)
    # dx = X2d + sdg  # vector (x - ĝ)
    # c=sinr #In case nrmse>1, using sin protects against that for this example.
    # lhs = l2(dx, axis=1) * np.sqrt(1 - c ** 2)
    # 
    # # sin(angle(x, ĝ)) using dot product formula, protects against nrmse>1.
    # norm_x = l2(X2d, axis=1)
    # sinphi = np.sqrt(1 - ((X2d @ sdg) ** 2) / ((norm_x ** 2.) * (gd_norm ** 2.)))
    # 
    # ecs = lhs / ((gd_norm) * (sinphi + c_mult * c))

    # new method, check v and u^T\hat{g} directly
    X2d = np.column_stack([X.ravel(), Y.ravel()])
    norm_x = l2(X2d, axis=1)
    dx = X2d + X2d * ((X2d @ sdg) / (norm_x ** 2)).reshape(norm_x.shape[0], 1)  # sdg #sdg  # vector (dux - duĝ)
    # v=v.reshape(v.shape[0])
    c = sinr  # In case nrmse>1, using sin protects against that for this example.
    css = 1 - c ** 2.
    cs = mt.sqrt(css)
    lhs = l2(dx, axis=1)  # * np.sqrt(1 - c ** 2)
    # As this is 2 dimensional it does not include z_sig, we set it to sqrt(d) so it cancels the dimensional term.
    ecs = lhs * cs / (gd_norm * c)

    feasible = (ecs <= 1.).reshape(X.shape[0], X.shape[1]).T.flatten()  # vtk with strange coordinates.
    
    colors = np.empty((X2d.shape[0], 3), np.uint8)
    colors[:]=[255, 0, 0] #red
    
    if ver==1:
        #new method, check v and u^T\hat{g} directly
        # c = sinr  # In case nrmse>1, using sin protects against that for this example.
        # css = 1 - c ** 2.
        # lhs = l2(dx, axis=1)  # * np.sqrt(1 - c ** 2)
        ecs = (lhs**2)*css / ((gd_norm * c)**2 + sigm**2)

        #feasible = (ecs <= 1.).reshape(X.shape[0], X.shape[1]).T.flatten()  # vtk with strange coordinates.
    elif ver==2:
        #old method, with hybrid direct check with sinphi relationship (no noise)
        dx = X2d + sdg  # vector (x - ĝ)
        c = sinr  # In case nrmse>1, using sin protects against that for this example.
        css = 1 - c ** 2.
        cs = mt.sqrt(css)
        lhs = l2(dx, axis=1)

        # sin(angle(x, ĝ)) using dot product formula, protects against nrmse>1.
        norm_x = l2(X2d, axis=1)
        sinphi = np.sqrt(1 - ((X2d @ sdg) ** 2) / ((norm_x ** 2.) * (gd_norm ** 2.)))

        ecs = lhs * cs / (gd_norm * np.sqrt(c ** 2 + (1 - c ** 2) * (sinphi ** 2)))
        
    #ecs=np.maximum(lhs/(s*c + a),ecs)
    
    feasible2 = (ecs <= 1.).reshape(X.shape[0], X.shape[1]).T.flatten()  # vtk with strange coordinates.

    #colors[feasible] = [0, 255, 0]  # green
    #colors[feasible2] = [0, 0, 255]  # blue
    colors[feasible2] = [0, 0, 255]  # blue
    colors[feasible] = [0, 255, 0]  # green
    plane['feasible_rgb'] = colors
    
    pl.add_mesh(plane, scalars='feasible_rgb', opacity=0.6, show_edges=False)


    pl.add_key_event("c", lambda: print_camera_position(pl))
    pl.add_key_event("s", lambda: save_screenshot(pl,pic_dir))
    pl.add_key_event("p", lambda: zoom_in(pl))
    pl.add_key_event("o", lambda: zoom_out(pl))

    # scene cosmetics
    pl.add_axes()
    pl.show_grid()
    #pl.camera.zoom('tight')
    pl.show(title=title)

#import numba as nb


#@nb.njit(fastmath=True, cache=True)
def cosine_range(v, u, ghat, sin_er, mmat):
    w=1-mt.sqrt(1-sin_er**2)
    # cgu=u.dot(ghat)/(l2(u)*l2(ghat))

    # 1) Compute alpha = cos(theta_[ghat, u])
    cts = 0
    #c = mt.sqrt(max(0.0, 1.0 - sin_er ** 2))
    for i in range(mmat.shape[0]):
        #us = (v[i]**2)*(u[i, 0] ** 2 + u[i, 1] ** 2)
        us =  (u[i, 0] ** 2 + u[i, 1] ** 2)
        gs = ghat[0] ** 2 + ghat[1] ** 2
        #npd = v[i]*u[i, 0] * ghat[0] + v[i]*u[i, 1] * ghat[1]
        npd = u[i, 0] * ghat[0] +  u[i, 1] * ghat[1]
        alpha = npd / ((us**.5) * (gs**.5)) #np.sign(v[i])*
        nalpha=1 - alpha
        # alpha = np.dot(ghat, u) /(l2(u)*l2(ghat))

        # 2) Compute sin(theta_[ghat,u]) and cos(delta)
        # s1 = mt.sqrt(max(0.0, 1.0 - alpha ** 2))
        # s2 = sin_er

        # 3) Endpoints of the cosine interval
        # upper = np.clip(alpha * c + s1 * s2, -1.0, 1.0)
        # lower = np.clip(alpha * c - s1 * s2, -1.0, 1.0)

        #phi0 = np.arccos(alpha)
        #delta = np.arccos(c)  # min(max(cos_c, -1.0), 1.0))

        #phi_upper = phi0 - delta  # np.clip(phi0 - delta, 0.0, np.pi)
        #phi_lower = phi0 + delta  # np.clip(phi0 + delta, 0.0, np.pi)
        #upper = np.cos(phi_upper)
        #lower = np.cos(phi_lower)

        # 4) Select bound based on sign of v
        #m = upper if (v[i] < 0) else lower
        if v[i]<0:
            if alpha<w: m=-1
            else: m=alpha-w
        elif alpha > w: m=1
        else: m = alpha + w
        cts += 1
        if cts % 1000 == 0:
            print('m low up', v[i], m)#, lower, upper)
        mmat[i] = m
    print('w',w,)

#legend making

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

def annotate_image_with_legend(
    input_path: str,
    output_name: str,
    nmse: str,
    cos_theta: str,
    nrmse: str,
    sin_theta: str,
    corner: str = "top-right",
    pad_frac: float = 0.02,
    fontsize: int = 18,       # increased default font size
    box_alpha: float = 1.0,   # irrelevant since facecolor=None (transparent)
    box_round: float = 0.0,   # 0 => square corners
    border_width: float = 1.5 # thicker border
) -> str:
    directory = os.path.dirname(input_path)
    root, in_ext = os.path.splitext(os.path.basename(input_path))
    out_root, out_ext = os.path.splitext(output_name)
    if not out_ext:
        output_name = output_name + in_ext
    output_path = os.path.join(directory, output_name)

    img = Image.open(input_path)
    width, height = img.size

    dpi = 100.0
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")

    # Positioning
    corner = corner.lower()
    x = 1 - pad_frac if "right" in corner else pad_frac
    y = 1 - pad_frac if "top" in corner else pad_frac
    ha_anchor = "right" if "right" in corner else "left"
    va = "top" if "top" in corner else "bottom"

    # Legend text (mathtext)
    text_lines = [
        r"$\mathbf{Gradient\ Estimator:}$",
        r"$\mathbf{\mathrm{N\!-\!MSE} = " + str(nmse) + r"}$",
        r"$\mathbf{\cos\theta = " + str(cos_theta) + r"}$",
        r"$\mathbf{\mathrm{N\!-\!RMSE} = " + str(nrmse) + r"}$",
        r"$\mathbf{\sin\theta = " + str(sin_theta) + r"}$",
    ]
    payload = "\n".join(text_lines)

    # Bbox: transparent face, thicker edge
    boxstyle = "square" if box_round <= 0 else f"round,pad=0.5,rounding_size={box_round*10}"
    bbox_props = dict(
        boxstyle=boxstyle,
        facecolor="none",               # transparent
        edgecolor="black",
        linewidth=border_width,
        alpha=1.0
    )

    ax.text(
        x, y, payload,
        transform=ax.transAxes,
        ha=ha_anchor, va=va,
        fontsize=fontsize,
        bbox=bbox_props,
        linespacing=1.2,
        multialignment="left"  # left-align multiline text within the box
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path

def make_legends():
    sample_paths = [
        "marks_ratio_png/screenshot_20250821_075816.png",
        # "/mnt/data/screenshot_20250821_084017.png",
        # "/mnt/data/screenshot_20250821_084201.png",
        # "/mnt/data/screenshot_20250821_084441.png",
    ]
    cos=['.1',]

    outputs = []
    for i, p in enumerate(sample_paths, 1):
        out = annotate_image_with_legend(
            p,
            f"cossim_{cos[i]}.png",
            nmse="0.012",
            cos_theta=cos[i],
            nrmse="0.11",
            sin_theta="0.37",
            corner="top-right",
            pad_frac=0.02,
            fontsize=20,  # even larger for visibility
            box_round=0.0,  # square corners
            border_width=2.0  # thicker border
        )
        outputs.append(out)

    outputs
    



if __name__ == "__main__":
    #draw_gradient_ellipse_scene()
    #.205 cossim ~.25, .65 cossim ~ .9, .9 cossim ~ .995
    gnorm=1.0735455276791943
    grad=np.array([.5, .95])
    default_cam=[(-1.3998497570807962, -25.143898563190163, 3.2175412828844783),
 (0.018648518877575793, -0.06660733085192488, -0.5457226230420016),
 (0.0004612495587092754, 0.14837933684788424, 0.9889304119327237)] #old for cossim .1
    default_cam=((-1.2247893566982606, -10.255481205158292, 1.2358633463365274), (0.10853352722868848, -0.0545392082465741, -0.3432423194748957), (0.006058641817432512, 0.15220170561046006, 0.98833088268484), (4.346776589598497, 18.035054960434366), 30.006000900120018, 3.94405016072083, False)
 #for the rest.
    #.3028**2 cossim .1 , rmse .995 #CAMERA: [(-0.6712659442703749, -25.387861739988146, -0.2251809035756731),(0,0,0),(0,0,0)]
    #.605**2 cossim .5 rmse .866
    #.821**2 cossim .9 roughly rmse .435
    #.907 cossim .995 rmse .102
    dir=os.path.join(os.getcwd(),'marks_ratio_png')
    #wd=(1150,900)
    wd=(1270,1000)
    plot_ecg_scene(sample_dir=.907,gradient=grad,plane_half_width=1.8,plane_resolution=2000,circle_resolution=400,sigm=gnorm*.25,ver=1,pic_dir=dir,window_dims=wd,camera_default=default_cam)
    #plot_feasibility_boundary(mode='cartesian',n=2000,k=1)
