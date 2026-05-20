'''
Setup of this simulation:
In-control distributions: 𝜇=N(m,Σ)
Out-of-control distributions:  =N(m+Δ,Σ) (pure translation)

Dimensions: d = 1, d=2
Covariance: \Sigma = sigma^2I_d

Set N = 500
'''

import numpy as np
from pathlib import Path
from typing import Tuple



# -------------- Utilities ---------------

def generate_gaussian_phaseI_phaseII(
    out_dir: Path,
    op_name: str = "gauss_shift",
    d: int = 2,
    n_points: int = 100,
    n_phaseI: int = 100,
    n_phaseII_ic: int = 200,
    n_phaseII_oc: int = 200,
    sigma: float = 1.0,
    delta: np.ndarray | None = None,
    random_state: int | None = 123,
) -> Tuple[Path, Path, Path]:
    """
    Generate Gaussian in-control and mean-shifted out-of-control distribution-valued
    samples and save them as NPZ files compatible with `process_phaseI`.

    Each distribution-valued sample is represented as an (n_points, d) array of
    points drawn from N(mean, Sigma), with Sigma = sigma^2 * I_d.

    Phase I NPZ:
        - 'phaseI'     : array/list of length n_phaseI, each (n_points, d)
        - 'map_params' : list of None (placeholder)
        - 'n_points'   : integer n_points

    Phase II NPZ:
        - 'phaseII'    : array/list of length (n_phaseII_ic + n_phaseII_oc)
                         first n_phaseII_ic are in-control, rest are shifted
        - 'map_params' : list of None (placeholder)
        - 'n_points'   : integer n_points
    """
    rng = np.random.default_rng(random_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- parameters ---------------------------------------------------------
    # covariance: isotropic Gaussian Sigma = sigma^2 I_d
    Sigma = (sigma ** 2) * np.eye(d)

    # base mean and shift direction
    base_mean = np.zeros(d)
    if delta is None:
        # default: shift along first coordinate
        delta = np.zeros(d)
        delta[0] = 0.5  # you can tune this
    delta = np.asarray(delta, dtype=float)

    oc_mean = base_mean + delta

    # --- Phase I: all in-control -------------------------------------------
    phaseI_clouds = []
    for i in range(n_phaseI):
        # each cloud: (n_points, d) from N(base_mean, Sigma)
        X = rng.multivariate_normal(mean=base_mean, cov=Sigma, size=n_points)
        phaseI_clouds.append(X)

    phaseI_arr = np.array(phaseI_clouds, dtype=object)  # compatible with allow_pickle=True
    map_params_I = np.array([None] * n_phaseI, dtype=object)

    phaseI_file = out_dir / f"{op_name}_phaseI.npz"
    np.savez(
        phaseI_file,
        phaseI=phaseI_arr,
        map_params=map_params_I,
        n_points=n_points,
    )

    # --- Phase II: in-control segment + out-of-control segment -------------
    phaseII_clouds = []

    # Phase II in-control (for ARL0)
    for t in range(n_phaseII_ic):
        X = rng.multivariate_normal(mean=base_mean, cov=Sigma, size=n_points)
        phaseII_clouds.append(X)

    phaseII_arr = np.array(phaseII_clouds, dtype=object)
    map_params_II = np.array([None] * len(phaseII_clouds), dtype=object)

    phaseII_file_ic = out_dir / f"{op_name}_phaseII_ic.npz"
    np.savez(
        phaseII_file_ic,
        streams=phaseII_arr,
        map_params=map_params_II,
        n_points=n_points,
    )

    # Phase II out-of-control (mean shift) for ARL1
    phaseII_clouds = []
    for t in range(n_phaseII_oc):
        X = rng.multivariate_normal(mean=oc_mean, cov=Sigma, size=n_points)
        phaseII_clouds.append(X)

    phaseII_arr = np.array(phaseII_clouds, dtype=object)
    map_params_II = np.array([None] * len(phaseII_clouds), dtype=object)

    phaseII_file_oc = out_dir / f"{op_name}_phaseII_oc.npz"
    np.savez(
        phaseII_file_oc,
        streams=phaseII_arr,
        map_params=map_params_II,
        n_points=n_points,
    )

    return phaseI_file, phaseII_file_ic, phaseII_file_oc


def recenter_phaseII_ic_to_phaseI_mean(phaseI_file, phaseII_ic_file):
    """Shift PhaseII IC so its global mean matches Phase I global mean."""
    zI = np.load(phaseI_file, allow_pickle=True)
    zIC = np.load(phaseII_ic_file, allow_pickle=True)

    phaseI_clouds = list(zI["phaseI"])
    streams_ic = list(zIC["streams"])

    all_I  = np.vstack(phaseI_clouds)
    all_ic = np.vstack(streams_ic)

    mean_I  = all_I.mean(axis=0)
    mean_ic = all_ic.mean(axis=0)
    shift   = mean_I - mean_ic

    print("Phase I mean :", mean_I)
    print("Phase II IC mean (before):", mean_ic)
    print("Shift to apply:", shift)

    # apply same shift to every IC cloud
    streams_ic_shifted = [cloud + shift for cloud in streams_ic]

    # overwrite / resave PhaseII_ic file
    streams_ic_arr = np.array(streams_ic_shifted, dtype=object)
    np.savez(
        phaseII_ic_file,
        streams=streams_ic_arr,
        map_params=zIC["map_params"],
        n_points=zIC["n_points"],
    )

    # quick check
    all_ic_new = np.vstack(streams_ic_shifted)
    print("Phase II IC mean (after):", all_ic_new.mean(axis=0))


#%% Utilities to check the simualted data distribution

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _load_cloud_list(npz_path: Path, key: str):
    """Load list of clouds from a .npz (object array)."""
    z = np.load(npz_path, allow_pickle=True)
    arr = z[key]
    # typically saved as object array: shape (n_clouds,), dtype=object
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(a) for a in arr]
    # fallback if saved differently
    return [np.asarray(a) for a in arr]


def visualize_gaussian_simulation(
    phaseI_file: Path,
    phaseII_file_ic: Path,
    phaseII_file_oc: Path,
    n_show_phaseI: int = 20,
    n_show_phaseII: int = 20,
    random_state: int = 0,
):
    """
    Visualize 2D Gaussian simulation:
      - Phase I clouds (IC)
      - Phase II IC clouds
      - Phase II OC clouds overlaid on Phase I

    Parameters
    ----------
    phaseI_file : Path
        .npz with key 'phaseI'.
    phaseII_file_ic : Path
        .npz with key 'streams' (all in-control).
    phaseII_file_oc : Path
        .npz with key 'streams' (all out-of-control).
    n_show_phaseI : int
        Number of Phase I clouds to subsample for plotting.
    n_show_phaseII : int
        Number of Phase II clouds (IC/OC) to subsample for plotting.
    random_state : int
        Seed for reproducible subsampling.
    """
    phaseI_clouds   = _load_cloud_list(phaseI_file, "phaseI")
    phaseII_ic_clouds = _load_cloud_list(phaseII_file_ic, "streams")
    phaseII_oc_clouds = _load_cloud_list(phaseII_file_oc, "streams")

    rng = np.random.default_rng(random_state)

    # choose which clouds to show
    idx_I  = rng.choice(len(phaseI_clouds),
                        size=min(n_show_phaseI, len(phaseI_clouds)),
                        replace=False)
    idx_ic = rng.choice(len(phaseII_ic_clouds),
                        size=min(n_show_phaseII, len(phaseII_ic_clouds)),
                        replace=False)
    idx_oc = rng.choice(len(phaseII_oc_clouds),
                        size=min(n_show_phaseII, len(phaseII_oc_clouds)),
                        replace=False)

    X_I  = np.vstack([phaseI_clouds[i] for i in idx_I])
    X_ic = np.vstack([phaseII_ic_clouds[i] for i in idx_ic])
    X_oc = np.vstack([phaseII_oc_clouds[i] for i in idx_oc])

    assert X_I.shape[1] == 2, "This visualization is only implemented for 2D data."

    all_I = np.vstack(phaseI_clouds)
    all_ic = np.vstack(phaseII_ic_clouds)
    all_oc = np.vstack(phaseII_oc_clouds)

    print("Average means over ALL points:")
    print("  Phase I     :", all_I.mean(axis=0))
    print("  Phase II IC :", all_ic.mean(axis=0))
    print("  Phase II OC :", all_oc.mean(axis=0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    # Phase I only
    axes[0].scatter(X_I[:, 0], X_I[:, 1], s=5, alpha=0.5)
    axes[0].set_title("Phase I (IC)")

    # Phase II IC only
    axes[1].scatter(X_I[:, 0], X_I[:, 1], s=5, alpha=0.15, label="Phase I")
    axes[1].scatter(X_ic[:, 0], X_ic[:, 1], s=5, alpha=0.5, label="Phase II (IC)")
    axes[1].legend()
    axes[1].set_title("Phase I vs Phase II (IC)")

    # Overlay Phase I (faded) + Phase II OC
    axes[2].scatter(X_I[:, 0], X_I[:, 1], s=5, alpha=0.15, label="Phase I")
    axes[2].scatter(X_oc[:, 0], X_oc[:, 1], s=5, alpha=0.5, label="Phase II (OC)")
    axes[2].legend()
    axes[2].set_title("Phase I vs Phase II (OC)")

    for ax in axes:
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")

    fig.suptitle("Gaussian simulation: Phase I and Phase II (IC/OC)")
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    from pathlib import Path

    # data_save_path = r'C:\Users\zengyy\Research\DIDO_CC\Simulation\kde_comp'
    # out_dir = Path(data_save_path)
    # DIM = 2

    # # first test without replications
    #
    # phaseI_file, phaseII_file_ic, phaseII_file_oc = generate_gaussian_phaseI_phaseII(
    #     out_dir=out_dir,
    #     op_name="gauss_shift_d{}".format(DIM),
    #     d=DIM,
    #     n_points=200,
    #     n_phaseI=100,
    #     n_phaseII_ic=200,
    #     n_phaseII_oc=200,
    #     sigma=1.0,
    #     delta=np.array([0.5, 0.0]),  # shift along x1
    #     random_state=42,
    # )
    #
    # recenter_phaseII_ic_to_phaseI_mean(phaseI_file, phaseII_file_ic)
    #
    # visualize_gaussian_simulation(
    #     phaseI_file,
    #     phaseII_file_ic,
    #     phaseII_file_oc,
    #     n_show_phaseI=20,
    #     n_show_phaseII=20,
    #     random_state=0,
    # )

    import os, argparse
    parser = argparse.ArgumentParser(description="Generate Gaussian translation simulation data.")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else Path(
        os.environ.get("DIDO_DATA_ROOT", Path(__file__).parent.parent.parent / "data")
    ) / "gaussian_translation"
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = [1, 2, 5]

    # ----- simulation design -----
    n_rep       = 10          # replications per (sigma, delta)
    n_points    = 100         # particles per distribution
    n_phaseI    = 200
    n_phaseII_ic = 200
    n_phaseII_oc = 200

    sigmas  = [0.5, 1.0, 2.0]   # edit as you like
    deltas1 = [0.1, 0.5]   # shifts along x1 (delta = (delta1, 0))
    base_seed = 42

    for d in dims:
        for sigma in sigmas:
            for delta1 in deltas1:
                # delta vector in R^d: shift along all the dimensions
                delta_vec = np.zeros(d)
                for ddd in range(d):
                    delta_vec[ddd] = delta1

                # one folder per (d, sigma, delta)
                cfg_tag = f"gauss_shift_d{d}_sig{sigma:.2f}_del{delta1:.2f}"
                cfg_tag = cfg_tag.replace(".", "p")  # e.g. gauss_shift_d2_sig1p00_del0p50
                cfg_dir = out_dir / cfg_tag
                cfg_dir.mkdir(parents=True, exist_ok=True)

                for rep in range(n_rep):
                    # op_name gets a replicate suffix, but still saved inside cfg_dir
                    op_name = f"{cfg_tag}_rep{rep:02d}"

                    seed = (base_seed
                            + 1000 * d
                            + 100 * int(round(10 * sigma))
                            + 10 * int(round(10 * delta1))
                            + rep)

                    phaseI_file, phaseII_file_ic, phaseII_file_oc = generate_gaussian_phaseI_phaseII(
                        out_dir=cfg_dir,  # <-- same folder for same config
                        op_name=op_name,
                        d=d,
                        n_points=n_points,
                        n_phaseI=n_phaseI,
                        n_phaseII_ic=n_phaseII_ic,
                        n_phaseII_oc=n_phaseII_oc,
                        sigma=sigma,
                        delta=delta_vec,
                        random_state=seed,
                    )

                    recenter_phaseII_ic_to_phaseI_mean(phaseI_file, phaseII_file_ic)

                    if d == 2 and rep == 0:
                        visualize_gaussian_simulation(
                            phaseI_file,
                            phaseII_file_ic,
                            phaseII_file_oc,
                            n_show_phaseI=20,
                            n_show_phaseII=20,
                            random_state=0,
                        )

                    print(f"Generated: d={d}, sigma={sigma}, delta1={delta1}, rep={rep:02d}, dir={cfg_dir}")
