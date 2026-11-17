"""Partie 1 : reconnaissance faciale par ACP (Eigenfaces) sur la base Olivetti."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from common import OUT, load_olivetti

KS = [10, 30, 50, 100, 150]


def fit_pca(X, k):
    X_mean = X.mean(axis=0)
    pca = PCA(n_components=k, random_state=42)
    X_pca = pca.fit_transform(X - X_mean)
    return pca, X_pca, X_mean


def plot_eigenfaces(pca, X_mean, n=15):
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    axes = axes.ravel()
    axes[0].imshow(X_mean.reshape(64, 64), cmap="gray")
    axes[0].set_title("Visage moyen", fontsize=8)
    for i in range(n):
        axes[i + 1].imshow(pca.components_[i].reshape(64, 64), cmap="RdBu_r")
        axes[i + 1].set_title(f"EF {i + 1}", fontsize=8)
    for a in axes:
        a.axis("off")
    fig.suptitle(f"Eigenfaces (k={pca.n_components_})")
    fig.savefig(OUT / "eigenfaces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance(X, kmax=200):
    pca = PCA(n_components=kmax, random_state=42).fit(X - X.mean(axis=0))
    var = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(var)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(1, kmax + 1), var)
    ax[0].set(title="Variance par composante", xlabel="Composantes", ylabel="%")
    ax[1].plot(range(1, kmax + 1), cum)
    for s in (90, 95):
        ax[1].axhline(s, ls="--", color="gray")
    for k in KS:
        ax[1].axvline(k, ls=":", color="gray", lw=0.7)
    ax[1].set(title="Variance cumulée", xlabel="Composantes", ylabel="%")
    fig.savefig(OUT / "variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cum


def nearest_neighbors(X_pca, query_idx, n=10):
    dists = np.linalg.norm(X_pca - X_pca[query_idx], axis=1)
    dists[query_idx] = np.inf  # exclure l'image elle-même
    idx = np.argsort(dists)[:n]
    return idx, dists[idx]


def plot_recognition(images, X, k, query_idx=5):
    pca, X_pca, _ = fit_pca(X, k)
    idx, d = nearest_neighbors(X_pca, query_idx)
    var = pca.explained_variance_ratio_.sum() * 100
    fig, axes = plt.subplots(1, 11, figsize=(15, 2.5))
    axes[0].imshow(images[query_idx], cmap="gray")
    axes[0].set_title("Requête", fontsize=8)
    for j, (i, dist) in enumerate(zip(idx, d), start=1):
        axes[j].imshow(images[i], cmap="gray")
        axes[j].set_title(f"#{j}\nd={dist:.1f}", fontsize=7)
    for a in axes:
        a.axis("off")
    fig.suptitle(f"Reconnaissance ACP, k={k} (variance expliquée : {var:.1f}%)")
    fig.savefig(OUT / f"recognition_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction(images, X, ids=(5, 10)):
    fig, axes = plt.subplots(len(ids), len(KS) + 1, figsize=(12, 4))
    for r, i in enumerate(ids):
        axes[r, 0].imshow(images[i], cmap="gray")
        axes[r, 0].set_title("Original", fontsize=8)
    for c, k in enumerate(KS, start=1):
        pca, X_pca, X_mean = fit_pca(X, k)
        X_rec = pca.inverse_transform(X_pca) + X_mean
        var = pca.explained_variance_ratio_.sum() * 100
        for r, i in enumerate(ids):
            axes[r, c].imshow(X_rec[i].reshape(64, 64), cmap="gray")
            if r == 0:
                axes[r, c].set_title(f"k={k}\n({var:.0f}%)", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.savefig(OUT / "reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def loo_accuracy(X, labels, k):
    """Précision 1-plus-proche-voisin en leave-one-out (mesure quantitative)."""
    _, X_pca, _ = fit_pca(X, k)
    ok = 0
    for i in range(len(X)):
        idx, _ = nearest_neighbors(X_pca, i, n=1)
        ok += labels[idx[0]] == labels[i]
    return ok / len(X)


if __name__ == "__main__":
    images, X, labels = load_olivetti()
    pca, _, X_mean = fit_pca(X, 50)
    plot_eigenfaces(pca, X_mean)
    cum = plot_variance(X)
    for k in KS:
        print(f"k={k:>3} | variance cumulée = {cum[k - 1]:.1f}% | "
              f"précision 1-NN (LOO) = {loo_accuracy(X, labels, k) * 100:.1f}%")
    for k in (10, 50, 150):
        plot_recognition(images, X, k)
    plot_reconstruction(images, X)
    print(f"Figures enregistrées dans {OUT}/")
