"""Partie 2 : compression d'images par SVD (N&B, niveaux de gris, couleur)."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from common import OUT, load_olivetti

KS = [1, 5, 10, 20, 50, 100, 150]
THRESHOLD = 0.45  # seuil de binarisation de l'image N&B


def svd_compress(img_2d, k):
    """Compression SVD d'une image 2D (niveaux de gris)."""
    U, S, Vt = np.linalg.svd(img_2d, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def compress_rgb(img_rgb, k):
    """Compression SVD canal par canal pour image couleur."""
    return np.stack([np.clip(svd_compress(img_rgb[:, :, c], k), 0, 1)
                     for c in range(3)], axis=2)


def compression_ratio(m, n, k):
    return (m * n) / (k * (m + n + 1))


def psnr(orig, rec):
    mse = np.mean((orig - rec) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 1e-10 else float("inf")


def resize(img, size=256):
    pil = Image.fromarray((img * 255).astype(np.uint8))
    return np.asarray(pil.resize((size, size), Image.BICUBIC)) / 255.0


def build_images(images):
    gray = resize(images[0])
    bw = (gray > THRESHOLD).astype(float)
    # sujets 1, 6 et 11 (indices 0, 5, 10) sur les canaux R, G, B
    color = np.stack([resize(images[i]) for i in (0, 5, 10)], axis=2)
    return {"Noir & Blanc": bw, "Niveaux de gris": gray, "Couleur": color}


def run(name, img):
    m, n = img.shape[:2]
    rows = []
    fig, axes = plt.subplots(1, len(KS) + 1, figsize=(16, 2.8))
    axes[0].imshow(img, cmap=None if img.ndim == 3 else "gray")
    axes[0].set_title("Original", fontsize=8)
    for a, k in zip(axes[1:], KS):
        rec = compress_rgb(img, k) if img.ndim == 3 else np.clip(svd_compress(img, k), 0, 1)
        p, cr = psnr(img, rec), compression_ratio(m, n, k)
        rows.append((k, cr, p))
        a.imshow(rec, cmap=None if img.ndim == 3 else "gray")
        a.set_title(f"k={k}\nCR={cr:.1f}x\nPSNR={p:.1f} dB", fontsize=7)
    for a in axes:
        a.axis("off")
    fig.suptitle(f"Compression SVD — {name}")
    fig.savefig(OUT / f"svd_{name.replace(' & ', '_').replace(' ', '_').lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_singular_values(imgs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, img) in zip(axes, imgs.items()):
        ch = img if img.ndim == 2 else img[:, :, 0]
        S = np.linalg.svd(ch, compute_uv=False)[:100]
        ax.semilogy(S + 1e-16)
        ax.set(title=name, xlabel="Rang", ylabel="Valeur singulière")
    fig.savefig(OUT / "singular_values.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psnr(results):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in results.items():
        ks = [r[0] for r in rows]
        ps = [r[2] if np.isfinite(r[2]) else np.nan for r in rows]  # inf = reconstruction parfaite
        ax.plot(ks, ps, marker="o", label=name)
    ax.axhline(30, ls="--", color="gray")
    ax.axhline(40, ls="--", color="gray")
    ax.set(xlabel="k", ylabel="PSNR (dB)", title="PSNR vs k (inf non tracé)")
    ax.legend()
    fig.savefig(OUT / "psnr_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    images, _, _ = load_olivetti()
    imgs = build_images(images)
    results = {name: run(name, img) for name, img in imgs.items()}
    plot_singular_values(imgs)
    plot_psnr(results)
    print(f"{'k':>4} {'CR':>8} " + " ".join(f"{n:>16}" for n in results))
    for i, k in enumerate(KS):
        cr = results["Couleur"][i][1]
        print(f"{k:>4} {cr:>7.2f}x " + " ".join(f"{results[n][i][2]:>13.1f} dB" for n in results))
