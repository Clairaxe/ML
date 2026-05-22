import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from shutil import make_archive
from matplotlib.patches import Rectangle

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FIG_DIR = Path("figures_tme5")
FIG_DIR.mkdir(exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")

def print_section(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))

table_poi = pd.read_csv("data/poi-paris.csv")
districts = pd.read_csv("data/districts-paris.csv", sep=",")
districts["bounds"] = districts["bounds"].apply(eval)

def draw_districts():
    for x in districts["bounds"]:
        plt.plot([c[0] for c in x], [c[1] for c in x], color="black", linewidth=0.8)

type_cols = [
    "furniture_store", "laundry", "bakery", "cafe", "home_goods_store",
    "clothing_store", "atm", "lodging", "night_club", "convenience_store",
    "restaurant", "bar"
]

tab20 = plt.get_cmap("tab20")

print_section("Données")
print("Nombre total de POIs :", len(table_poi))
print("Nombre de types utilisés :", len(type_cols))
print("Types utilisés :", ", ".join(type_cols))

type_means = table_poi[type_cols].mean().sort_values(ascending=False)
print()
print("Types les plus fréquents :")
for name, val in type_means.head(8).items():
    print(f"  {name} : {val:.3f}")


# FIGURE 1 : POIS

plt.figure(figsize=(7, 6))
draw_districts()
plt.scatter(table_poi["longitude"], table_poi["latitude"], s=1, alpha=0.5)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Points d'intérêt à Paris")
savefig("01_poi_paris.png")
plt.show()


# DISCRÉTISATION EN GRILLE


N = 20
K_GRID = 6

lomin, lomax = table_poi["longitude"].min(), table_poi["longitude"].max()
lamin, lamax = table_poi["latitude"].min(), table_poi["latitude"].max()

def coord_to_cell(longitude, latitude, N):
    i = np.floor(N * (longitude - lomin) / (lomax - lomin)).astype(int)
    j = np.floor(N * (latitude - lamin) / (lamax - lamin)).astype(int)

    i = np.clip(i, 0, N - 1)
    j = np.clip(j, 0, N - 1)

    return i, j

nb_types = len(type_cols)

descriptions = np.zeros((N, N, nb_types))
nb_pois_cellule = np.zeros((N, N))

cell_i, cell_j = coord_to_cell(
    table_poi["longitude"].values,
    table_poi["latitude"].values,
    N
)

for p in range(len(table_poi)):
    i = cell_i[p]
    j = cell_j[p]
    vect_type = table_poi.iloc[p][type_cols].astype(float).values

    descriptions[i, j, :] += vect_type
    nb_pois_cellule[i, j] += 1

for i in range(N):
    for j in range(N):
        if nb_pois_cellule[i, j] > 0:
            descriptions[i, j, :] /= nb_pois_cellule[i, j]

X_grid = descriptions.reshape(N * N, nb_types)
mask_grid = nb_pois_cellule.reshape(-1) > 0

km_grid = KMeans(n_clusters=K_GRID, random_state=0, n_init=10)
labels_non_vides = km_grid.fit_predict(X_grid[mask_grid])

labels_grid_flat = -np.ones(N * N, dtype=int)
labels_grid_flat[mask_grid] = labels_non_vides
labels_grid = labels_grid_flat.reshape(N, N)

def get_clust(i, j):
    if labels_grid[i, j] == -1:
        return -1
    return labels_grid[i, j]

print_section("Discrétisation en grille")
print("Taille de la grille :", f"{N} x {N}")
print("Nombre total de cellules :", N * N)
print("Nombre de cellules non vides :", int(mask_grid.sum()))
print("Proportion de cellules non vides :", round(mask_grid.mean(), 3))
print("K retenu pour la grille :", K_GRID)
print("Inertie KMeans grille :", round(km_grid.inertia_, 4))


# FIGURE 2 : CLUSTERING GRILLE

plt.figure(figsize=(7, 6))
draw_districts()

for i in range(N):
    for j in range(N):
        label = get_clust(i, j)

        if label != -1:
            x = lomin + (lomax - lomin) * i / N
            y = lamin + (lamax - lamin) * j / N
            width = (lomax - lomin) / N
            height = (lamax - lamin) / N

            plt.gca().add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    color=tab20.colors[label % 20],
                    alpha=0.5
                )
            )

plt.scatter(table_poi["longitude"], table_poi["latitude"], s=0.1, color="black", alpha=0.2)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Clustering des cellules de la grille")
savefig("02_clustering_grille.png")
plt.show()


# FIGURE 3 : CENTROÏDES GRILLE

plt.figure(figsize=(10, 5))
plt.imshow(km_grid.cluster_centers_, aspect="auto")
plt.colorbar(label="proportion moyenne")
plt.xticks(np.arange(len(type_cols)), type_cols, rotation=90)
plt.yticks(np.arange(K_GRID), [f"cluster {k}" for k in range(K_GRID)])
plt.title("Centroïdes des clusters : grille")
savefig("03_centroides_grille.png")
plt.show()

print()
print("Types dominants des clusters de grille :")
for k, center in enumerate(km_grid.cluster_centers_):
    order = np.argsort(center)[::-1][:3]
    top = ", ".join([f"{type_cols[i]} ({center[i]:.3f})" for i in order])
    print(f"  cluster {k} : {top}")


# FIGURE 4 : ELBOW GRILLE

K_values_grid = range(1, 15)
inertias_grid = []

for K in K_values_grid:
    km_tmp = KMeans(n_clusters=K, random_state=0, n_init=10)
    km_tmp.fit(X_grid[mask_grid])
    inertias_grid.append(km_tmp.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(K_values_grid), inertias_grid, marker="o")
plt.xlabel("Nombre de clusters K")
plt.ylabel("Inertie")
plt.title("Courbe elbow : grille")
plt.grid()
savefig("04_elbow_grille.png")
plt.show()

print_section("Elbow grille")
for K, inertia in zip(K_values_grid, inertias_grid):
    print(f"K={K:2d} | inertie={inertia:.4f}")

print("Suggestion pour le rapport : le coude semble raisonnable autour de K = 5 ou K = 6.")


# QUANTIZATION SPATIALE

K_GEO = 100
coords = table_poi[["longitude", "latitude"]].values

km_spatial = KMeans(n_clusters=K_GEO, random_state=0, n_init=10)
labels_spatial = km_spatial.fit_predict(coords)

def get_cluster_spatial(i):
    return labels_spatial[i]

print_section("Quantization spatiale")
print("Nombre de régions spatiales :", K_GEO)
print("Inertie spatiale :", round(km_spatial.inertia_, 6))


# FIGURE 5 : QUANTIZATION SPATIALE

plt.figure(figsize=(7, 6))
draw_districts()

plt.scatter(
    table_poi["longitude"],
    table_poi["latitude"],
    color=[tab20.colors[get_cluster_spatial(i) % 20] for i in range(len(table_poi))],
    s=0.5
)

for i in range(K_GEO):
    plt.plot(
        km_spatial.cluster_centers_[i, 0],
        km_spatial.cluster_centers_[i, 1],
        marker="*",
        color="black",
        markersize=4
    )

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Quantization spatiale des POIs")
savefig("05_quantization_spatiale.png")
plt.show()


# DESCRIPTION DES RÉGIONS SPATIALES

region_desc = np.zeros((K_GEO, nb_types))
region_counts = np.zeros(K_GEO)

for p in range(len(table_poi)):
    k = labels_spatial[p]
    vect_type = table_poi.iloc[p][type_cols].astype(float).values

    region_desc[k] += vect_type
    region_counts[k] += 1

for k in range(K_GEO):
    if region_counts[k] > 0:
        region_desc[k] /= region_counts[k]

K_REGIONS = 6

km_regions = KMeans(n_clusters=K_REGIONS, random_state=0, n_init=10)
labels_regions = km_regions.fit_predict(region_desc)

print_section("Clustering des régions spatiales")
print("K retenu pour les régions spatiales :", K_REGIONS)
print("Inertie KMeans régions spatiales :", round(km_regions.inertia_, 4))
print("Taille min des régions spatiales :", int(region_counts.min()))
print("Taille max des régions spatiales :", int(region_counts.max()))
print("Taille moyenne des régions spatiales :", round(region_counts.mean(), 2))


# FIGURE 6 : CLUSTERING RÉGIONS SPATIALES

plt.figure(figsize=(7, 6))
draw_districts()

colors_regions = [
    tab20.colors[labels_regions[labels_spatial[i]] % 20]
    for i in range(len(table_poi))
]

plt.scatter(
    table_poi["longitude"],
    table_poi["latitude"],
    color=colors_regions,
    s=0.5
)

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Clustering des régions spatiales selon les types de POIs")
savefig("06_clustering_regions_spatiales.png")
plt.show()


# FIGURE 7 : CENTROÏDES RÉGIONS SPATIALES

plt.figure(figsize=(10, 5))
plt.imshow(km_regions.cluster_centers_, aspect="auto")
plt.colorbar(label="proportion moyenne")
plt.xticks(np.arange(len(type_cols)), type_cols, rotation=90)
plt.yticks(np.arange(K_REGIONS), [f"cluster {k}" for k in range(K_REGIONS)])
plt.title("Centroïdes des clusters : régions spatiales")
savefig("07_centroides_regions_spatiales.png")
plt.show()

print()
print("Types dominants des clusters de régions spatiales :")
for k, center in enumerate(km_regions.cluster_centers_):
    order = np.argsort(center)[::-1][:3]
    top = ", ".join([f"{type_cols[i]} ({center[i]:.3f})" for i in order])
    print(f"  cluster {k} : {top}")


# FIGURE 8 : ELBOW RÉGIONS SPATIALES

K_values_regions = range(1, 15)
inertias_regions = []

for K in K_values_regions:
    km_tmp = KMeans(n_clusters=K, random_state=0, n_init=10)
    km_tmp.fit(region_desc)
    inertias_regions.append(km_tmp.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(K_values_regions), inertias_regions, marker="o")
plt.xlabel("Nombre de clusters K")
plt.ylabel("Inertie")
plt.title("Courbe elbow : régions spatiales")
plt.grid()
savefig("08_elbow_regions_spatiales.png")
plt.show()

print_section("Elbow régions spatiales")
for K, inertia in zip(K_values_regions, inertias_regions):
    print(f"K={K:2d} | inertie={inertia:.4f}")

print("Suggestion pour le rapport : le coude semble raisonnable autour de K = 5 ou K = 6.")


# FIGURE 9 : CORRÉLATIONS ENTRE TYPES

corr_df = pd.DataFrame(region_desc, columns=type_cols).corr()

plt.figure(figsize=(8, 7))
plt.imshow(corr_df, vmin=-1, vmax=1)
plt.colorbar(label="corrélation")
plt.xticks(np.arange(len(type_cols)), type_cols, rotation=90)
plt.yticks(np.arange(len(type_cols)), type_cols)
plt.title("Corrélation entre types de POIs")
savefig("09_correlation_types.png")
plt.show()

print_section("Corrélations entre types")
corr_pairs = []

for i in range(len(type_cols)):
    for j in range(i + 1, len(type_cols)):
        corr_pairs.append((type_cols[i], type_cols[j], corr_df.iloc[i, j]))

corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)

print("Corrélations les plus fortes en valeur absolue :")
for a, b, c in corr_pairs[:8]:
    print(f"  {a} / {b} : {c:.3f}")


# FIGURE 10 : STANDARDISATION

scaler = StandardScaler()
region_desc_std = scaler.fit_transform(region_desc)

km_std = KMeans(n_clusters=K_REGIONS, random_state=0, n_init=10)
labels_regions_std = km_std.fit_predict(region_desc_std)

plt.figure(figsize=(7, 6))
draw_districts()

colors_std = [
    tab20.colors[labels_regions_std[labels_spatial[i]] % 20]
    for i in range(len(table_poi))
]

plt.scatter(
    table_poi["longitude"],
    table_poi["latitude"],
    color=colors_std,
    s=0.5
)

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Clustering après standardisation")
savefig("10_clustering_standardise.png")
plt.show()

print_section("Standardisation")
print("Inertie avant standardisation :", round(km_regions.inertia_, 4))
print("Inertie après standardisation :", round(km_std.inertia_, 4))
print("K utilisé :", K_REGIONS)
print("Remarque : les inerties ne sont pas directement comparables car les échelles ont changé.")
