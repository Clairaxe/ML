import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")


POI_FILENAME = "data/poi-paris.pkl"
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

xmin, xmax = 2.23, 2.48
ymin, ymax = 48.806, 48.916
coords = [xmin, xmax, ymin, ymax]


class Density(object):
    def fit(self,data):
        pass
    def predict(self,data):
        pass
    def score(self,data):
        #A compléter : retourne la log-vraisemblance
        return np.sum(np.log(self.predict(data) + 1e-10))

class Histogramme(Density):
    def __init__(self,steps=10):
        Density.__init__(self)
        self.steps = steps
    def fit(self,x):
        #A compléter : apprend l'histogramme de la densité sur x
        self.xmin = x.min(axis=0)
        self.xmax = x.max(axis=0)

        hist, edges = np.histogramdd(
            x,
            bins=self.steps,
            range=[[self.xmin[0], self.xmax[0]],
                   [self.xmin[1], self.xmax[1]]]
        )

        self.hist = hist
        self.edges = edges
        self.n = x.shape[0]

        self.bin_widths = [
            edges[0][1] - edges[0][0],
            edges[1][1] - edges[1][0]
        ]

        volume = self.bin_widths[0] * self.bin_widths[1]
        self.density = hist / (self.n * volume)

    def predict(self,x):
        #A compléter : retourne la densité associée à chaque point de x
        idx0, idx1 = self.to_bin(x)
        return self.density[idx0, idx1]

    def to_bin(self,x):
        idx = []
        for d in range(2):
            i = ((x[:,d] - self.xmin[d]) /
                 (self.xmax[d] - self.xmin[d]) * self.steps).astype(int)
            i = np.clip(i, 0, self.steps - 1)
            idx.append(i)
        return idx

class KernelDensity(Density):
    def __init__(self,kernel=None,sigma=0.1):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma
    def fit(self,x):
        self.x = x
        self.n, self.d = x.shape
    def predict(self,data):
        #A compléter : retourne la densité associée à chaque point de data
        res = np.zeros(data.shape[0])
        for xi in self.x:
            diff = (data - xi) / self.sigma
            res += self.kernel(diff)
        return res / (self.n * self.sigma ** self.d)


class Nadaraya(object):
    def __init__(self,kernel=None,sigma=0.1):
        self.kernel = kernel
        self.sigma = sigma
    def fit(self,x,y):
        self.x = x
        self.y = y
    def predict(self,data):
        res = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            diff = (data[i] - self.x) / self.sigma
            poids = self.kernel(diff)
            res[i] = np.sum(poids * self.y) / (np.sum(poids) + 1e-10)
        return res


def kernel_uniform(x):
    return np.all(np.abs(x) <= 0.5, axis=1).astype(float)

def kernel_gaussian(x):
    d = x.shape[1]
    return (2 * np.pi) ** (-d / 2) * np.exp(-0.5 * np.sum(x**2, axis=1))


def get_density2D(f,data,steps=100):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:,0].min(), data[:,0].max()
    ymin, ymax = data[:,1].min(), data[:,1].max()
    xlin,ylin = np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps)
    xx, yy = np.meshgrid(xlin,ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin

def show_density(f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure()
    show_img()
    if log:
        res = np.log(res+1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    show_img(res)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)


def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)


def load_poi(typepoi,fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])
    
    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, 
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1],v[1][0][0]] for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data,note
    

plt.ion()

geo_mat, notes = load_poi("bar")
geo_mat_resto, notes_resto = load_poi("restaurant")
geo_mat_night, notes_night = load_poi("night_club")

plt.figure()
show_img()
plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.8,s=3,label="bar")
plt.scatter(geo_mat_resto[:,0],geo_mat_resto[:,1],alpha=0.8,s=3,label="restaurant")
plt.legend()
plt.title("POI à Paris")
savefig("01_poi.png")
plt.show()


# HISTOGRAMMES

for s in [5,10,20,40]:
    f = Histogramme(steps=s)
    f.fit(geo_mat)
    show_density(f,geo_mat,log=True)
    plt.title("Histogramme bars, steps = " + str(s))
    savefig("02_hist_density_" + str(s) + ".png")
    plt.show()


X_train, X_test = train_test_split(geo_mat,test_size=0.3,random_state=0)

steps_list = [5,10,20,30,40]
ll_train = []
ll_test = []

for s in steps_list:
    f = Histogramme(steps=s)
    f.fit(X_train)
    ll_train.append(f.score(X_train))
    ll_test.append(f.score(X_test))

plt.figure()
plt.plot(steps_list,ll_train,label="train")
plt.plot(steps_list,ll_test,label="test")
plt.legend()
plt.title("Histogramme : log-vraisemblance")
savefig("03_hist_ll.png")
plt.show()

best_idx = np.argmax(ll_test)
print("Histogramme (bars) :")
print("  steps testés :", steps_list)
print("  log-vraisemblance test :", ll_test)
print("  meilleur steps =", steps_list[best_idx])
print()

# Même expérience sur les night_club

X_train, X_test = train_test_split(geo_mat_night,test_size=0.3,random_state=0)

ll_train = []
ll_test = []

for s in steps_list:
    f = Histogramme(steps=s)
    f.fit(X_train)
    ll_train.append(f.score(X_train))
    ll_test.append(f.score(X_test))

plt.figure()
plt.plot(steps_list,ll_train,label="train")
plt.plot(steps_list,ll_test,label="test")
plt.legend()
plt.title("Histogramme night_club : log-vraisemblance")
savefig("04_hist_ll_night_club.png")
plt.show()

best_idx = np.argmax(ll_test)
print("Histogramme (night_club) :")
print("  log-vraisemblance test :", ll_test)
print("  meilleur steps =", steps_list[best_idx])
print()

# montrer que c'est un densité (integrale vaut 1)

f = Histogramme(steps=20)
f.fit(geo_mat)

res, xlin, ylin = get_density2D(f, geo_mat, steps=100)
dx = xlin[1] - xlin[0]
dy = ylin[1] - ylin[0]
integrale = np.sum(res) * dx * dy

print("Intégrale histogramme =", integrale)


# METHODES A NOYAUX

X_train, X_test = train_test_split(geo_mat,test_size=0.3,random_state=0)

sigma_list = [0.003,0.005,0.01,0.02,0.05]
ll_train = []
ll_test = []

for sigma in sigma_list:
    f = KernelDensity(kernel=kernel_gaussian,sigma=sigma)
    f.fit(X_train)
    ll_train.append(f.score(X_train))
    ll_test.append(f.score(X_test))

plt.figure()
plt.plot(sigma_list,ll_train,label="train")
plt.plot(sigma_list,ll_test,label="test")
plt.xscale("log")
plt.legend()
plt.title("KDE gaussien : log-vraisemblance")
savefig("05_kde_ll_gaussian.png")
plt.show()

best_idx = np.argmax(ll_test)
print("KDE gaussien :")
print("  sigmas testés :", sigma_list)
print("  log-vraisemblance test :", ll_test)
print("  meilleur sigma =", sigma_list[best_idx])
print()


for sigma in [0.005,0.01,0.02]:
    f = KernelDensity(kernel=kernel_gaussian,sigma=sigma)
    f.fit(geo_mat)
    show_density(f,geo_mat)
    plt.title("KDE gaussien, sigma = " + str(sigma))
    savefig("06_kde_density_" + str(sigma) + ".png")
    plt.show()


ll_train = []
ll_test = []

for sigma in sigma_list:
    f = KernelDensity(kernel=kernel_uniform,sigma=sigma)
    f.fit(X_train)
    ll_train.append(f.score(X_train))
    ll_test.append(f.score(X_test))

plt.figure()
plt.plot(sigma_list,ll_train,label="train")
plt.plot(sigma_list,ll_test,label="test")
plt.xscale("log")
plt.legend()
plt.title("KDE uniforme : log-vraisemblance")
savefig("07_kde_ll_uniform.png")
plt.show()

best_idx = np.argmax(ll_test)
print("KDE uniforme :")
print("  meilleur sigma =", sigma_list[best_idx])
print()


# NADARAYA-WATSON

X_train, X_test, y_train, y_test = train_test_split(
    geo_mat,notes,test_size=0.3,random_state=0
)

sigma_list = [0.002,0.005,0.01,0.02,0.05]
mse_list = []

for sigma in sigma_list:
    f = Nadaraya(kernel=kernel_gaussian,sigma=sigma)
    f.fit(X_train,y_train)
    y_pred = f.predict(X_test)
    mse_list.append(np.mean((y_test - y_pred)**2))

plt.figure()
plt.plot(sigma_list,mse_list,marker="o")
plt.xscale("log")
plt.title("Nadaraya-Watson : MSE")
savefig("08_nadaraya_mse.png")
plt.show()

best_idx = np.argmin(mse_list)
print("Nadaraya-Watson :")
print("  sigmas testés :", sigma_list)
print("  MSE :", mse_list)
print("  meilleur sigma =", sigma_list[best_idx])
print("  MSE minimale =", mse_list[best_idx])
print()