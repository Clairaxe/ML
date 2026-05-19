import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mltools import plot_data, plot_frontiere, make_grid, gen_arti


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")


def reshape_inputs(w,x,y):
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0],w.shape[0])
    return w,x,y


def mse(w,x,y):
    # a implémenter
    w,x,y = reshape_inputs(w,x,y)
    return (x.dot(w) - y) ** 2

def mse_grad(w,x,y):
    # a implémenter
    w,x,y = reshape_inputs(w,x,y)
    return 2 * x * (x.dot(w) - y)

def reglog(w,x,y):
    #a implémenter
    w,x,y = reshape_inputs(w,x,y)
    return np.log(1 + np.exp(-y * x.dot(w)))

def reglog_grad(w,x,y):
    #a implémenter
    w,x,y = reshape_inputs(w,x,y)
    return -y * x / (1 + np.exp(y * x.dot(w)))


def check_fonctions():
    ## On fixe la seed de l'aléatoire pour vérifier les fonctions
    np.random.seed(0)
    datax, datay = gen_arti(epsilon=0.1)
    wrandom = np.random.randn(datax.shape[1],1)
    assert(np.isclose(mse(wrandom,datax,datay).mean(),0.54731,rtol=1e-4))
    assert(np.isclose(reglog(wrandom,datax,datay).mean(), 0.57053,rtol=1e-4))
    assert(np.isclose(mse_grad(wrandom,datax,datay).mean(),-1.43120,rtol=1e-4))
    assert(np.isclose(reglog_grad(wrandom,datax,datay).mean(),-0.42714,rtol=1e-4))
    np.random.seed()
    print("Tests des fonctions : OK")


def descente_gradient(datax,datay,f_loss,f_grad,eps=0.01,iter=100):
    w = np.zeros((datax.shape[1],1))

    w_list = []
    loss_list = []

    for i in range(iter):
        w_list.append(w.copy())
        loss_list.append(f_loss(w,datax,datay).mean())

        grad = f_grad(w,datax,datay).mean(axis=0).reshape(-1,1)
        w = w - eps * grad

    return w, np.array(w_list), np.array(loss_list)


def accuracy(w,x,y):
    y = y.reshape(-1,1)
    pred = np.sign(x.dot(w))
    pred[pred == 0] = 1
    return np.mean(pred == y)


if __name__=="__main__":

    check_fonctions()

    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    np.random.seed(0)
    datax, datay = gen_arti(epsilon=0.1)

    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    
    plt.figure()
    ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
    w  = np.random.randn(datax.shape[1],1)
    plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
    plot_data(datax,datay)
    plt.title("Frontière aléatoire")
    savefig("00_frontiere_aleatoire.png")
    plt.show()

    ## Visualisation de la fonction de coût en 2D
    plt.figure()
    plt.contourf(
        x_grid,
        y_grid,
        np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),
        levels=20
    )
    plt.colorbar()
    plt.title("Surface du coût MSE")
    savefig("00_surface_mse.png")
    plt.show()

     # EXPERIENCE 1 : DEUX GAUSSIENNES
 
    datax, datay = gen_arti(data_type=0, epsilon=0.1, nbex=1000)

    w_mse, ws_mse, loss_mse = descente_gradient(
        datax, datay, mse, mse_grad, eps=0.05, iter=100
    )

    w_log, ws_log, loss_log = descente_gradient(
        datax, datay, reglog, reglog_grad, eps=0.5, iter=100
    )

    acc_mse = accuracy(w_mse,datax,datay)
    acc_log = accuracy(w_log,datax,datay)

    print()
    print("Deux gaussiennes :")
    print("  coût initial MSE =", loss_mse[0])
    print("  coût final MSE =", loss_mse[-1])
    print("  accuracy MSE =", acc_mse)
    print("  poids MSE =", w_mse.reshape(-1))
    print("  coût initial logistique =", loss_log[0])
    print("  coût final logistique =", loss_log[-1])
    print("  accuracy logistique =", acc_log)
    print("  poids logistique =", w_log.reshape(-1))

    plt.figure()
    plot_frontiere(datax, lambda x: np.sign(x.dot(w_mse)), step=100)
    plot_data(datax, datay)
    plt.title("Deux gaussiennes — frontière MSE")
    savefig("01_frontiere_mse_deux_gaussiennes.png")
    plt.show()

    plt.figure()
    plot_frontiere(datax, lambda x: np.sign(x.dot(w_log)), step=100)
    plot_data(datax, datay)
    plt.title("Deux gaussiennes — frontière logistique")
    savefig("02_frontiere_logistique_deux_gaussiennes.png")
    plt.show()

    plt.figure()
    plt.plot(loss_mse,label="MSE")
    plt.plot(loss_log,label="Logistique")
    plt.xlabel("itération")
    plt.ylabel("coût moyen")
    plt.title("Évolution du coût — deux gaussiennes")
    plt.legend()
    savefig("03_cout_deux_gaussiennes.png")
    plt.show()


     # EXPERIENCE 2 : EFFET DU PAS DE GRADIENT
 
    eps_list = [0.001,0.01,0.1,1.0]

    print()
    print("Effet du pas de gradient, perte logistique :")

    plt.figure()

    for eps in eps_list:
        w_eps, ws_eps, loss_eps = descente_gradient(
            datax, datay, reglog, reglog_grad, eps=eps, iter=100
        )

        print("  eps =", eps)
        print("    coût initial =", loss_eps[0])
        print("    coût final =", loss_eps[-1])
        print("    accuracy =", accuracy(w_eps,datax,datay))

        plt.plot(loss_eps,label="eps=" + str(eps))

    plt.xlabel("itération")
    plt.ylabel("coût moyen")
    plt.title("Effet du pas de gradient — régression logistique")
    plt.legend()
    savefig("04_effet_pas_gradient.png")
    plt.show()


     # EXPERIENCE 3 : SEPARABLE VS NON SEPARABLE
 
    data_sep, y_sep = gen_arti(data_type=0, epsilon=0.02, nbex=1000)
    data_noise, y_noise = gen_arti(data_type=0, epsilon=1.0, nbex=1000)

    w_sep, ws_sep, loss_sep = descente_gradient(
        data_sep, y_sep, reglog, reglog_grad, eps=0.5, iter=100
    )

    w_noise, ws_noise, loss_noise = descente_gradient(
        data_noise, y_noise, reglog, reglog_grad, eps=0.5, iter=100
    )

    print()
    print("Cas séparable / non séparable :")
    print("  coût final séparable =", loss_sep[-1])
    print("  accuracy séparable =", accuracy(w_sep,data_sep,y_sep))
    print("  coût final non séparable =", loss_noise[-1])
    print("  accuracy non séparable =", accuracy(w_noise,data_noise,y_noise))

    plt.figure()
    plot_frontiere(data_sep, lambda x: np.sign(x.dot(w_sep)), step=100)
    plot_data(data_sep, y_sep)
    plt.title("Cas presque séparable")
    savefig("05_frontiere_separable.png")
    plt.show()

    plt.figure()
    plot_frontiere(data_noise, lambda x: np.sign(x.dot(w_noise)), step=100)
    plot_data(data_noise, y_noise)
    plt.title("Cas non séparable")
    savefig("06_frontiere_non_separable.png")
    plt.show()

    plt.figure()
    plt.plot(loss_sep,label="presque séparable")
    plt.plot(loss_noise,label="non séparable")
    plt.xlabel("itération")
    plt.ylabel("coût moyen")
    plt.title("Coût : séparable vs non séparable")
    plt.legend()
    savefig("07_cout_separable_vs_non_separable.png")
    plt.show()


     # EXPERIENCE 4 : SURFACE DE COUT + TRAJECTOIRE
 
    grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

    cost_grid_mse = np.array([
        mse(w.reshape(-1,1),datax,datay).mean()
        for w in grid
    ]).reshape(x_grid.shape)

    cost_grid_log = np.array([
        reglog(w.reshape(-1,1),datax,datay).mean()
        for w in grid
    ]).reshape(x_grid.shape)

    plt.figure()
    plt.contourf(x_grid,y_grid,cost_grid_mse,levels=30)
    plt.colorbar()
    plt.plot(ws_mse[:,0,0],ws_mse[:,1,0],"w.-",markersize=3)
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.title("Surface du coût MSE et trajectoire")
    savefig("08_surface_mse_trajectoire.png")
    plt.show()

    plt.figure()
    plt.contourf(x_grid,y_grid,cost_grid_log,levels=30)
    plt.colorbar()
    plt.plot(ws_log[:,0,0],ws_log[:,1,0],"w.-",markersize=3)
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.title("Surface du coût logistique et trajectoire")
    savefig("09_surface_logistique_trajectoire.png")
    plt.show()

    # EXPERIENCE 5 : AUTRES TYPES DE DONNEES

    print()
    print("Autres types de données :")

    for data_type, name in [(1,"quatre_gaussiennes"),(2,"echiquier")]:
        data, y = gen_arti(data_type=data_type, epsilon=0.1, nbex=1000)

        w_log, ws_log_other, loss_log_other = descente_gradient(
            data, y, reglog, reglog_grad, eps=0.5, iter=100
        )

        acc = accuracy(w_log,data,y)

        print("  " + name)
        print("    coût initial =", loss_log_other[0])
        print("    coût final =", loss_log_other[-1])
        print("    accuracy logistique =", acc)
        print("    poids =", w_log.reshape(-1))

        plt.figure()
        plot_frontiere(data, lambda x: np.sign(x.dot(w_log)), step=100)
        plot_data(data, y)
        plt.title("Régression logistique — " + name)
        savefig("10_frontiere_logistique_" + name + ".png")
        plt.show()

        plt.figure()
        plt.plot(loss_log_other)
        plt.xlabel("itération")
        plt.ylabel("coût moyen")
        plt.title("Évolution du coût — " + name)
        savefig("11_cout_logistique_" + name + ".png")
        plt.show()

    print()
    print("Toutes les figures ont été sauvegardées dans :", FIG_DIR)