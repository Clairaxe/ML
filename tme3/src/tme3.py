import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

from mltools import plot_data, plot_frontiere, make_grid, gen_arti


  
# DOSSIER FIGURES
  

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")


  
# COUTS ET GRADIENTS
  

def perceptron_loss(w,datax,datay):
    return np.maximum(np.zeros((len(datay),1)), -datay * datax.dot(w))

def perceptron_grad(w,datax,datay):
    mask = (datay * datax.dot(w) < 0).astype(float)
    return -datax.T.dot(mask * datay) / len(datay)

def hinge_loss(w,datax,datay,alpha=1.0,lamb=0.0):
    return np.maximum(0, alpha - datay * datax.dot(w)) + lamb * np.sum(w ** 2)

def hinge_grad(w,datax,datay,alpha=1.0,lamb=0.0):
    mask = (datay * datax.dot(w) < alpha).astype(float)
    return -datax.T.dot(mask * datay) / len(datay) + 2 * lamb * w


  
# PROJECTIONS
  

def proj_identite(datax):
    return datax

def proj_ajout_biais(datax):
    return np.hstack((np.ones((datax.shape[0],1)), datax))

def proj_poly(datax):
    n, d = datax.shape
    nb_quad = d * (d + 1) // 2
    res = np.zeros((n, d + nb_quad))
    res[:, :d] = datax

    for i in range(n):
        mat = datax[i].reshape(-1,1).dot(datax[i].reshape(1,-1))
        res[i, d:] = mat[np.triu_indices(d)]

    return res

def proj_biais(datax):
    n, d = datax.shape
    nb_quad = d * (d + 1) // 2
    res = np.zeros((n, 1 + d + nb_quad))
    res[:,0] = 1
    res[:,1:1+d] = datax

    for i in range(n):
        mat = datax[i].reshape(-1,1).dot(datax[i].reshape(1,-1))
        res[i, 1+d:] = mat[np.triu_indices(d)]

    return res

def proj_gauss(datax,base,sigma):
    distances = ((datax[:,None,:] - base[None,:,:]) ** 2).sum(axis=2)
    return np.exp(-distances / (2 * sigma ** 2))


  
# OUTILS USPS
  

def binarize_labels(y,neg,pos):
    return np.where(y == neg, -1, 1).reshape(-1,1)

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp = list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data,title=None):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    if title is not None:
        plt.title(title)
    plt.axis("off")


  
# CLASSE LINEAIRE
  

class Lineaire(object):
    def __init__(
        self,
        loss=perceptron_loss,
        loss_g=perceptron_grad,
        max_iter=100,
        eps=0.01,
        projection=None,
        batch_size=32,
        loss_params=None,
        random_state=0
    ):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
        self.projection = projection
        self.batch_size = batch_size
        self.loss_params = {} if loss_params is None else loss_params
        self.random_state = random_state

        self.histo_cout = []
        self.score_train = []
        self.score_test = []
        self.err_train = []
        self.err_test = []
        
    def _transform(self,datax):
        if self.projection is not None:
            return self.projection(datax)
        return datax
        
    def fit(self,datax,datay,testx=None,testy=None,type_learning="normal"):
        rng = np.random.default_rng(self.random_state)

        x = self._transform(datax)
        y = datay.reshape(-1,1)

        self.w = np.random.randn(x.shape[1],1) * 0.01

        self.histo_cout = []
        self.score_train = []
        self.score_test = []
        self.err_train = []
        self.err_test = []

        n = len(x)

        for epoch in range(self.max_iter):

            if type_learning == "normal":
                self.w = self.w - self.eps * self.loss_g(self.w,x,y,**self.loss_params)

            elif type_learning == "stochastic":
                indices = rng.permutation(n)
                for j in indices:
                    xj = x[j:j+1]
                    yj = y[j:j+1]
                    self.w = self.w - self.eps * self.loss_g(self.w,xj,yj,**self.loss_params)

            elif type_learning == "mini-batch":
                indices = rng.permutation(n)
                for start in range(0,n,self.batch_size):
                    batch = indices[start:start+self.batch_size]
                    xb = x[batch]
                    yb = y[batch]
                    self.w = self.w - self.eps * self.loss_g(self.w,xb,yb,**self.loss_params)

            else:
                raise ValueError("type_learning doit être normal, stochastic ou mini-batch")

            self.histo_cout.append(np.mean(self.loss(self.w,x,y,**self.loss_params)))

            sc_train = self.score(datax,datay)
            self.score_train.append(sc_train)
            self.err_train.append(1 - sc_train)

            if testx is not None and testy is not None:
                sc_test = self.score(testx,testy)
                self.score_test.append(sc_test)
                self.err_test.append(1 - sc_test)

        return self.w,self.histo_cout,self.score_test

    def predict(self,datax):
        x = self._transform(datax)
        pred = np.sign(x.dot(self.w))
        pred[pred == 0] = 1
        return pred

    def score(self,datax,datay):
        return np.mean(self.predict(datax) == datay)


  
# FIGURES
  

def plot_loss(model,title,filename):
    plt.figure()
    plt.plot(model.histo_cout)
    plt.xlabel("Époque")
    plt.ylabel("Coût moyen")
    plt.title(title)
    savefig(filename)
    plt.show()

def plot_errors(model,title,filename):
    plt.figure()
    plt.plot(model.err_train,label="train")
    if len(model.err_test) > 0:
        plt.plot(model.err_test,label="test")
    plt.xlabel("Époque")
    plt.ylabel("Erreur")
    plt.title(title)
    plt.legend()
    savefig(filename)
    plt.show()

def plot_usps_examples(datax,datay,filename,n=12):
    plt.figure(figsize=(12,3))
    idx = np.random.choice(len(datax),size=n,replace=False)

    for k,i in enumerate(idx):
        plt.subplot(1,n,k+1)
        show_usps(datax[i],title=str(datay[i]))

    savefig(filename)
    plt.show()

def plot_weight_usps(w,title,filename):
    plt.figure()
    w_img = w.reshape(-1)

    if len(w_img) == 257:
        w_img = w_img[1:]

    show_usps(w_img,title=title)
    plt.colorbar()
    savefig(filename)
    plt.show()

def plot_frontiere_modele(datax,datay,model,title,filename):
    plt.figure()
    plot_frontiere(datax,model.predict,step=100)
    plot_data(datax,datay)
    plt.title(title)
    plt.legend()
    savefig(filename)
    plt.show()


  
# EXPERIENCES
  

def experience_usps_6_vs_9():
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"

    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)

    neg = 6
    pos = 9

    datax,datay_raw = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy_raw = get_usps([neg,pos],alltestx,alltesty)

    datay = binarize_labels(datay_raw,neg,pos)
    testy = binarize_labels(testy_raw,neg,pos)

    plot_usps_examples(datax,datay_raw,"01_exemples_usps_6_vs_9.png")

    l = Lineaire(
        loss=perceptron_loss,
        loss_g=perceptron_grad,
        max_iter=100,
        eps=0.01
    )

    w,histo_cout,score_test = l.fit(
        datax=datax,
        datay=datay,
        testx=testx,
        testy=testy,
        type_learning="normal"
    )

    sc_train = l.score(datax,datay)
    sc_test = l.score(testx,testy)

    print()
    print("=== USPS 6 vs 9 ===")
    print(f"Score train : {sc_train:.3f}")
    print(f"Score test  : {sc_test:.3f}")
    print(f"Erreur test : {1 - sc_test:.3f}")
    print(f"Coût final  : {l.histo_cout[-1]:.4f}")

    plot_loss(l,"USPS 6 vs 9 — évolution du coût","02_loss_usps_6_vs_9.png")
    plot_errors(l,"USPS 6 vs 9 — erreurs train/test","03_erreurs_usps_6_vs_9.png")
    plot_weight_usps(w,"USPS 6 vs 9 — poids du perceptron","04_poids_usps_6_vs_9.png")

    return l


def experience_usps_6_vs_all():
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"

    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)

    pos = 6

    datay = np.where(alltrainy == pos,1,-1).reshape(-1,1)
    testy = np.where(alltesty == pos,1,-1).reshape(-1,1)

    l = Lineaire(
        loss=perceptron_loss,
        loss_g=perceptron_grad,
        max_iter=100,
        eps=0.01
    )

    w,histo_cout,score_test = l.fit(
        datax=alltrainx,
        datay=datay,
        testx=alltestx,
        testy=testy,
        type_learning="normal"
    )

    sc_train = l.score(alltrainx,datay)
    sc_test = l.score(alltestx,testy)

    print()
    print("=== USPS 6 vs all ===")
    print(f"Score train : {sc_train:.3f}")
    print(f"Score test  : {sc_test:.3f}")
    print(f"Erreur test : {1 - sc_test:.3f}")
    print(f"Coût final  : {l.histo_cout[-1]:.4f}")

    plot_loss(l,"USPS 6 vs all — évolution du coût","05_loss_usps_6_vs_all.png")
    plot_errors(l,"USPS 6 vs all — erreurs train/test","06_erreurs_usps_6_vs_all.png")
    plot_weight_usps(w,"USPS 6 vs all — poids du perceptron","07_poids_usps_6_vs_all.png")

    return l


def experience_batch_comparison():
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"

    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)

    neg = 6
    pos = 9

    datax,datay_raw = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy_raw = get_usps([neg,pos],alltestx,alltesty)

    datay = binarize_labels(datay_raw,neg,pos)
    testy = binarize_labels(testy_raw,neg,pos)

    configs = [
        ("normal","batch complet",32),
        ("stochastic","stochastique",1),
        ("mini-batch","mini-batch 32",32),
    ]

    print()
    print("=== Comparaison batch / stochastique / mini-batch ===")

    plt.figure()

    for type_learning,label,batch_size in configs:
        l = Lineaire(
            loss=perceptron_loss,
            loss_g=perceptron_grad,
            max_iter=50,
            eps=0.01,
            batch_size=batch_size
        )

        l.fit(datax,datay,testx=testx,testy=testy,type_learning=type_learning)

        print(label)
        print(f"  coût final  : {l.histo_cout[-1]:.4f}")
        print(f"  score train : {l.score(datax,datay):.3f}")
        print(f"  score test  : {l.score(testx,testy):.3f}")

        plt.plot(l.histo_cout,label=label)

    plt.xlabel("Époque")
    plt.ylabel("Coût moyen")
    plt.title("Comparaison des descentes — USPS 6 vs 9")
    plt.legend()
    savefig("08_comparaison_batch_stochastic_minibatch.png")
    plt.show()


def experience_projection_poly():
    print()
    print("=== Projection polynomiale ===")

    for data_type,name in [(0,"deux_gaussiennes"),(1,"quatre_gaussiennes"),(2,"echiquier")]:
        datax,datay = gen_arti(data_type=data_type,epsilon=0.1,nbex=1000)

        l = Lineaire(
            loss=perceptron_loss,
            loss_g=perceptron_grad,
            max_iter=200,
            eps=0.01,
            projection=proj_biais,
            batch_size=32
        )

        l.fit(datax,datay,type_learning="mini-batch")

        score = l.score(datax,datay)

        print(name)
        print(f"  score train : {score:.3f}")
        print(f"  erreur train : {1 - score:.3f}")
        print(f"  coût final : {l.histo_cout[-1]:.4f}")

        plot_frontiere_modele(
            datax,
            datay,
            l,
            "Projection polynomiale degré 2 — " + name,
            "09_projection_poly_" + name + ".png"
        )

        plot_loss(
            l,
            "Coût — projection polynomiale — " + name,
            "10_loss_projection_poly_" + name + ".png"
        )


def experience_projection_gaussienne(data_type=2,nb_base=80,sigma=0.5):
    datax,datay = gen_arti(data_type=data_type,epsilon=0.1,nbex=1000)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(datax),size=nb_base,replace=False)
    base = datax[idx]

    projection = lambda x: proj_gauss(x,base=base,sigma=sigma)

    l = Lineaire(
        loss=hinge_loss,
        loss_g=hinge_grad,
        max_iter=200,
        eps=0.05,
        projection=projection,
        batch_size=32,
        loss_params={"alpha":1.0,"lamb":1e-3}
    )

    l.fit(datax,datay,type_learning="mini-batch")

    score = l.score(datax,datay)

    print()
    print("=== Projection gaussienne ===")
    print(f"data_type = {data_type}, nb_base = {nb_base}, sigma = {sigma}")
    print(f"Score train : {score:.3f}")
    print(f"Erreur train : {1 - score:.3f}")
    print(f"Coût final : {l.histo_cout[-1]:.4f}")

    plt.figure()
    plot_frontiere(datax,l.predict,step=100)
    plot_data(datax,datay)

    weights = np.abs(l.w.reshape(-1))
    sizes = 20 + 300 * weights / (weights.max() + 1e-12)

    plt.scatter(
        base[:,0],
        base[:,1],
        s=sizes,
        facecolors="none",
        edgecolors="black",
        label="points de base"
    )

    plt.title(f"Projection gaussienne — base={nb_base}, sigma={sigma}")
    plt.legend()
    savefig(f"11_projection_gaussienne_type{data_type}_base{nb_base}_sigma{sigma}.png")
    plt.show()

    plot_loss(
        l,
        f"Coût — projection gaussienne — base={nb_base}, sigma={sigma}",
        f"12_loss_projection_gaussienne_type{data_type}_base{nb_base}_sigma{sigma}.png"
    )

    return l


def experience_gaussienne_parametres():
    params = [
        (20,0.2),
        (20,1.0),
        (80,0.2),
        (80,1.0),
    ]

    for nb_base,sigma in params:
        experience_projection_gaussienne(data_type=2,nb_base=nb_base,sigma=sigma)


def experience_hinge_alpha_lambda():
    datax,datay = gen_arti(data_type=2,epsilon=0.1,nbex=1000)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(datax),size=80,replace=False)
    base = datax[idx]

    projection = lambda x: proj_gauss(x,base=base,sigma=0.5)

    configs = [
        (0.5,1e-4),
        (1.0,1e-4),
        (2.0,1e-4),
        (1.0,1e-2),
        (1.0,1e-1),
    ]

    print()
    print("=== Effet de alpha et lambda ===")

    for alpha,lamb in configs:
        l = Lineaire(
            loss=hinge_loss,
            loss_g=hinge_grad,
            max_iter=200,
            eps=0.05,
            projection=projection,
            batch_size=32,
            loss_params={"alpha":alpha,"lamb":lamb}
        )

        l.fit(datax,datay,type_learning="mini-batch")

        score = l.score(datax,datay)

        print(f"alpha={alpha}, lambda={lamb}")
        print(f"  score train : {score:.3f}")
        print(f"  erreur train : {1 - score:.3f}")
        print(f"  coût final : {l.histo_cout[-1]:.4f}")

        plt.figure()
        plot_frontiere(datax,l.predict,step=100)
        plot_data(datax,datay)
        plt.title(f"Hinge gaussien — alpha={alpha}, lambda={lamb}")
        plt.legend()
        savefig(f"13_hinge_alpha{alpha}_lambda{lamb}.png")
        plt.show()


def experience_svm_2d():
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, train_test_split

    datax,datay = gen_arti(data_type=2,epsilon=0.1,nbex=1000)
    datay = datay.reshape(-1)

    xtrain,xtest,ytrain,ytest = train_test_split(
        datax,
        datay,
        test_size=0.3,
        random_state=0,
        stratify=datay
    )

    param_grid = [
        {"kernel":["linear"],"C":[0.1,1,10]},
        {"kernel":["rbf"],"C":[0.1,1,10],"gamma":[0.1,1,10]},
        {"kernel":["poly"],"C":[0.1,1,10],"degree":[2,3],"gamma":["scale"]},
    ]

    grid = GridSearchCV(SVC(),param_grid,cv=5)
    grid.fit(xtrain,ytrain)

    svm = grid.best_estimator_

    print()
    print("=== SVM 2D ===")
    print("Meilleurs paramètres :", grid.best_params_)
    print(f"Score train : {svm.score(xtrain,ytrain):.3f}")
    print(f"Score test  : {svm.score(xtest,ytest):.3f}")
    print(f"Nb vecteurs supports : {len(svm.support_)}")

    plt.figure()
    plot_frontiere(datax,lambda x: svm.predict(x),step=100)
    plot_data(datax,datay.reshape(-1,1))

    plt.scatter(
        xtrain[svm.support_,0],
        xtrain[svm.support_,1],
        s=80,
        facecolors="none",
        edgecolors="black",
        label="vecteurs supports"
    )

    plt.title("SVM — échiquier — vecteurs supports")
    plt.legend()
    savefig("14_svm_echiquier_vecteurs_supports.png")
    plt.show()

    return svm,grid


def experience_svm_usps():
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"

    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)

    neg = 6
    pos = 9

    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)

    param_grid = [
        {"kernel":["linear"],"C":[0.1,1,10]},
        {"kernel":["rbf"],"C":[0.1,1,10],"gamma":[0.001,0.01,0.1]},
        {"kernel":["poly"],"C":[0.1,1,10],"degree":[2,3],"gamma":["scale"]},
    ]

    grid = GridSearchCV(SVC(),param_grid,cv=5)
    grid.fit(datax,datay)

    svm = grid.best_estimator_

    print()
    print("=== SVM USPS 6 vs 9 ===")
    print("Meilleurs paramètres :", grid.best_params_)
    print(f"Score train : {svm.score(datax,datay):.3f}")
    print(f"Score test  : {svm.score(testx,testy):.3f}")
    print(f"Nb vecteurs supports : {len(svm.support_)}")

    return svm,grid


  
# MAIN
  

if __name__ =="__main__":

    np.random.seed(0)

    experience_usps_6_vs_9()
    experience_usps_6_vs_all()
    experience_batch_comparison()
    experience_projection_poly()
    experience_gaussienne_parametres()
    experience_hinge_alpha_lambda()
    experience_svm_2d()
    experience_svm_usps()

    print()
    print("Toutes les figures ont été sauvegardées dans :", FIG_DIR)