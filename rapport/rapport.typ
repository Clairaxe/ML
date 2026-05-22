#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Rapport des TME de Machine Learning],
  authors: (
    (
      name: "Claire Chambaz 21522431 and Camila Maura Llauri 21522444",
    ),
  ),
)

#let fig(path, caption) = figure(
image(path, width: 75%),
caption: caption,
)

#let figwide(path, caption) = figure(
image(path, width: 95%),
caption: caption,
)

= TME 1 : Estimation de densité

== Objectif

Nous étudions deux méthodes d’estimation de densité : l’histogramme et la méthode à noyaux (KDE), sur des données de points d’intérêt (POI) à Paris. Nous utilisons également l’estimateur de Nadaraya–Watson pour prédire la note d’un lieu à partir de sa position.

== Données et visualisation

Les données proviennent de l’API Google Places et contiennent les coordonnées GPS de différents types de POI.

Le jeu de données contient :
- 4888 bars
- 6914 restaurants
- 556 boîtes de nuit

La note moyenne des bars est ≈ 3.10.

#figure(
  image("figures_tme1/01_poi.png", width: 80%),
  caption: [Répartition des bars et restaurants à Paris.]
)

== Estimation par histogramme

L’estimateur par histogramme découpe l’espace en cellules régulières.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme1/02_hist_density_5.png", width: 100%)],
    [#image("figures_tme1/02_hist_density_20.png", width: 100%)],
  ),
  caption: [Densité estimée par histogramme pour différents nombres de cellules.]
)

Pour un petit nombre de cellules (steps=5), la densité est trop grossière.  
Pour un grand nombre (steps=40), elle devient bruitée.

→ Une valeur intermédiaire est préférable.

Cette observation est confirmée quantitativement :
- meilleur steps (test) = *10* pour les bars

== Vérification de la densité

On vérifie que l’intégrale de la densité est proche de 1 :

- intégrale ≈ *1.02*

Cela confirme que l’estimateur est bien une densité.

== Choix du nombre de cellules

#figure(
  image("figures_tme1/03_hist_ll.png", width: 75%),
  caption: [Log-vraisemblance moyenne en apprentissage et en test pour l'estimation par histogramme.]
)

La log-vraisemblance test est maximale pour :
- *steps = 10* (bars)

== Cas des données rares : boîtes de nuit

Les boîtes de nuit sont moins nombreuses.

#figure(
  image("figures_tme1/04_hist_ll_night_club.png", width: 75%),
  caption: [Log-vraisemblance moyenne en apprentissage et en test pour les boîtes de nuit.]
)

- meilleur steps = *5*

→ Une discrétisation plus grossière est nécessaire pour éviter des cellules vides.

Le meilleur hyperparamètre n’est donc pas du même ordre que pour les bars.

== Estimation par noyaux (KDE)

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme1/06_kde_density_0.005.png", width: 100%)],
    [#image("figures_tme1/06_kde_density_0.02.png", width: 100%)],
  ),
  caption: [Estimation de densité estimée par KDE gaussienne pour différentes valuers de σ.]
)

Un petit σ donne une densité bruitée.  
Un grand σ donne une densité trop lissée.

#figure(
  image("figures_tme1/05_kde_ll_gaussian.png", width: 70%),
  caption: [Log-vraisemblance moyenne pour la KDE gaussienne en fonction de σ.]
)

Résultats :
- meilleur σ (gaussien) = *0.003*
- meilleur σ (uniforme) = *0.02*

Le noyau gaussien produit une densité plus lisse que le noyau uniforme.
Les intégrales obtenues sont proches de 1, on obtient donc bien une densité.

== Régression par Nadaraya–Watson

#figure(
  image("figures_tme1/08_nadaraya_mse.png", width: 70%),
  caption: [Erreur quadratique moyenne de l'estimateur Nadaraya-Watson selon σ.]
)

Résultat :
- meilleur σ = *0.01*
- MSE minimale ≈ *3.49*

→ Petit σ : sur-apprentissage  
→ Grand σ : sous-apprentissage

== Conclusion

Le choix des hyperparamètres est crucial :

- Histogramme : nombre de cellules
- KDE : σ
- Nadaraya-Watson : σ

Les résultats illustrent le compromis biais–variance.  
La validation sur un ensemble de test permet de choisir un modèle qui généralise correctement.

= TME 2 : Descente de gradient

== Objectif

Ce TME étudie la descente de gradient pour deux fonctions de coût : la perte aux moindres carrés et la perte logistique. L’objectif est d’observer la convergence de l’algorithme, l’effet du pas de gradient, et les limites d’un classifieur linéaire selon la structure des données.

== Fonctions de coût et gradients

#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("figures_tme2/00_frontiere_aleatoire.png", width: 100%)
    ],

    [
      #image("figures_tme2/00_surface_mse.png", width: 100%)
    ],
  ),
  caption: [
    Exemple de frontière de décision aléatoire et visualisation de la surface du coût MSE.
  ],
)

La surface de coût MSE présente une structure quadratique convexe.
La frontière aléatoire illustre l’effet du vecteur de poids sur la séparation des classes.

Nous avons implémenté les fonctions `mse`, `reglog`, ainsi que leurs gradients. Les fonctions prennent en entrée une matrice d’exemples $X in RR^(n times d)$, un vecteur de poids $w in RR^(d times 1)$ et un vecteur de labels $Y in RR^(n times 1)$.

Les tests fournis par `check_fonctions()` sont validés. Cela permet de vérifier que les coûts et gradients sont cohérents avant de lancer la descente de gradient.

== Deux gaussiennes

On commence par tester l’algorithme sur un problème simple à deux gaussiennes. Les deux classes sont presque linéairement séparables.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("figures_tme2/01_frontiere_mse_deux_gaussiennes.png", width: 100%)
    ],

    [
      #image("figures_tme2/02_frontiere_logistique_deux_gaussiennes.png", width: 100%)
    ],
  ),
  caption: [Frontières de décision obtenues avec la perte MSE et la perte logistique.],
)

Les deux méthodes trouvent une frontière linéaire correcte.

Résultats :
- coût final MSE ≈ 0.053
- coût final logistique ≈ 0.016
- accuracy MSE = 1.00
- accuracy logistique = 1.00

La perte logistique est plus adaptée à la classification, car elle pénalise directement les erreurs via le terme $y f(x)$.

#figure(
  image("figures_tme2/03_cout_deux_gaussiennes.png", width: 75%),
  caption: [Évolution du coût moyen au cours des itérations.],
)

Le coût diminue au cours des itérations pour les deux pertes.

== Effet du pas de gradient

On fait varier le pas de gradient $epsilon$ afin d’observer son effet.

#figure(
  image("figures_tme2/04_effet_pas_gradient.png", width: 75%),
  caption: [Effet du pas de gradient sur la convergence.],
)

Résultats :
- $epsilon = 0.001$ : coût final ≈ 0.646 (convergence lente)
- $epsilon = 0.01$ : coût final ≈ 0.382
- $epsilon = 0.1$ : coût final ≈ 0.067
- $epsilon = 1.0$ : coût final ≈ 0.009 (convergence rapide)

Dans cette expérience, l’accuracy reste égale à 1.00 pour toutes les valeurs testées de $epsilon$.

Lorsque le pas est trop petit, la convergence est lente. Un pas plus grand accélère la convergence, mais peut devenir instable dans d’autres contextes.

== Cas séparable et non séparable

On compare un cas presque séparable avec un cas bruité.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("figures_tme2/05_frontiere_separable.png", width: 100%)
    ],

    [
      #image("figures_tme2/06_frontiere_non_separable.png", width: 100%)
    ],
  ),
  caption: [Frontières de décision dans un cas séparable et non séparable.],
)

Résultats :
- cas séparable : coût ≈ 0.016, accuracy = 1.00
- cas non séparable : coût ≈ 0.232, accuracy ≈ 0.895

Dans le cas non séparable, certaines observations sont forcément mal classées.

#figure(
  image("figures_tme2/07_cout_separable_vs_non_separable.png", width: 75%),
  caption: [Évolution du coût moyen dans les cas séparable et non séparable.]
)

Le coût reste plus élevé lorsque les classes se recouvrent.

== Surface de coût et trajectoire de descente

On visualise la fonction de coût dans l’espace des poids.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("figures_tme2/08_surface_mse_trajectoire.png", width: 100%)
    ],

    [
      #image("figures_tme2/09_surface_logistique_trajectoire.png", width: 100%)
    ],
  ),
  caption: [Surface du coût et trajectoire des poids au cours de la descente de gradient.]
)

La trajectoire montre que les poids évoluent progressivement vers une zone de faible coût.  
La surface MSE est quadratique, tandis que la surface logistique est plus aplatie.

== Autres types de données

On teste la régression logistique sur des données non linéaires.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("figures_tme2/10_frontiere_logistique_quatre_gaussiennes.png", width: 100%)
    ],

    [
      #image("figures_tme2/10_frontiere_logistique_echiquier.png", width: 100%)
    ],
  ),
  caption: [Frontières de décision obtenues par régression logistique sur des données non linéaires.]
)

Résultats :
- quatre gaussiennes : accuracy ≈ 0.507, coût ≈ 0.693
- échiquier : accuracy ≈ 0.504, coût ≈ 0.693

Ces performances sont proches du hasard.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("figures_tme2/11_cout_logistique_quatre_gaussiennes.png", width: 100%)
    ],

    [
      #image("figures_tme2/11_cout_logistique_echiquier.png", width: 100%)
    ],
  ),
  caption: [Évolution du coût sur données non linéaires.],
)

Le modèle linéaire ne peut pas capturer des frontières non linéaires.

== Conclusion

La descente de gradient permet d’optimiser efficacement une fonction de coût lorsque le pas est bien choisi. La perte logistique est mieux adaptée à la classification que la MSE.

Les expériences montrent :
- l’importance du pas de gradient,
- l’impact du bruit,
- et les limites fondamentales d’un modèle linéaire face à des données non linéaires.


= TME 3 : Perceptron et SVM

== Introduction

L'objectif de ce TME est d'étudier plusieurs méthodes de classification binaire : le perceptron, ses variantes d'apprentissage par descente de gradient, les projections non linéaires, la perte hinge pénalisée, puis les SVM. Nous utilisons d'abord des données artificielles en deux dimensions, puis les données USPS de chiffres manuscrits.

Dans tout le rapport, les labels sont encodés en $-1$ et $+1$. Le classifieur linéaire prédit donc le signe de $x^T w$.

== Perceptron et classe linéaire

La perte perceptron utilisée est
$
ell(w, x, y) = max(0, - y x^T w).
$

Cette perte est nulle lorsque l'exemple est bien classé avec une marge positive, et strictement positive lorsqu'il est mal classé. Le gradient utilisé dans le code correspond à la moyenne des contributions des exemples mal classés.

Le modèle est implémenté dans une classe `Lineaire`. Cette classe permet de changer la fonction de coût, le gradient, le nombre d'itérations, le pas de gradient, la projection utilisée, ainsi que le type d'apprentissage : batch complet, stochastique ou mini-batch.

== Données USPS : classification 6 contre 9

Nous isolons d'abord deux classes, les chiffres 6 et 9. Quelques exemples des images utilisées sont représentés ci-dessous.

#fig("figures_tme3/01_exemples_usps_6_vs_9.png", [Exemples USPS utilisés pour la classification 6 contre 9.])

Le perceptron est ensuite entraîné sur ces deux classes.

#table(
columns: 4,
inset: 6pt,
align: center,
[Expérience], [Score train], [Score test], [Erreur test],
[USPS 6 vs 9], [0.998], [0.994], [0.006],
)

Le modèle distingue donc très bien les deux chiffres.

#fig("figures_tme3/02_loss_usps_6_vs_9.png", [Évolution du coût moyen du perceptron sur USPS 6 contre 9.])

#fig("figures_tme3/03_erreurs_usps_6_vs_9.png", [Erreurs d'apprentissage et de test du perceptron sur USPS 6 contre 9.])

L'erreur de test reste très proche de l'erreur d'apprentissage. On ne constate donc pas de sur-apprentissage marqué.

#fig("figures_tme3/04_poids_usps_6_vs_9.png", [Matrice de poids apprise par le perceptron pour séparer les chiffres 6 et 9.])

Cette matrice s'interprète comme un masque discriminant. Les pixels de poids positif favorisent la classe positive, ici le 9, tandis que les pixels de poids négatif favorisent la classe négative, ici le 6.

== Classification 6 contre toutes les autres classes

On entraîne ensuite un perceptron pour séparer le chiffre 6 de tous les autres chiffres. Le problème est plus difficile, car la classe négative est très hétérogène.

#table(
columns: 4,
inset: 6pt,
align: center,
[Expérience], [Score train], [Score test], [Erreur test],
[USPS 6 vs all], [0.976], [0.974], [0.026],
)

Le score reste élevé, mais l'erreur test est plus importante que dans le cas 6 contre 9. Cela confirme que le problème est plus difficile.

#fig("figures_tme3/05_loss_usps_6_vs_all.png", [Évolution du coût moyen du perceptron pour la classification 6 contre toutes les autres classes.])

#fig("figures_tme3/06_erreurs_usps_6_vs_all.png", [Erreurs d'apprentissage et de test pour la classification 6 contre toutes les autres classes.])

#fig("figures_tme3/07_poids_usps_6_vs_all.png", [Matrice de poids apprise pour séparer le chiffre 6 de toutes les autres classes.])

La matrice de poids est moins facile à interpréter que dans le cas 6 contre 9, car la classe négative regroupe plusieurs chiffres différents.

== Descente batch, stochastique et mini-batch

Nous comparons ensuite trois variantes d'apprentissage.

#figwide("figures_tme3/08_comparaison_batch_stochastic_minibatch.png", [Comparaison des coûts moyens pour les descentes batch, stochastique et mini-batch sur USPS 6 contre 9.])

#table(
columns: 4,
inset: 6pt,
align: center,
[Méthode], [Coût final], [Score train], [Score test],
[Batch complet], [0.0001], [0.996], [0.991],
[Stochastique], [0.0000], [1.000], [0.997],
[Mini-batch 32], [0.0000], [1.000], [0.994],
)

Les trois méthodes convergent bien sur ce problème. Le mini-batch et la descente stochastique progressent vite, car ils effectuent davantage de mises à jour pendant une époque.

== Effet du bruit sur la convergence

L'énoncé demande aussi de comparer la vitesse de convergence en fonction du bruit dans le jeu de données. Le coût perceptron peut devenir nul très vite lorsque les données sont séparables. Pour rendre la comparaison plus lisible, on trace ici l'erreur d'apprentissage au cours des époques.

#figure(
grid(
columns: 3,
gutter: 8pt,
[#image("figures_tme3/15_bruit_erreur_epsilon0.2.png", width: 100%)],
[#image("figures_tme3/15_bruit_erreur_epsilon0.6.png", width: 100%)],
[#image("figures_tme3/15_bruit_erreur_epsilon1.0.png", width: 100%)],
),
caption: [Effet du bruit sur l'erreur d'apprentissage pour les descentes batch, stochastique et mini-batch.],
)

Lorsque le bruit augmente, les classes se recouvrent davantage. L'erreur finale devient plus élevée et les trajectoires sont moins régulières. Les méthodes stochastique et mini-batch peuvent être plus bruitées, car elles utilisent moins d'exemples à chaque mise à jour.

== Projection polynomiale

Un modèle linéaire dans l'espace initial ne peut produire qu'une frontière linéaire. Pour augmenter son expressivité, on projette les données dans un espace de plus grande dimension.

$
(1, x_1, x_2, ..., x_d, x_1^2, x_1 x_2, ..., x_d^2).
$

#table(
columns: 3,
inset: 6pt,
align: center,
[Jeu de données], [Score train], [Erreur train],
[Deux gaussiennes], [1.000], [0.000],
[Quatre gaussiennes], [0.995], [0.005],
[Échiquier], [0.496], [0.504],
)

#fig("figures_tme3/09_projection_poly_deux_gaussiennes.png", [Frontière obtenue avec une projection polynomiale de degré 2 avec biais sur deux gaussiennes.])

#fig("figures_tme3/09_projection_poly_quatre_gaussiennes.png", [Frontière obtenue avec une projection polynomiale de degré 2 avec biais sur quatre gaussiennes.])

#fig("figures_tme3/09_projection_poly_echiquier.png", [Frontière obtenue avec une projection polynomiale de degré 2 avec biais sur l'échiquier.])

La projection polynomiale fonctionne très bien pour les deux gaussiennes et les quatre gaussiennes. En revanche, elle échoue sur l'échiquier : le score est proche de 0.5, donc proche du hasard. Une frontière quadratique reste insuffisante pour cette structure.

== Projection gaussienne

Nous utilisons ensuite une projection gaussienne sur des points de base $b_1, ..., b_m$ :
$
phi(x) = ( exp(- ||x - b_1||^2 / (2 sigma^2)), ..., exp(- ||x - b_m||^2 / (2 sigma^2)) ).
$

#figure(
grid(
columns: 3,
gutter: 8pt,
[#image("figures_tme3/16_projection_gaussienne_deux_gaussiennes.png", width: 100%)],
[#image("figures_tme3/16_projection_gaussienne_quatre_gaussiennes.png", width: 100%)],
[#image("figures_tme3/16_projection_gaussienne_echiquier.png", width: 100%)],
),
caption: [Projection gaussienne sur les trois jeux de données artificielles.],
)

La projection gaussienne fonctionne très bien sur les deux gaussiennes et les quatre gaussiennes, mais reste plus difficile à ajuster sur l'échiquier.

#table(
columns: 4,
inset: 6pt,
align: center,
[Nombre de points de base], [$sigma$], [Score train], [Erreur train],
[20], [0.2], [0.491], [0.509],
[20], [1.0], [0.547], [0.453],
[80], [0.2], [0.630], [0.370],
[80], [1.0], [0.557], [0.443],
)

#figure(
grid(
columns: 2,
gutter: 10pt,
[#image("figures_tme3/11_projection_gaussienne_type2_base20_sigma0.2.png", width: 100%)],
[#image("figures_tme3/11_projection_gaussienne_type2_base80_sigma0.2.png", width: 100%)],
),
caption: [Effet du nombre de points de base pour une projection gaussienne avec $sigma = 0.2$ sur l'échiquier.],
)

La meilleure configuration testée ici est `nb_base = 80` et $sigma = 0.2$. L’augmentation du nombre de points de base améliore l’expressivité du modèle. Un petit $sigma$ donne une influence plus locale des points de base, alors qu’un grand $sigma$ produit une frontière plus lisse. Les points de base avec les plus grands poids se situent surtout près des zones où la frontière doit changer.

== Perte hinge et pénalisation

On remplace ensuite la perte perceptron par une perte hinge pénalisée :
$
ell(w, x, y) = max(0, alpha - y x^T w) + lambda ||w||^2.
$

#table(
columns: 4,
inset: 6pt,
align: center,
[$alpha$], [$lambda$], [Score train], [Coût final],
[0.5], [1e-4], [0.641], [0.3740],
[1.0], [1e-4], [0.622], [0.7967],
[2.0], [1e-4], [0.567], [1.6986],
[1.0], [1e-2], [0.560], [0.9628],
[1.0], [1e-1], [0.542], [0.9963],
)

#fig("figures_tme3/13_hinge_resume_scores.png", [Score d'apprentissage selon les valeurs de $alpha$ et $lambda$ pour la perte hinge gaussienne.])

#figure(
grid(
columns: 2,
gutter: 10pt,
[#image("figures_tme3/13_hinge_alpha0.5_lambda0.0001.png", width: 100%)],
[#image("figures_tme3/13_hinge_alpha1.0_lambda0.1.png", width: 100%)],
),
caption: [Deux frontières hinge gaussiennes : faible régularisation à gauche et forte régularisation à droite.],
)

Lorsque $alpha$ augmente, le coût final augmente car la marge demandée est plus grande. Lorsque $lambda$ augmente, les poids sont davantage pénalisés, ce qui peut entraîner du sous-apprentissage. Ici, le meilleur score parmi les configurations testées est obtenu avec $alpha = 0.5$ et $lambda = 10^(-4)$.

Cette perte est proche de celle utilisée dans les SVM : elle introduit une marge et une pénalisation sur les poids. La différence principale est qu'ici nous optimisons directement notre modèle projeté, alors que les SVM utilisent une formulation standard avec noyaux et vecteurs supports.

== SVM et Grid Search

Enfin, nous utilisons les SVM de `sklearn`. Les paramètres sont choisis par validation croisée avec `GridSearchCV`. Pour l'échiquier, la grille de paramètres a été élargie pour le noyau RBF, car une grille trop petite donnait des résultats trop faibles.

#fig("figures_tme3/14_svm_echiquier_vecteurs_supports.png", [Frontière de décision du meilleur SVM sur l'échiquier, avec les vecteurs supports indiqués en noir.])

L'énoncé demande aussi comment évolue le nombre de vecteurs supports selon le noyau et les paramètres.

#figwide("figures_tme3/17_svm_2d_parametres_vecteurs_supports.png", [Comparaison des noyaux SVM sur l'échiquier : score test et nombre de vecteurs supports.])

#figwide("figures_tme3/18_svm_usps_parametres_vecteurs_supports.png", [Comparaison des noyaux SVM sur USPS 6 contre 9 : score test et nombre de vecteurs supports.])

Les vecteurs supports sont les exemples qui participent directement à la définition de la frontière. Sur l'échiquier, leur nombre est élevé, ce qui reflète la complexité de la frontière. Sur USPS 6 contre 9, certains paramètres donnent un excellent score avec peu de vecteurs supports, alors qu'un noyau trop flexible peut utiliser beaucoup plus de points.

Lorsque `gamma` devient très grand pour le noyau RBF, le modèle devient très local. Le score d'apprentissage peut alors devenir presque parfait tandis que le score test diminue : on observe du sur-apprentissage. Le nombre de vecteurs supports augmente également fortement.

Dans le cas linéaire, on retrouve une frontière proche de celle obtenue avec le perceptron. Les noyaux non linéaires, en particulier RBF, permettent d'obtenir des frontières plus flexibles.

== Conclusion

Ce TME met en évidence les limites et les extensions naturelles des modèles linéaires. Le perceptron fonctionne bien lorsque les données sont presque linéairement séparables, comme pour USPS 6 contre 9. Le cas 6 contre toutes les autres classes est plus difficile, mais reste bien traité.

Les projections polynomiales permettent de résoudre certains problèmes non linéaires simples, comme les quatre gaussiennes, mais échouent sur l'échiquier. Les projections gaussiennes améliorent les résultats sur certains problèmes, mais restent limitées sur l'échiquier avec les paramètres testés. La perte hinge introduit une notion de marge et rapproche le modèle des SVM.

Enfin, les SVM avec noyau RBF donnent les meilleurs résultats sur les données complexes, notamment l'échiquier et USPS 6 contre 9.

= TME 5 : Clustering spatial des points d'intérêt parisiens

== Introduction

L'objectif de ce TME est de caractériser spatialement l'espace urbain parisien à partir des points d'intérêt présents dans différentes zones. Chaque région est représentée par un profil de types de POIs, puis ces profils sont regroupés par KMeans.

Dans le jeu de données utilisé, les types de POIs ne sont pas codés par une seule colonne catégorielle. Ils sont représentés par des colonnes indicatrices : `restaurant`, `bar`, `cafe`, `bakery`, etc. Une région est donc décrite par la moyenne de ces colonnes sur les POIs qu'elle contient.

#fig("figures_tme5/01_poi_paris.png", [Répartition des points d'intérêt dans Paris.])

== Discrétisation manuelle par grille

On ne peut pas utiliser directement les arrondissements, car ils sont trop grands et peuvent regrouper des sous-régions très différentes. On commence donc par discrétiser l'espace avec une grille régulière de taille $N times N$.

Pour une coordonnée GPS (longitude, latitude), la cellule correspondante est calculée par :

i = floor(N x (longitude - lomin) / (lomax - lomin))

j = floor(N x (latitude - lamin) / (lamax - lamin))

Chaque cellule est ensuite décrite par un vecteur de dimension 12. Chaque coordonnée correspond à la proportion d'un type de POI dans la cellule. Une cellule contenant surtout des restaurants et des bars aura donc des valeurs élevées sur ces coordonnées.

On applique ensuite KMeans aux descriptions des cellules non vides.

#fig("figures_tme5/02_clustering_grille.png", [Clustering des cellules obtenu avec une discrétisation en grille.])

Les cellules d'une même couleur ont des profils de POIs similaires. Cette visualisation permet de repérer des régions ayant une composition urbaine proche, même si elles ne sont pas nécessairement voisines.

== Centroïdes des clusters de la grille

Les centroïdes permettent d'interpréter les clusters. Chaque centroïde représente le profil moyen des cellules appartenant au cluster.

#fig("figures_tme5/03_centroides_grille.png", [Centroïdes des clusters obtenus avec la grille.])

Un cluster peut par exemple être dominé par les restaurants et bars, tandis qu'un autre peut être davantage marqué par les commerces, les cafés ou les hébergements. C'est cette lecture des centroïdes qui donne un sens aux couleurs observées sur la carte.

== Choix du nombre de clusters : méthode elbow

Le score elbow correspond à l'inertie du modèle KMeans. L'inertie mesure la somme des distances quadratiques entre les points et leur centroïde. Elle diminue nécessairement lorsque le nombre de clusters augmente.

#fig("figures_tme5/04_elbow_grille.png", [Courbe elbow pour le clustering des cellules de la grille.])

On cherche un compromis : un nombre de clusters assez grand pour bien décrire les différences entre régions, mais pas trop grand pour garder une interprétation simple. Le nombre optimal est situé près du coude de la courbe, c'est-à-dire au moment où la baisse d'inertie devient nettement plus faible.

Dans notre cas, le coude semble apparaître autour de $K = $ #strong[à compléter]. On retient donc ce nombre comme compromis raisonnable pour la discrétisation en grille.

== Limites de la discrétisation en grille

La grille est simple à construire et à interpréter, mais elle reste arbitraire. Certaines cellules contiennent beaucoup de POIs, d'autres très peu, voire aucun. De plus, la grille peut couper artificiellement des quartiers cohérents ou mélanger des zones différentes dans une même cellule.

Cette méthode donne donc une première approximation, mais elle ne tient pas compte de la densité réelle des POIs.

== Discrétisation automatique de l'espace

Pour obtenir des régions plus adaptées aux données, on applique KMeans directement aux coordonnées GPS des POIs. Cette étape correspond à une quantization spatiale. Une grille $10 times 10$ contient 100 cellules ; on peut donc comparer cette approche à une quantization avec$K_{"geo"}$
clusters.

#fig("figures_tme5/05_quantization_spatiale.png", [Quantization spatiale des POIs par KMeans.])

Contrairement à la grille régulière, cette discrétisation s'adapte à la densité des données. Les régions sont plus petites dans les zones riches en POIs et plus grandes dans les zones moins denses.

== Clustering des régions spatiales

Une fois les clusters spatiaux obtenus, chaque région spatiale est décrite par la distribution moyenne de ses types de POIs. On applique ensuite un second KMeans à ces descriptions.

#fig("figures_tme5/06_clustering_regions_spatiales.png", [Clustering des régions spatiales selon les types de POIs.])

Cette carte est généralement plus pertinente que celle obtenue avec la grille. Les régions sont moins arbitraires et suivent davantage la géométrie réelle des points d'intérêt.

#fig("figures_tme5/07_centroides_regions_spatiales.png", [Centroïdes des clusters obtenus après quantization spatiale.])

Les centroïdes s'interprètent comme précédemment : chaque barre indique l'importance moyenne d'un type de POI dans un cluster. Ils permettent de comprendre pourquoi certaines régions sont regroupées.

== Choix du nombre de clusters pour les régions spatiales

On trace de nouveau la courbe elbow, cette fois pour le clustering des régions spatiales décrites par leurs types de POIs.

#fig("figures_tme5/08_elbow_regions_spatiales.png", [Courbe elbow pour les régions spatiales.])

Le coude de la courbe indique le nombre de clusters à retenir. Dans notre cas, il semble apparaître autour de $K = $ #strong[à compléter]. Ce choix doit aussi rester interprétable : trop peu de clusters donnent une description trop grossière, tandis que trop de clusters rendent les résultats fragmentés.

== Comparaison entre grille et quantization spatiale

La discrétisation en grille est simple, mais elle impose une structure régulière qui ne correspond pas nécessairement à la densité des POIs. La quantization spatiale est plus adaptée, car elle regroupe les POIs selon leur proximité géographique.

Ainsi, la grille est plus facile à expliquer, mais la quantization spatiale donne souvent des régions plus pertinentes. Pour un rapport, il est intéressant de montrer les deux : la grille comme méthode simple de référence, puis la quantization spatiale comme amélioration.

== Corrélation entre types de POIs

On observe ensuite les corrélations entre types de POIs à partir des descriptions régionales.

#fig("figures_tme5/09_correlation_types.png", [Matrice de corrélation entre types de POIs.])

Des corrélations positives indiquent que certains types apparaissent souvent ensemble dans les mêmes régions. Par exemple, les restaurants, bars et cafés peuvent être associés dans des zones très fréquentées. Des corrélations négatives indiquent au contraire que certains types sont rarement présents dans les mêmes profils régionaux.

Ces corrélations peuvent influencer le KMeans, car des variables redondantes peuvent compter plusieurs fois dans la distance euclidienne.

== Traitement proposé pour améliorer les résultats

Un traitement simple consiste à standardiser les colonnes avant le clustering. Cela donne un poids comparable à chaque type de POI et évite que les catégories les plus fréquentes dominent complètement la distance.

#fig("figures_tme5/10_clustering_standardise.png", [Clustering obtenu après standardisation des descriptions de types.])

La standardisation peut faire apparaître des clusters plus équilibrés, en donnant davantage d'importance aux types rares. Elle peut cependant rendre l'interprétation des centroïdes moins directe, car les variables ne sont plus des proportions brutes.

D'autres traitements seraient possibles : normalisation des profils, PCA pour réduire les redondances entre types corrélés, ou encore choix d'une autre distance plus adaptée aux distributions.

== Conclusion

Ce TME montre comment caractériser l'espace urbain parisien à partir des distributions de POIs. La discrétisation en grille fournit une première approche simple, mais assez arbitraire. La quantization spatiale par KMeans donne une partition plus adaptée à la densité réelle des points.

L'interprétation repose sur trois éléments complémentaires : les cartes, les centroïdes et les courbes elbow. Les cartes montrent l'organisation spatiale des clusters, les centroïdes expliquent leur composition, et les courbes elbow aident à choisir un nombre raisonnable de clusters.

Enfin, l'analyse des corrélations entre types montre qu'un prétraitement peut être utile. La standardisation est une première solution pour limiter la domination des types les plus fréquents et rendre le clustering plus équilibré.
