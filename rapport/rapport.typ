#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Rapport des TME de Machine Learning],
  authors: (
    (
      name: "Claire Chambaz (21522431) et Camila Maura Llauri (21522444)",
    ),
  ),
)

#set par(
  spacing: 1em,
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

== Introduction

On étudie deux méthodes d’estimation de densité : l’histogramme et la méthode à noyaux (KDE), sur des données de points d’intérêt (POI) à Paris. On  utilise également l’estimateur de Nadaraya-Watson pour prédire la note d’un lieu à partir de sa position.

Les données proviennent de l’API Google Places et contiennent les coordonnées GPS de différents types de POI. Le jeu de données contient :
- 4888 bars (La note moyenne des bars est ≈ 3.10)
- 6914 restaurants
- 556 boîtes de nuit

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

Lorsque steps augmente, la log-vraisemblance d’apprentissage augmente : l’histogramme devient de plus en plus précis sur les données d’apprentissage. En revanche, la log-vraisemblance test commence par augmenter puis diminue fortement pour les grandes valeurs de steps. Cela indique du sur-apprentissage : les cellules deviennent trop petites et l’estimateur généralise moins bien.

- Pour un petit nombre de cellules (steps=5), la densité est trop grossière.  
- Pour un grand nombre (steps=40), elle devient bruitée.

#figure(
  table(
    columns: 4,
    inset: 6pt,
    align: center,
    [steps], [Log-vraisemblance train], [Log-vraisemblance test], [Intégrale],
    [5], [3.7832], [3.7752], [1.0203],
    [10], [3.8732], [3.8431], [1.0203],
    [20], [4.0197], [3.7194], [1.0203],
    [30], [4.1711], [3.1956], [1.0135],
    [40], [4.3123], [2.1981], [1.0157],
  ),
  caption: [Influence du nombre de cellules sur la log-vraisemblance moyenne et l’intégrale numérique de l’histogramme.]
)

*La meilleure valeur empirique est ici : steps = 10*
Cette valeur correspond également au meilleur compromis observé visuellement sur les figures de densité. *Les intégrales numériques restent proches de 1 pour toutes les discrétisations testées.* L’estimateur se comporte donc bien comme une densité de probabilité.

== Choix du nombre de cellules

Pour choisir le nombre de cellules, on compare la log-vraisemblance moyenne en apprentissage et en test pour plusieurs valeurs de steps.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme1/03_hist_ll.png", width: 100%)],
    [#image("figures_tme1/04_hist_ll_night_club.png", width: 100%)],
  ),
  caption: [Log-vraisemblance moyenne en apprentissage et en test pour les bars et les boîtes de nuit.]
)

Pour les bars, la log-vraisemblance d’apprentissage augmente avec steps, car l’histogramme devient plus fin et s’adapte mieux aux données d’apprentissage. En revanche, la log-vraisemblance test atteint son maximum pour steps = 10, puis diminue. Cela indique que les grandes valeurs de steps produisent du sur-apprentissage.

Pour les boîtes de nuit, le phénomène est encore plus marqué. Comme il y a beaucoup moins de données, les grandes valeurs de steps créent de nombreuses cellules vides ou presque vides. La log-vraisemblance test chute donc très vite.

Les meilleurs hyperparamètres empiriques sont :
- steps = 10 pour les bars ;
- steps = 5 pour les boîtes de nuit.

Le meilleur nombre de cellules est donc plus petit pour les données rares.

== Estimation par noyaux (KDE)

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme1/06_kde_density_0.005.png", width: 100%)],
    [#image("figures_tme1/06_kde_density_0.02.png", width: 100%)],
  ),
  caption: [Estimation de densité estimée par KDE gaussienne pour différentes valuers de $sigma$.]
)

Un petit $sigma$ donne une densité bruitée.  
Un grand $sigma$ donne une densité trop lissée.

#figure(
  image("figures_tme1/05_kde_ll_gaussian.png", width: 70%),
  caption: [Log-vraisemblance moyenne pour la KDE gaussienne en fonct° de $sigma$.]
)

Résultats :
- meilleur $sigma$ (gaussien) = *0.003*
- meilleur $sigma$ (uniforme) = *0.02*

Le noyau gaussien produit une densité plus lisse que le noyau uniforme.
Les intégrales obtenues sont proches de 1, on obtient donc bien une densité.

== Régression par Nadaraya-Watson

#figure(
  image("figures_tme1/08_nadaraya_mse.png", width: 70%),
  caption: [Erreur quadratique moyenne de Nadaraya-Watson en fonct° de $sigma$.]
)

Résultat :
- meilleur $sigma$ = *0.01*
- MSE minimale ≈ *3.49*

== Conclusion

Le choix des hyperparamètres est crucial :

- Histogramme : nombre de cellules
- KDE : $sigma$
- Nadaraya-Watson : $sigma$

Les résultats illustrent le compromis biais-variance.  
La validation sur un ensemble de test permet de choisir un modèle qui généralise correctement.

= TME 2 : Descente de gradient

== Introduction

On étudie la descente de gradient pour deux fonctions de coût : la perte aux moindres carrés et la perte logistique. L’objectif est d’observer la convergence de l’algorithme, l’effet du pas de gradient, et les limites d’un classifieur linéaire selon la structure des données.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme2/00_frontiere_aleatoire.png", width: 100%)],
    [#image("figures_tme2/00_surface_mse.png", width: 100%)],
  ),
  caption: [Exemple de frontière de décision aléatoire et visualisation de la surface du coût MSE.]
)

La surface de coût MSE présente une structure quadratique convexe. La frontière aléatoire illustre l’effet du vecteur de poids sur la séparation des classes.

== Deux gaussiennes

On commence par tester l’algorithme sur un problème simple à deux gaussiennes. Les deux classes sont presque linéairement séparables.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme2/01_frontiere_mse_deux_gaussiennes.png", width: 100%)],
    [#image("figures_tme2/02_frontiere_logistique_deux_gaussiennes.png", width: 100%)],
  ),
  caption: [Frontières de décision obtenues avec la perte MSE et la perte logistique sur deux gaussiennes.]
)

Les deux méthodes trouvent une bonne frontière linéaire.

#figure(
  table(
    columns: 4,
    inset: 6pt,
    align: center,
    [Perte], [Coût initial], [Coût final], [Accuracy],
    [MSE], [1.000], [0.053], [1.00],
    [Logistique], [0.693], [0.016], [1.00],
  ),
  caption: [Résultats de la descente de gradient sur deux gaussiennes.]
)

La perte logistique est plus adaptée à la classification, car elle pénalise directement les erreurs via le terme $y f(x)$. Le coût diminue au cours des itérations pour les deux pertes.

#figure(
  image("figures_tme2/03_cout_deux_gaussiennes.png", width: 75%),
  caption: [Évolution du coût moyen pour la MSE et la perte logistique sur deux gaussiennes.]
)

== Effet du pas de gradient

On fait varier le pas de gradient $epsilon$ afin d’observer son effet.

#figure(
  image("figures_tme2/04_effet_pas_gradient.png", width: 75%),
  caption: [Effet du pas de gradient sur la convergence de la perte logistique.]
)

#figure(
  table(
    columns: 3,
    inset: 6pt,
    align: center,
    [$epsilon$], [Coût final], [Interprétation],
    [0.001], [0.646], [Convergence lente],
    [0.01], [0.382], [Convergence encore lente],
    [0.1], [0.067], [Convergence rapide],
    [1.0], [0.009], [Convergence très rapide ici],
  ),
  caption: [Influence du pas de gradient sur le coût final.]
)

Dans cette expérience, l’accuracy reste égale à 1.00 pour toutes les valeurs testées de $epsilon$.

Lorsque le pas est trop petit, la convergence est lente. Un pas plus grand accélère la convergence, mais peut devenir instable dans d’autres contextes.

== Cas séparable et non séparable

On compare un cas presque séparable avec un cas bruité.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme2/05_frontiere_separable.png", width: 100%)],
    [#image("figures_tme2/06_frontiere_non_separable.png", width: 100%)],
  ),
  caption: [Frontières de décision dans un cas presque séparable et dans un cas bruité non séparable.]
)

#figure(
  table(
    columns: 3,
    inset: 6pt,
    align: center,
    [Cas], [Coût final], [Accuracy],
    [Presque séparable], [0.016], [1.00],
    [Non séparable], [0.232], [0.895],
  ),
  caption: [Comparaison entre le cas presque séparable et le cas non séparable.]
)

Dans le cas non séparable, certaines observations sont forcément mal classées, car les classes se recouvrent.

#figure(
  image("figures_tme2/07_cout_separable_vs_non_separable.png", width: 75%),
  caption: [Évolution du coût moyen dans les cas presque séparable et non séparable.]
)

Le coût reste plus élevé lorsque les classes se recouvrent.

== Surface de coût et trajectoire de descente

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme2/08_surface_mse_trajectoire.png", width: 100%)],
    [#image("figures_tme2/09_surface_logistique_trajectoire.png", width: 100%)],
  ),
  caption: [Surface du coût et trajectoire des poids au cours de la descente de gradient.]
)

On visualise la fonction de coût dans l’espace des poids. La trajectoire montre l’évolution des poids au cours des itérations. Les figures sont moins lisibles pour la perte logistique, car les poids peuvent continuer à augmenter lorsque les données sont presque séparables.

== Autres types de données artificielles

On teste la régression logistique sur des données non linéaires.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme2/10_frontiere_logistique_quatre_gaussiennes.png", width: 100%)],
    [#image("figures_tme2/10_frontiere_logistique_echiquier.png", width: 100%)],
  ),
  caption: [Frontières de décision obtenues par régression logistique sur des données non linéaires.]
)

#figure(
  table(
    columns: 3,
    inset: 6pt,
    align: center,
    [Données], [Coût final], [Accuracy],
    [Quatre gaussiennes], [0.693], [0.507],
    [Échiquier], [0.693], [0.504],
  ),
  caption: [Résultats de la régression logistique sur les données non linéaires.]
)

Ces performances sont proches du hasard.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme2/11_cout_logistique_quatre_gaussiennes.png", width: 100%)],
    [#image("figures_tme2/11_cout_logistique_echiquier.png", width: 100%)],
  ),
  caption: [Évolution du coût logistique sur les données non linéaires.]
)

Les coûts restent proches de $log(2)$, ce qui correspond à une classification presque aléatoire. Cela confirme que le modèle linéaire ne peut pas capturer des frontières non linéaires comme celles des quatre gaussiennes ou de l’échiquier.

== Conclusion

La descente de gradient permet d’optimiser efficacement une fonction de coût lorsque le pas est bien choisi. La perte logistique est mieux adaptée à la classification que la MSE.

Les expériences montrent :
- l’importance du pas de gradient,
- l’impact du bruit,
- les limites d’un modèle linéaire face à des données non linéaires.

= TME 3 : Perceptron et SVM

== Introduction

On étudie plusieurs méthodes de classification binaire : le perceptron, ses variantes d'apprentissage par descente de gradient, les projections non linéaires, la perte hinge pénalisée, puis les SVM. On utilise d'abord des données artificielles en deux dimensions, puis les données USPS de chiffres manuscrits.

La perte perceptron utilisée est
$
ell(w, x, y) = max(0, - y x^T w).
$

Cette perte est nulle lorsque l'exemple est bien classé avec une marge positive, et strictement positive lorsqu'il est mal classé. Le gradient utilisé dans le code correspond à la moyenne des contributions des exemples mal classés.

== Données USPS : classification 6 contre 9

On isole d'abord deux classes, les chiffres 6 et 9. Quelques exemples des images utilisées sont représentés ci-dessous.

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

#figure(
  grid(
    rows: 2,
    gutter: 10pt,
    [#image("figures_tme3/02_loss_usps_6_vs_9.png", width: 75%)],
    [#image("figures_tme3/03_erreurs_usps_6_vs_9.png", width: 75%)],
  ),
  caption: [*En haut:* Évolution du coût moyen du perceptron sur USPS 6 contre 9. *En bas:* Erreurs d'apprentissage et de test du perceptron sur USPS 6 contre 9.]
)

L'erreur de test reste très proche de l'erreur d'apprentissage. On ne constate donc pas de sur-apprentissage marqué.

#fig("figures_tme3/04_poids_usps_6_vs_9.png", [Matrice de poids apprise par le perceptron pour séparer les chiffres 6 et 9.])

Cette matrice s'interprète comme un masque discriminant. Les pixels de poids positif favorisent la classe positive, ici le 9, tandis que les pixels de poids négatif favorisent la classe négative, ici le 6.

== Classification 6 contre toutes les autres classes


On entraîne ensuite un perceptron pour distinguer le chiffre 6 de tous les autres chiffres. Le problème devient plus difficile, car la classe négative regroupe plusieurs formes très différentes.

#table(
columns: 4,
inset: 6pt,
align: center,
[Expérience], [Score train], [Score test], [Erreur test],
[USPS 6 vs all], [0.976], [0.974], [0.026],
)

Le score reste élevé, mais l'erreur test est plus importante que dans le cas 6 contre 9. Cela confirme que le problème est plus difficile.

#figure(
  grid(
    rows: 2,
    gutter: 10pt,
    [#image("figures_tme3/05_loss_usps_6_vs_all.png", width: 75%)],
    [#image("figures_tme3/06_erreurs_usps_6_vs_all.png", width: 75%)],
  ),
  caption: [*En haut:* Évolution du coût moyen du perceptron pour la classification 6 contre toutes les autres classes. *En bas:* Erreurs d'apprentissage et de test pour la classification 6 contre toutes les autres classes.]
)

#fig("figures_tme3/07_poids_usps_6_vs_all.png", [Matrice de poids apprise pour séparer le chiffre 6 de toutes les autres classes.])

La matrice de poids est moins facile à interpréter que dans le cas 6 contre 9, car la classe négative regroupe plusieurs chiffres différents.

== Descente batch, stochastique et mini-batch

On compare ensuite trois variantes d'apprentissage.

#fig("figures_tme3/08_comparaison_batch_stochastic_minibatch.png", [Comparaison des coûts moyens pour les descentes batch, stochastique et mini-batch sur USPS 6 contre 9.])

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

On compare la vitesse de convergence en fonction du bruit dans le jeu de données. Le coût perceptron peut devenir nul très vite lorsque les données sont séparables. Pour rendre la comparaison plus lisible, on trace ici l'erreur d'apprentissage au cours des époques.

#figure(
grid(
columns: 2,
rows: 2,
gutter: 8pt,
[#image("figures_tme3/15_bruit_erreur_epsilon0.2.png", width: 100%)],
[#image("figures_tme3/15_bruit_erreur_epsilon0.6.png", width: 100%)],
[#image("figures_tme3/15_bruit_erreur_epsilon1.0.png", width: 100%)],
),
caption: [Effet du bruit sur l'erreur d'apprentissage pour les descentes batch, stochastique et mini-batch.],
)

Lorsque le bruit augmente, les classes se recouvrent davantage. L'erreur finale devient plus élevée et les trajectoires sont moins régulières. Les méthodes stochastique et mini-batch peuvent être plus bruitées, car elles utilisent moins d'exemples à chaque mise à jour.

== Projection

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

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme3/09_projection_poly_deux_gaussiennes.png", width: 100%)],
    [#image("figures_tme3/09_projection_poly_quatre_gaussiennes.png", width: 100%)],
  ),
  caption: [Frontière obtenue avec une projection polynomiale de degré 2 avec biais sur deux gaussiennes et sur quatre gaussienne.]
)

#fig("figures_tme3/09_projection_poly_echiquier.png", [Frontière obtenue avec une projection polynomiale de degré 2 avec biais sur l'échiquier.])

La projection polynomiale fonctionne très bien pour les deux gaussiennes et les quatre gaussiennes. En revanche, elle échoue sur l'échiquier : le score est proche de 0.5, donc proche du hasard. Une frontière quadratique reste insuffisante pour cette structure.

== Projection gaussienne

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

Cette perte est proche de celle utilisée dans les SVM : elle introduit une marge et une pénalisation sur les poids. La différence principale est qu'ici on optimise directement notre modèle projeté, alors que les SVM utilisent une formulation standard avec noyaux et vecteurs supports.

== SVM et Grid Search

Enfin, on utilise les SVM de `sklearn`. Les paramètres sont choisis par validation croisée avec `GridSearchCV`. Pour l'échiquier, la grille de paramètres a été élargie pour le noyau RBF, car une grille trop petite donnait des résultats trop faibles.

#fig("figures_tme3/14_svm_echiquier_vecteurs_supports.png", [Frontière de décision du meilleur SVM sur l'échiquier, avec les vecteurs supports indiqués en noir.])

*Comment évolue le nombre de vecteurs supports selon le noyau et les paramètres ?*

#figwide("figures_tme3/17_svm_2d_parametres_vecteurs_supports.png", [Comparaison des noyaux SVM sur l'échiquier : score test et nombre de vecteurs supports.])

#figwide("figures_tme3/18_svm_usps_parametres_vecteurs_supports.png", [Comparaison des noyaux SVM sur USPS 6 contre 9 : score test et nombre de vecteurs supports.])

Les vecteurs supports sont les exemples qui participent directement à la définition de la frontière. Sur l'échiquier, leur nombre est élevé, ce qui reflète la complexité de la frontière. Sur USPS 6 contre 9, certains paramètres donnent un très bon score avec peu de vecteurs supports, alors qu'un noyau trop flexible peut utiliser beaucoup plus de points.

Lorsque `gamma` devient très grand pour le noyau RBF, le modèle devient très local. Le score d'apprentissage peut alors devenir presque parfait tandis que le score test diminue : on observe du sur-apprentissage. Le nombre de vecteurs supports augmente également fortement.

Dans le cas linéaire, on retrouve une frontière proche de celle obtenue avec le perceptron. Les noyaux non linéaires, en particulier RBF, permettent d'obtenir des frontières plus flexibles.

== Conclusion

On a pu voir les limites et les extensions naturelles des modèles linéaires. Le perceptron fonctionne bien lorsque les données sont presque linéairement séparables, comme pour USPS 6 contre 9. Le cas 6 contre toutes les autres classes est plus difficile, mais reste bien traité. Les projections polynomiales permettent de résoudre certains problèmes non linéaires simples, comme les quatre gaussiennes, mais échouent sur l'échiquier. Les projections gaussiennes améliorent les résultats sur certains problèmes, mais restent limitées sur l'échiquier avec les paramètres testés. La perte hinge introduit une notion de marge et rapproche le modèle des SVM. Enfin, les SVM avec noyau RBF donnent les meilleurs résultats sur les données complexes, notamment l'échiquier et USPS 6 contre 9.

= TME 5 : Clustering spatial des points d'intérêt parisiens

== Introduction

On cherche à caractériser spatialement l'espace urbain parisien à partir des points d'intérêt présents dans différentes zones. Chaque région est représentée par un profil de types de POIs, puis ces profils sont regroupés par KMeans.

Dans le jeu de données utilisé, les types de POIs ne sont pas codés par une seule colonne catégorielle. Ils sont représentés par des colonnes indicatrices : `restaurant`, `bar`, `cafe`, `bakery`, etc. Une région est donc décrite par la moyenne de ces colonnes sur les POIs qu'elle contient.

#figure(
  image("figures_tme5/01_poi_paris.png", width: 75%),
  caption: [Répartition des points d'intérêt dans Paris.]
)

== Discrétisation manuelle par grille

On ne peut pas utiliser directement les arrondissements, car ils sont trop grands et peuvent regrouper des sous-régions très différentes. On commence donc par discrétiser l'espace avec une grille régulière de taille $N times N$. Chaque cellule est ensuite décrite par un vecteur de dimension 12. Chaque coordonnée correspond à la proportion d'un type de POI dans la cellule. On applique ensuite KMeans aux descriptions des cellules non vides.

#figure(
  table(
    columns: 2,
    inset: 6pt,
    align: center,
    [Quantité], [Valeur],
    [Taille de la grille], [20 x 20],
    [Nombre total de cellules], [400],
    [Nombre de cellules non vides], [392],
    [Proportion de cellules non vides], [0.98],
    [Nombre de clusters retenu], [6],
    [Inertie], [16.0379],
  ),
  caption: [Résumé de la discrétisation en grille.]
)

La grille contient presque uniquement des cellules non vides. La discrétisation est donc assez fine pour localiser les POIs, sans produire trop de régions vides.

#figure(
  image("figures_tme5/02_clustering_grille.png", width: 75%),
  caption: [Clustering des cellules obtenu avec une discrétisation en grille.]
)

Les cellules d'une même couleur ont des profils de POIs similaires. Cette visualisation permet de repérer des régions ayant une composition urbaine proche, même si elles ne sont pas nécessairement voisines.

== Centroïdes des clusters de la grille

Les centroïdes permettent d'interpréter les clusters. Chaque centroïde représente le profil moyen des cellules appartenant au cluster.

#figure(
  image("figures_tme5/03_centroides_grille.png", width: 85%),
  caption: [Centroïdes des clusters obtenus avec la grille.]
)

#figure(
  table(
    columns: 2,
    inset: 6pt,
    align: left,
    [Cluster], [Types dominants],
    [0], [`home_goods_store` (0.368), `restaurant` (0.259), `furniture_store` (0.250)],
    [1], [`clothing_store` (0.470), `restaurant` (0.204), `home_goods_store` (0.098)],
    [2], [`restaurant` (0.239), `lodging` (0.182), `home_goods_store` (0.157)],
    [3], [`restaurant` (0.967), `cafe` (0.100), `bar` (0.083)],
    [4], [`restaurant` (0.532), `bar` (0.180), `lodging` (0.121)],
    [5], [`restaurant` (0.363), `bar` (0.205), `lodging` (0.134)],
  ),
  caption: [Types dominants des clusters de grille.]
)

Les centroïdes montrent plusieurs profils urbains. Certains clusters sont fortement dominés par les restaurants, tandis que d'autres sont davantage associés aux commerces ou à l'hébergement. Le cluster 3 est presque entièrement dominé par les restaurants, alors que le cluster 1 est surtout associé aux magasins de vêtements.

== Choix du nombre de clusters : méthode elbow

Le score elbow correspond à l'inertie du modèle KMeans. L'inertie mesure la somme des distances quadratiques entre les points et leur centroïde. Elle diminue nécessairement lorsque le nombre de clusters augmente.

#figure(
  image("figures_tme5/04_elbow_grille.png", width: 75%),
  caption: [Courbe elbow pour le clustering des cellules de la grille.]
)

Le coude n'est pas parfaitement net, mais un choix autour de $K = 5$ ou $K = 6$ semble raisonnable. On retients $K = 6$ pour obtenir une description assez fine tout en gardant des clusters interprétables.

== Limites de la discrétisation en grille

La grille est simple à construire et à interpréter, mais elle reste arbitraire. Certaines cellules peuvent couper artificiellement des quartiers cohérents ou mélanger des zones différentes dans une même cellule.

Cette méthode donne donc une première approximation, mais elle ne tient pas compte de la densité réelle des POIs.

== Discrétisation automatique de l'espace

Pour obtenir des régions plus adaptées aux données, on applique KMeans directement aux coordonnées GPS des POIs. Cette étape correspond à une quantization spatiale. Une grille $10 times 10$ contient 100 cellules ; on peut donc comparer cette approche à une quantization avec $K_"geo" = 100$ clusters.

#figure(
  table(
    columns: 2,
    inset: 6pt,
    align: center,
    [Quantité], [Valeur],
    [Nombre de régions spatiales], [100],
    [Inertie spatiale], [0.933682],
    [Taille minimale des régions], [111],
    [Taille maximale des régions], [576],
    [Taille moyenne des régions], [318.52],
  ),
  caption: [Résumé de la quantization spatiale.]
)

#figure(
  image("figures_tme5/05_quantization_spatiale.png", width: 75%),
  caption: [Quantization spatiale des POIs par KMeans.]
)

Contrairement à la grille régulière, cette discrétisation s'adapte à la densité des données. Les régions sont plus petites dans les zones riches en POIs et plus grandes dans les zones moins denses.

== Clustering des régions spatiales

Une fois les clusters spatiaux obtenus, chaque région spatiale est décrite par la distribution moyenne de ses types de POIs. On applique ensuite un second KMeans à ces descriptions.

#figure(
  image("figures_tme5/06_clustering_regions_spatiales.png", width: 75%),
  caption: [Clustering des régions spatiales selon les types de POIs.]
)

Cette carte est plus adaptée que celle obtenue avec la grille. Les régions sont moins arbitraires et suivent davantage la géométrie réelle des points d'intérêt.

#figure(
  image("figures_tme5/07_centroides_regions_spatiales.png", width: 85%),
  caption: [Centroïdes des clusters obtenus après quantization spatiale.]
)

#figure(
  table(
    columns: 2,
    inset: 6pt,
    align: left,
    [Cluster], [Types dominants],
    [0], [`restaurant` (0.365), `bar` (0.191), `lodging` (0.166)],
    [1], [`restaurant` (0.249), `lodging` (0.177), `bar` (0.172)],
    [2], [`clothing_store` (0.439), `restaurant` (0.200), `home_goods_store` (0.140)],
    [3], [`restaurant` (0.292), `clothing_store` (0.222), `bar` (0.150)],
    [4], [`restaurant` (0.400), `bar` (0.155), `home_goods_store` (0.152)],
    [5], [`home_goods_store` (0.246), `restaurant` (0.235), `clothing_store` (0.194)],
  ),
  caption: [Types dominants des clusters de régions spatiales.]
)

Les centroïdes des régions spatiales sont plus équilibrés que ceux obtenus avec la grille. On retrouve des profils dominés par les restaurants et les bars, mais aussi des profils davantage associés aux commerces ou aux hébergements.

== Choix du nombre de clusters pour les régions spatiales

On trace de nouveau la courbe elbow, cette fois pour le clustering des régions spatiales décrites par leurs types de POIs.

#figure(
  image("figures_tme5/08_elbow_regions_spatiales.png", width: 75%),
  caption: [Courbe elbow pour les régions spatiales.]
)

Ici aussi, le coude n'est pas parfaitement net, mais $K = 5$ ou $K = 6$ semble raisonnable. On conserve $K = 6$ pour pouvoir comparer avec la discrétisation en grille.

== Comparaison entre grille et quantization spatiale

La discrétisation en grille est simple, mais elle impose une structure régulière qui ne correspond pas nécessairement à la densité des POIs. La quantization spatiale est plus adaptée, car elle regroupe les POIs selon leur proximité géographique.

Ainsi, la grille est plus facile à expliquer, mais la quantization spatiale donne des régions plus pertinentes géographiquement. Il est donc utile de montrer les deux : la grille comme méthode de référence, puis la quantization spatiale comme amélioration.

== Corrélation entre types de POIs

On observe ensuite les corrélations entre types de POIs à partir des descriptions régionales.

#figure(
  image("figures_tme5/09_correlation_types.png", width: 75%),
  caption: [Matrice de corrélation entre types de POIs.]
)

#figure(
  table(
    columns: 2,
    inset: 6pt,
    align: left,
    [Paire de types], [Corrélation],
    [`furniture_store` / `home_goods_store`], [0.926],
    [`cafe` / `night_club`], [0.595],
    [`clothing_store` / `bar`], [-0.522],
    [`bakery` / `clothing_store`], [-0.521],
    [`clothing_store` / `restaurant`], [-0.506],
    [`laundry` / `lodging`], [0.501],
    [`cafe` / `restaurant`], [-0.428],
    [`atm` / `night_club`], [0.416],
  ),
  caption: [Corrélations les plus fortes entre types de POIs.]
)

Certaines corrélations sont très fortes. Les magasins de meubles et les magasins d'équipement de maison apparaissent souvent ensemble. Les cafés et les boîtes de nuit sont également positivement corrélés. À l'inverse, les zones dominées par les magasins de vêtements sont moins associées aux bars ou aux restaurants.

Ces corrélations peuvent influencer KMeans, car des variables redondantes peuvent compter plusieurs fois dans la distance euclidienne.

== Traitement proposé pour améliorer les résultats

Un traitement simple consiste à standardiser les colonnes avant le clustering. Cela donne un poids comparable à chaque type de POI et évite que les catégories les plus fréquentes dominent complètement la distance.

#figure(
  image("figures_tme5/10_clustering_standardise.png", width: 75%),
  caption: [Clustering obtenu après standardisation des descriptions de types.]
)

Après standardisation, les types rares ont davantage d'influence dans le calcul des distances. Les clusters obtenus peuvent donc différer de ceux obtenus sur les proportions brutes.

L'inertie avant standardisation vaut 0.9412, tandis que l'inertie après standardisation vaut 631.9848. Ces valeurs ne sont pas directement comparables, car les échelles des variables ont changé.

== Conclusion

Ce TME montre comment caractériser l'espace urbain parisien à partir des distributions de POIs. La discrétisation en grille fournit une première approche simple, mais assez arbitraire. La quantization spatiale par KMeans donne une partition plus adaptée à la densité réelle des points.

L'interprétation repose sur trois éléments complémentaires : les cartes, les centroïdes et les courbes elbow. Les cartes montrent l'organisation spatiale des clusters, les centroïdes expliquent leur composition, et les courbes elbow aident à choisir un nombre raisonnable de clusters.

Les clusters obtenus reflètent différents profils urbains parisiens : zones commerciales, zones dominées par les restaurants et les bars, régions plus liées aux hébergements ou aux commerces spécialisés.

Enfin, l'analyse des corrélations entre types montre qu'un prétraitement peut être utile. La standardisation est une première solution pour limiter la domination des types les plus fréquents et rendre le clustering plus équilibré.