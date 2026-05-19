#set page(margin: 1.8cm)
#set text(size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true)

#let fig(path, caption) = figure(
  image(path, width: 75%),
  caption: caption,
)

#let figwide(path, caption) = figure(
  image(path, width: 95%),
  caption: caption,
)

= TME 3 — Perceptron et SVM

== Introduction

L'objectif de ce TME est d'étudier plusieurs méthodes de classification binaire : le perceptron, ses variantes d'apprentissage par descente de gradient, les projections non linéaires, la perte hinge pénalisée, puis les SVM. Nous utilisons d'abord des données artificielles en deux dimensions, puis les données USPS de chiffres manuscrits.

Dans tout le rapport, les labels sont encodés en $-1$ et $+1$. Le classifieur linéaire prédit donc le signe de $x^T w$.

== Perceptron

La perte perceptron utilisée est
$
  ell(w, x, y) = max(0, - y x^T w).
$

Cette perte est nulle lorsque l'exemple est bien classé avec une marge positive, et strictement positive lorsqu'il est mal classé. Le gradient utilisé dans le code correspond à la moyenne des contributions des exemples mal classés.

Le modèle est implémenté dans une classe `Lineaire`. Cette classe permet de changer la fonction de coût, le gradient, le nombre d'itérations, le pas de gradient, la projection utilisée, ainsi que le type d'apprentissage : batch complet, stochastique ou mini-batch.

== Données USPS : classification 6 contre 9

Nous isolons d'abord deux classes, les chiffres 6 et 9. Quelques exemples des images utilisées sont représentés ci-dessous.

#fig("figures_tme3/01_exemples_usps_6_vs_9.png", [Exemples USPS pour les classes 6 et 9.])

Le perceptron est ensuite entraîné sur ces deux classes. Les résultats obtenus sont les suivants.

#table(
  columns: 4,
  inset: 6pt,
  align: center,
  [Expérience], [Score train], [Score test], [Erreur test],
  [USPS 6 vs 9], [0.998], [0.991], [0.009],
)

Le coût final est très faible, environ 0.0001. Le modèle distingue donc très bien les deux chiffres.

#fig("figures_tme3/02_loss_usps_6_vs_9.png", [Évolution du coût du perceptron sur USPS 6 vs 9.])

#fig("figures_tme3/03_erreurs_usps_6_vs_9.png", [Erreurs train et test pour USPS 6 vs 9.])

L'erreur de test reste très proche de l'erreur d'apprentissage. On ne constate donc pas de sur-apprentissage marqué.

#fig("figures_tme3/04_poids_usps_6_vs_9.png", [Matrice de poids du perceptron pour USPS 6 vs 9.])

Cette matrice s'interprète comme un masque discriminant. Les pixels de poids positif favorisent la classe positive, ici le 9, tandis que les pixels de poids négatif favorisent la classe négative, ici le 6.

== Classification 6 contre toutes les autres classes

On entraîne ensuite un perceptron pour séparer le chiffre 6 de tous les autres chiffres. Le problème est plus difficile, car la classe négative est très hétérogène.

#table(
  columns: 4,
  inset: 6pt,
  align: center,
  [Expérience], [Score train], [Score test], [Erreur test],
  [USPS 6 vs all], [0.980], [0.969], [0.031],
)

Le score reste élevé, mais l'erreur test est plus importante que dans le cas 6 contre 9. Cela confirme que le problème est plus difficile.

#fig("figures_tme3/05_loss_usps_6_vs_all.png", [Évolution du coût pour USPS 6 vs all.])

#fig("figures_tme3/06_erreurs_usps_6_vs_all.png", [Erreurs train et test pour USPS 6 vs all.])

#fig("figures_tme3/07_poids_usps_6_vs_all.png", [Matrice de poids pour USPS 6 vs all.])

La matrice de poids est moins facile à interpréter que dans le cas 6 contre 9, car la classe négative regroupe plusieurs chiffres différents.

== Descente batch, stochastique et mini-batch

Nous comparons ensuite trois variantes d'apprentissage.

#figwide("figures_tme3/08_comparaison_batch_stochastic_minibatch.png", [Comparaison des descentes batch, stochastique et mini-batch.])

#table(
  columns: 4,
  inset: 6pt,
  align: center,
  [Méthode], [Coût final], [Score train], [Score test],
  [Batch complet], [0.0001], [0.995], [0.988],
  [Stochastique], [0.0000], [1.000], [0.997],
  [Mini-batch 32], [0.0000], [1.000], [1.000],
)

Les trois méthodes convergent bien sur ce problème. Le mini-batch donne ici le meilleur score test. La descente stochastique et la descente mini-batch progressent vite, car elles effectuent plus de mises à jour pendant une époque.

== Projection polynomiale

Un modèle linéaire dans l'espace initial ne peut produire qu'une frontière linéaire. Pour augmenter son expressivité, on projette les données dans un espace de plus grande dimension.

$
  1, x_1, x_2, dots, x_d, x_1^2, x_1 x_2, dots, x_d^2.
$

#table(
  columns: 3,
  inset: 6pt,
  align: center,
  [Jeu de données], [Score train], [Erreur train],
  [Deux gaussiennes], [1.000], [0.000],
  [Quatre gaussiennes], [0.998], [0.002],
  [Échiquier], [0.505], [0.495],
)

#fig("figures_tme3/09_projection_poly_deux_gaussiennes.png", [Projection polynomiale sur deux gaussiennes.])

#fig("figures_tme3/09_projection_poly_quatre_gaussiennes.png", [Projection polynomiale sur quatre gaussiennes.])

#fig("figures_tme3/09_projection_poly_echiquier.png", [Projection polynomiale sur l'échiquier.])

La projection polynomiale fonctionne très bien pour les deux gaussiennes et les quatre gaussiennes. En revanche, elle échoue sur l'échiquier : le score est proche de 0.5, donc proche du hasard. Une frontière quadratique reste insuffisante pour cette structure.

== Projection gaussienne

Nous utilisons ensuite une projection gaussienne sur des points de base $b_1, dots, b_m$ :
$
  phi(x) = ( exp(- ||x - b_1||^2 / (2 sigma^2)), dots, exp(- ||x - b_m||^2 / (2 sigma^2)) ).
$

#table(
  columns: 4,
  inset: 6pt,
  align: center,
  [Nombre de points de base], [$sigma$], [Score train], [Erreur train],
  [20], [0.2], [0.525], [0.475],
  [20], [1.0], [0.533], [0.467],
  [80], [0.2], [0.619], [0.381],
  [80], [1.0], [0.584], [0.416],
)

#fig("figures_tme3/11_projection_gaussienne_type2_base20_sigma0.2.png", [Projection gaussienne avec peu de points de base et petit sigma.])

#fig("figures_tme3/11_projection_gaussienne_type2_base20_sigma1.0.png", [Projection gaussienne avec peu de points de base et grand sigma.])

#fig("figures_tme3/11_projection_gaussienne_type2_base80_sigma0.2.png", [Projection gaussienne avec beaucoup de points de base et petit sigma.])

#fig("figures_tme3/11_projection_gaussienne_type2_base80_sigma1.0.png", [Projection gaussienne avec beaucoup de points de base et grand sigma.])

La meilleure configuration testée ici est `nb_base = 80` et $sigma = 0.2$, avec un score train de 0.619. L’augmentation du nombre de points de base améliore l’expressivité du modèle. Les performances restent cependant limitées sur l’échiquier.

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
  [0.5], [1e-4], [0.644], [0.3722],
  [1.0], [1e-4], [0.635], [0.7857],
  [2.0], [1e-4], [0.617], [1.6716],
  [1.0], [1e-2], [0.599], [0.9602],
  [1.0], [1e-1], [0.598], [0.9960],
)

#fig("figures_tme3/13_hinge_alpha0.5_lambda0.0001.png", [Perte hinge avec alpha=0.5 et lambda=1e-4.])

#fig("figures_tme3/13_hinge_alpha1.0_lambda0.0001.png", [Perte hinge avec alpha=1.0 et lambda=1e-4.])

#fig("figures_tme3/13_hinge_alpha2.0_lambda0.0001.png", [Perte hinge avec alpha=2.0 et lambda=1e-4.])

#fig("figures_tme3/13_hinge_alpha1.0_lambda0.1.png", [Perte hinge avec alpha=1.0 et lambda=1e-1.])

Lorsque $alpha$ augmente, le coût final augmente car la marge demandée est plus grande. Lorsque $lambda$ augmente, les poids sont davantage pénalisés, ce qui peut entraîner du sous-apprentissage. Ici, le meilleur score parmi les configurations testées est obtenu avec $alpha = 0.5$ et $lambda = 10^(-4)$.

== SVM et Grid Search

Enfin, nous utilisons les SVM de `sklearn`. Les paramètres sont choisis par validation croisée avec `GridSearchCV`.

#table(
  columns: 5,
  inset: 6pt,
  align: center,
  [Données], [Meilleur noyau], [Meilleurs paramètres], [Score test], [Nb vecteurs supports],
  [Échiquier], [RBF], [`C=1, gamma=10`], [0.807], [550],
  [USPS 6 vs 9], [RBF], [`C=1, gamma=0.01`], [0.997], [150],
)

#fig("figures_tme3/14_svm_echiquier_vecteurs_supports.png", [SVM sur l'échiquier avec vecteurs supports.])

Sur l'échiquier, le SVM RBF obtient un score test de 0.807, nettement supérieur aux projections précédentes. Sur USPS 6 contre 9, le score test atteint 0.997, ce qui est légèrement meilleur que le perceptron simple.

Les vecteurs supports sont les exemples qui participent directement à la définition de la frontière. Sur l'échiquier, leur nombre est élevé, ce qui reflète la complexité de la frontière.

== Conclusion

Ce TME met en évidence les limites et les extensions naturelles des modèles linéaires. Le perceptron fonctionne bien lorsque les données sont presque linéairement séparables, comme pour USPS 6 contre 9. Le cas 6 contre toutes les autres classes est plus difficile, mais reste bien traité.

Les projections polynomiales permettent de résoudre certains problèmes non linéaires simples, comme les quatre gaussiennes, mais échouent sur l'échiquier. Les projections gaussiennes améliorent légèrement les résultats, mais restent limitées avec les paramètres testés. La perte hinge introduit une notion de marge et rapproche le modèle des SVM.

Enfin, les SVM avec noyau RBF donnent les meilleurs résultats sur les données complexes, notamment l'échiquier et USPS 6 contre 9.