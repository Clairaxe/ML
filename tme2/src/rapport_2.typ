= TME 2 — Descente de gradient

== Objectif

Ce TME étudie la descente de gradient pour deux fonctions de coût : la perte aux moindres carrés et la perte logistique. L’objectif est d’observer la convergence de l’algorithme, l’effet du pas de gradient, et les limites d’un classifieur linéaire selon la structure des données.

== Fonctions de coût et gradients

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

Dans tous les cas, l’accuracy reste égale à 1.00.

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
  caption: [Évolution du coût.],
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
  caption: [Surface de coût et trajectoire suivie par la descente de gradient.],
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
  caption: [Frontières obtenues sur des données non linéaires.],
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