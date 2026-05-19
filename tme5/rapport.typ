#set page(margin: 1.8cm)
#set text(size: 11pt)
#set par(justify: true)

#let fig(path, caption) = {
  figure(
    image(path, width: 50%),
    caption: caption,
  )
}

= TME 5 — Clustering spatial des points d'intérêt parisiens

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