= TME 1 — Estimation de densité

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

On observe une forte concentration dans le centre de Paris, cohérente avec l’activité commerciale et touristique.

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
- meilleur steps (test) = **10** pour les bars

== Vérification de la densité

On vérifie que l’intégrale de la densité est proche de 1 :

- intégrale ≈ **1.02**

Cela confirme que l’estimateur est bien une densité.

== Choix du nombre de cellules

#figure(
  image("figures_tme1/03_hist_ll.png", width: 75%),
  caption: [Log-vraisemblance en fonction du nombre de cellules.]
)

La log-vraisemblance test est maximale pour :
- **steps = 10** (bars)

== Cas des données rares : boîtes de nuit

Les boîtes de nuit sont moins nombreuses.

- meilleur steps = **5**

→ Une discrétisation plus grossière est nécessaire pour éviter des cellules vides.

== Estimation par noyaux (KDE)

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [#image("figures_tme1/05_kde_density_0.005.png", width: 100%)],
    [#image("figures_tme1/05_kde_density_0.02.png", width: 100%)],
  ),
  caption: [Densité estimée par KDE pour différents σ.]
)

Un petit σ donne une densité bruitée.  
Un grand σ donne une densité trop lissée.

#figure(
  image("figures_tme1/04_kde_ll.png", width: 70%),
  caption: [Log-vraisemblance pour la KDE.]
)

Résultats :
- meilleur σ (gaussien) = **0.003**
- meilleur σ (uniforme) = **0.02**

Le noyau gaussien produit une densité plus lisse que le noyau uniforme.

== Régression par Nadaraya–Watson

#figure(
  image("figures_tme1/06_nadaraya_mse.png", width: 70%),
  caption: [Erreur quadratique moyenne selon σ.]
)

Résultat :
- meilleur σ = **0.01**
- MSE minimale ≈ **3.49**

→ Petit σ : sur-apprentissage  
→ Grand σ : sous-apprentissage

== Conclusion

Le choix des hyperparamètres est crucial :

- Histogramme : nombre de cellules
- KDE : σ
- Nadaraya-Watson : σ

Les résultats illustrent le compromis biais–variance.  
La validation sur un ensemble de test permet de choisir un modèle qui généralise correctement.