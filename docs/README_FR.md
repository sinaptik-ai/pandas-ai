# ![PandaAI](../assets/logo.png)

<p align="center">
[ <a href="../README.md">En</a> |
<a href="README_CN.md">中</a> |
<b>Fr</b> |
<a href="README_JA.md">日</a> ] 
</p>

[![Release](https://img.shields.io/pypi/v/pandasai?label=Release&style=flat-square)](https://pypi.org/project/pandasai/)
[![CI](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/ci-core.yml/badge.svg)](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/ci-core.yml/badge.svg)
[![CD](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/cd.yml/badge.svg)](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/cd.yml/badge.svg)
[![Coverage](https://codecov.io/gh/sinaptik-ai/pandas-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/sinaptik-ai/pandas-ai)
[![Discord](https://dcbadge.vercel.app/api/server/kF7FqH2FwS?style=flat&compact=true)](https://discord.gg/KYKj9F2FRH)
[![Downloads](https://static.pepy.tech/badge/pandasai)](https://pepy.tech/project/pandasai) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZnO-njhL7TBOYPZaqvMvGtsjckZKrv2E?usp=sharing)

PandaAI est une plateforme Python qui permet de poser des questions à vos données en langage naturel. Elle aide les utilisateurs non techniques à interagir avec leurs données de manière plus naturelle, et elle aide les utilisateurs techniques à gagner du temps et des efforts lorsqu'ils travaillent avec des données.

# 🔧 Premiers pas

Vous pouvez trouver la documentation complète de PandaAI [ici](https://pandas-ai.readthedocs.io/en/latest/).

Vous pouvez choisir d'utiliser PandaAI dans vos notebooks Jupyter, vos applications Streamlit, ou utiliser l'architecture client-serveur du dépôt.

## ☁️ Utilisation de la plateforme

La bibliothèque peut être utilisée avec notre puissante plateforme de données, permettant une analyse de données conversationnelle de bout en bout avec seulement quelques lignes de code.

Chargez vos données, enregistrez-les sous forme de dataframe, et poussez-les vers la plateforme

```python
import pandasai as pai

pai.api_key.set("votre-clé-api-pai")

file = pai.read_csv("./chemin/du/fichier.csv")

dataset = pai.create(path="votre-organisation/nom-du-dataset",
    df=file,
    name="nom-du-dataset",
    description="description-du-dataset")

dataset.push()
```

Votre équipe peut maintenant accéder et interroger ces données en langage naturel via la plateforme.

![PandaAI](assets/demo.gif)

## 📚 Utilisation de la bibliothèque

### Prérequis Python

Version de Python `3.8+ <3.12`

### 📦 Installation

Vous pouvez installer la bibliothèque PandaAI en utilisant pip ou poetry.

Avec pip :

```bash
pip install "pandasai>=3.0.0b2"
```

Avec poetry :

```bash
poetry add "pandasai>=3.0.0b2"
```

### 💻 Utilisation

#### Poser des questions

```python
import pandasai as pai

# Exemple de DataFrame
df = pai.DataFrame({
    "country": ["États-Unis", "Royaume-Uni", "France", "Allemagne", "Italie", "Espagne", "Canada", "Australie", "Japon", "Chine"],
    "revenue": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})

# Par défaut, à moins que vous ne choisissiez un autre LLM, il utilisera BambooLLM.
# Vous pouvez obtenir votre clé API gratuite en vous inscrivant sur https://app.pandabi.ai (vous pouvez également la configurer dans votre fichier .env)
pai.api_key.set("votre-clé-api-pai")

df.chat('Quels sont les 5 premiers pays par ventes ?')
```

```
Chine, États-Unis, Japon, Allemagne, Australie
```

---

Ou vous pouvez poser des questions plus complexes :

```python
df.chat(
    "Quel est le total des ventes pour les 3 premiers pays par ventes ?"
)
```

```
Le total des ventes pour les 3 premiers pays par ventes est de 16500.
```

#### Visualiser des graphiques

Vous pouvez également demander à PandaAI de générer des graphiques pour vous :

```python
df.chat(
    "Tracez l'histogramme des pays en montrant pour chacun le PIB. Utilisez des couleurs différentes pour chaque barre",
)
```

![Graphique](../assets/histogram-chart.png?raw=true)

#### Plusieurs DataFrames

Vous pouvez également passer plusieurs dataframes à PandaAI et poser des questions les reliant.

```python
import pandasai as pai

employees_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],
    'Department': ['RH', 'Ventes', 'IT', 'Marketing', 'Finance']
}

salaries_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Salary': [5000, 6000, 4500, 7000, 5500]
}

employees_df = pai.DataFrame(employees_data)
salaries_df = pai.DataFrame(salaries_data)

# Par défaut, à moins que vous ne choisissiez un autre LLM, il utilisera BambooLLM.
# Vous pouvez obtenir votre clé API gratuite en vous inscrivant sur https://app.pandabi.ai (vous pouvez également la configurer dans votre fichier .env)
pai.api_key.set("votre-clé-api-pai")

pai.chat("Qui est le mieux payé ?", employees_df, salaries_df)
```

```
Olivia est la mieux payée.
```

#### Sandbox Docker

Vous pouvez exécuter PandaAI dans un sandbox Docker, fournissant un environnement sécurisé et isolé pour exécuter du code en toute sécurité et atténuer le risque d'attaques malveillantes.

##### Prérequis Python

```bash
pip install "pandasai-docker"
```

##### Utilisation

```python
import pandasai as pai
from pandasai_docker import DockerSandbox

# Initialiser le sandbox
sandbox = DockerSandbox()
sandbox.start()

employees_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],
    'Department': ['RH', 'Ventes', 'IT', 'Marketing', 'Finance']
}

salaries_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Salary': [5000, 6000, 4500, 7000, 5500]
}

employees_df = pai.DataFrame(employees_data)
salaries_df = pai.DataFrame(salaries_data)

# Par défaut, à moins que vous ne choisissiez un autre LLM, il utilisera BambooLLM.
# Vous pouvez obtenir votre clé API gratuite en vous inscrivant sur https://app.pandabi.ai (vous pouvez également la configurer dans votre fichier .env)
pai.api_key.set("votre-clé-api-pai")

pai.chat("Qui est le mieux payé ?", employees_df, salaries_df, sandbox=sandbox)

# N'oubliez pas d'arrêter le sandbox une fois terminé
sandbox.stop()
```

```
Olivia est la mieux payée.
```

Vous pouvez trouver plus d'exemples dans le répertoire [examples](examples).

## 📜 Licence

PandaAI est disponible sous la licence MIT expat, à l'exception du répertoire `pandasai/ee` de ce dépôt, qui a sa [licence ici](https://github.com/sinaptik-ai/pandas-ai/blob/main/ee/LICENSE).

Si vous êtes intéressé par PandaAI Cloud géré ou l'offre Enterprise auto-hébergée, [contactez-nous](https://getpanda.ai/pricing).

## Ressources

> **Avis Beta**  
> La version v3 est actuellement en version bêta. La documentation et les exemples suivants reflètent les fonctionnalités en cours de développement et peuvent changer avant la version finale.

- [Docs](https://pandas-ai.readthedocs.io/en/latest/) pour la documentation complète
- [Exemples](examples) pour les notebooks d'exemples
- [Discord](https://discord.gg/KYKj9F2FRH) pour discuter avec la communauté et l'équipe PandaAI

## 🤝 Contributions

Les contributions sont les bienvenues ! Veuillez consulter les problèmes en suspens et n'hésitez pas à ouvrir une pull request.
Pour plus d'informations, veuillez consulter les [lignes directrices de contribution](CONTRIBUTING.md).

### Merci !

[![Contributeurs](https://contrib.rocks/image?repo=sinaptik-ai/pandas-ai)](https://github.com/sinaptik-ai/pandas-ai/graphs/contributors)