# rag-tutorial-v2

Sur ce repository, vous trouverez le code source de l'application de tutoriel de RAG v2 ansi qu'un petit exemple de tokenisation

## Le RAG

### Prérequis

- Python 3.10
- Ollama installé et disponible

### Installation

1. Clonez le dépôt

2. Rendez-vous dans le répertoire de Django :
    ```bash
    cd server
    ```

3. Créez un environnement virtuel et activez-le :
    ```bash
    python3.10 -m venv env
    source env/bin/activate
    ```

4. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

5. Assurez-vous qu'Ollama est installé et disponible sur votre système.

### Lancer l'application

1. Appliquez les migrations de la base de données :
    ```bash
    python manage.py migrate
    ```

2. Lancez le serveur de développement :
    ```bash
    python manage.py runserver
    ```

3. Accédez à l'application via votre navigateur à l'adresse `http://127.0.0.1:8000`.

## Tokenisation

### Prérequis

- Python 3.10

### Installation

1. Se déplacer dans le répertoire de tokenisation :
    ```bash
    cd tokenization
    ```

2. Créez un environnement virtuel et activez-le :
    ```bash
    python3.10 -m venv env
    source env/bin/activate
    ```

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

### Lancer l'application

1. Lancez le script de tokenisation :
    ```bash
    python tokenisation_openai.py
    ```
