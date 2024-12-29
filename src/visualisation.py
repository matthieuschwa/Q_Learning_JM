from stable_baselines3 import DQN
from src.env import GameEnv

def test_agent_obs():
    """
    Test de l'agent entrainé sur 1 seul scénario afin de voir le chemin qu'il a emprunté jusqu'au trésor.

    Fonctionnement :
    - Réinitialise l'environnement pour commencer un nouvel épisode.
    - Utilise le modèle chargé pour prédire les actions optimales à chaque étape.
    - Affiche l'état de l'environnement après chaque action via la méthode `render()`.

    L'épisode se termine lorsqu'une condition d'arrêt définie dans l'environnement est atteinte.

    Assurez-vous que le fichier "src/models/hero_agent.zip" existe avant d'exécuter cette fonction.
    S'il n'existe pas, commencez par entrainer le modèle.
    """

    env = GameEnv()
    model = DQN.load("src/models/hero_agent")
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True) #deterministic pour que l'agent agisse selon la politique apprise exclusivement
        obs, _, done, _ = env.step(action)
        env.render()