from stable_baselines3 import DQN
from src.env import GameEnv
import numpy as np

def test_agent():
    """
    Teste les performances d'un agent entraîné sur l'environnement.

    Notes :
    - Le fichier du modèle (par défaut "src/models/hero_agent.zip") doit exister.
    - L'environnement GameEnv doit être correctement configuré pour indiquer les succès via `info["is_success"]`.

    Returns:
        Aucun retour direct, mais affiche les statistiques des performances de l'agent :
        - Récompense moyenne sur tous les épisodes testés.
        - Taux de succès (en pourcentage).

    """
    env = GameEnv() 

    model = DQN.load("src/models/hero_agent")

    #variables pour suivre les performances
    num_episodes = 100  #nb d'épisodes à tester
    total_rewards = []
    successes = 0

    for episode in range(num_episodes):
        obs = env.reset()  #réinitialiser à chaque fois l'environnement
        done = False
        episode_reward = 0

        while not done:
            #prediction de l'action qu'il va prendre
            action, _ = model.predict(obs, deterministic=True)
            
            #appliquer l'action et recupérer la valeur de la récompense
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        #compteur : si l'épisode est un succes ou non
        is_success = info.get("is_success", False)
        if is_success:
            successes += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {'Yes' if is_success else 'No'}")

    #résumé des perf
    print(f"\nTest terminé : {num_episodes} épisodes")
    print(f"Récompense moyenne : {np.mean(total_rewards):.2f}")
    print(f"Taux de succès : {successes / num_episodes * 100:.2f}%")
