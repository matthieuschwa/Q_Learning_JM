from stable_baselines3 import DQN
from src.env import GameEnv
from src.callbacks import RewardTrackerCallback
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
      """
      Calcule la moyenne mobile d'un tableau de données.

      Args:
            data (array-like): Les données à lisser.
            window_size (int): La taille de la fenêtre utilisée pour calculer la moyenne.

      Returns:
            numpy.ndarray: Les données lissées en utilisant une moyenne mobile.
      """
      return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train_agent():
      """
      Entraîne un agent sur l'environnement GameEnv et sauvegarde le modèle entraîné.

      Fonctionnement :
      - Entraine l'agent
      - Suit les métriques d'entraînement à l'aide de RewardTrackerCallback.
      - Sauvegarde le modèle entraîné dans un fichier.
      - Génère des graphiques pour visualiser les performances de l'agent pendant l'entraînement.

      !! Important !!
      Le modèle sauvegardé écrasera tout fichier existant portant le même nom.
      """

      env = GameEnv()
      model = DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.8, exploration_final_eps=0.2) #mlp pour Multilayer perceptron

      reward_callback = RewardTrackerCallback() #pour suivre les performances

      model.learn(total_timesteps=200000, callback=reward_callback) #entrainement jusqu'à 200000 unités de temps écoulées
      model.save("src/models/hero_agent") #enregistre le modèle ou remplace le modèle entrainé précédemment

      episode_rewards = np.array(reward_callback.episode_rewards).flatten()
      smoothed_rewards = moving_average(episode_rewards, window_size=50)

      #plot des rewards
      plt.figure(figsize=(10, 6))
      plt.plot(reward_callback.episode_rewards, label="Total Reward per Episode")
      plt.plot(smoothed_rewards, label=f"Moving Average (window=50)")
      plt.xlabel("Episode")
      plt.ylabel("Total Reward")
      plt.title("Training Progress")
      plt.legend()
      plt.show()

      #plot des episodes lengths
      plt.figure(figsize=(10, 6))
      plt.plot(reward_callback.episode_lengths, label="Episode Length")
      plt.xlabel("Episode")
      plt.ylabel("Steps")
      plt.title("Episode Lengths")
      plt.legend()
      plt.show()

      #plot du success rate
      plt.figure(figsize=(10, 6))
      plt.plot(reward_callback.success_rate, label="Success Rate")
      plt.xlabel("Episode")
      plt.ylabel("Success Rate")
      plt.title("Success Rate")
      plt.legend()
      plt.show()

      ########## non utilisé
      # Exploration VS Exploitation

      # exploration_smooth = moving_average(reward_callback.episode_exploration, window_size=50)
      # exploitation_smooth = moving_average(reward_callback.episode_exploitation, window_size=50)

      # plt.figure(figsize=(10, 6))
      # plt.plot(exploration_smooth, label="Exploration Ratio (smoothed)", color="blue")
      # plt.plot(exploitation_smooth, label="Exploitation Ratio (smoothed)", color="green")
      # plt.xlabel("Episode")
      # plt.ylabel("Ratio")
      # plt.title("Exploration vs Exploitation During Training (Smoothed)")
      # plt.legend()
      # plt.show()
