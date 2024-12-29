from stable_baselines3.common.callbacks import BaseCallback

class RewardTrackerCallback(BaseCallback):
    """
    Callback personnalisé pour suivre les performances d'un agent pendant l'entraînement.

    Métriques retenues : 
    - Les récompenses cumulées par épisode.
    - La longueur des épisodes (en nombre de pas).
    - Le taux de succès (si une condition de réussite est définie dans l'environnement).

    Notes :
    - La collecte des données se fait à chaque étape via la méthode `_on_step`.
    - Les métriques sont automatiquement réinitialisées à la fin de chaque épisode.

    Attributs:
        episode_rewards (list): Récompenses cumulées pour chaque épisode.
        episode_lengths (list): Nombre de pas par épisode.
        success_rate (list): Taux de succès cumulé par épisode.
        current_rewards (float): Récompense cumulée pour l'épisode en cours.
        current_length (int): Longueur actuelle de l'épisode (en pas).
        success_count (int): Nombre total de succès obtenus.
        exploration_actions (int): Compteur d'actions exploratoires.
        exploitation_actions (int): Compteur d'actions exploitatives.
        episode_exploration (list): Ratio d'exploration par épisode.
        episode_exploitation (list): Ratio d'exploitation par épisode.
    """

    def __init__(self, verbose=0):
        super(RewardTrackerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.current_rewards = 0
        self.current_length = 0
        self.success_count = 0

        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.episode_exploration = []
        self.episode_exploitation = []

    def _on_step(self) -> bool:
        """
        - Incrémente les récompenses et le compteur de pas pour l'épisode en cours.
        - Finalise les métriques à la fin de chaque épisode.
        - Réinitialise les compteurs pour l'épisode suivant.

        Returns:
            bool: Toujours True, ce qui permet de continuer l'entraînement.
        """
        
        self.current_rewards += self.locals["rewards"]
        self.current_length += 1

        ####### non utilisée
        # verifie si l'action était exploratory ou exploitative
        # epsilon = self.locals.get("epsilon", 0.1)  
        # if np.random.rand() < epsilon:
        #     self.exploration_actions += 1
        # else:
        #     self.exploitation_actions += 1

        #si l'épisode est terminé on rajoute à la liste
        if self.locals["dones"]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)
            
            #on vérifie si c'est succes ou echec
            if self.locals["infos"][0].get("is_success", False):  # Check success
                self.success_count += 1

            #calcul du succes rate
            total_episodes = len(self.episode_rewards)
            self.success_rate.append(self.success_count / total_episodes)

            # # exploration vs exploitation ratios
            # total_actions = self.exploration_actions + self.exploitation_actions
            # if total_actions > 0:
            #     self.episode_exploration.append(self.exploration_actions / total_actions)
            #     self.episode_exploitation.append(self.exploitation_actions / total_actions)

            #on reset pour la suite
            self.current_rewards = 0
            self.current_length = 0
            self.exploration_actions = 0
            self.exploitation_actions = 0

        return True