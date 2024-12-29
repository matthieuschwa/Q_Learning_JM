import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class GameEnv(gym.Env):
    """
    Environnement personnalisé pour notre problème de RL.

    Cet environnement simule un plateau où un héros doit trouver un trésor tout en évitant des monstres.
    Le héros peut se déplacer sur une grille, collecter des observations sur son état actuel et recevoir des
    récompenses en fonction de ses actions.

    Attributes:
        grid_size (int): Taille de la grille (par défaut 10x10).
        observation_space (gym.spaces.Box): Espace des observations (vecteur d'état).
        action_space (gym.spaces.Discrete): Espace des actions (5 actions possibles).
        hero_pos (np.ndarray): Position actuelle du héros.
        treasure_pos (np.ndarray): Position actuelle du trésor.
        monsters_pos (list): Liste des positions des monstres.
        step_count (int): Nombre d'étapes dans l'épisode en cours.
        hero_img, treasure_img, monster_img: Images pour le rendu visuel...
    """
    def __init__(self):
        super(GameEnv, self).__init__()
        self.grid_size = 10
        self.observation_space = spaces.Box(low=0, high=1, shape=(106,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  #actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY

        self.hero_pos = None
        self.treasure_pos = None
        self.monsters_pos = None
        self.previous_monster_positions = None
        self.previous_hero_pos = None
        self.step_count = 0

        self.hero_img = plt.imread("src/assets/hero.png")  
        self.treasure_img = plt.imread("src/assets/treasure.png")  
        self.monster_img = plt.imread("src/assets/monster.png") 

    def reset(self):
        """
        Reset l'environnement pour un nouvel épisode.
        Le héros, le trésor, et les monstres sont placés aléatoirement sur la grille, en s'assurant qu'ils ne se chevauchent pas,
        ne sortent pas du cadre...

        Return:
            np.ndarray: Observation initiale de l'état.
        """
        self.hero_pos = np.random.randint(0, self.grid_size, size=2)  #random position de l'agent
        self.previous_hero_pos = self.hero_pos.copy()
        self.treasure_pos = np.random.randint(0, self.grid_size, size=2)  #random position du tresor
        self.monsters_pos = []
        self.previous_monster_positions = []
        self.step_count = 0
        #conditions de spawn :
        while len(self.monsters_pos) < 3:
            monster_pos = np.random.randint(0, self.grid_size, size=2)
            if (not np.array_equal(monster_pos, self.hero_pos) and #les monstres ne doivent pas se trouver à la même position que l'agent
                not np.array_equal(monster_pos, self.treasure_pos) and #les monstres ne doivent pas se trouver à la même position que le trésor 
                np.linalg.norm(monster_pos - self.hero_pos, ord=1) >= 3 and #les monstres doivent être à une distance de Manhattan d'au moins 3 cases du héros 
                np.linalg.norm(monster_pos - self.treasure_pos, ord=1) >= 3): #les monstres doivent être à une distance de Manhattan d'au moins 3 cases du trésor 
                self.monsters_pos.append(monster_pos)
                self.previous_monster_positions.append(monster_pos.copy())
                #si cette position satisfait pas toutes les contraintes alors nouvelle tentative
        return self._get_obs()

    def _get_obs(self):
        """
        Génère une observation à partir de l'état actuel.

        L'observation inclut :
        - La grille aplatie avec les positions du héros, du trésor et des monstres.
        - La position relative du trésor par rapport au héros.
        - Les distances normalisées entre le héros et chaque monstre.
        - Un indicateur de proximité au trésor.

        Returns:
            np.ndarray: Vecteur d'état complet.
        """
        #représenter numériquement la localisation des entitées
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[self.hero_pos[0], self.hero_pos[1]] = 1  #valeur heros
        grid[self.treasure_pos[0], self.treasure_pos[1]] = 0.5  #valeur treasure
        for monster in self.monsters_pos:
            grid[monster[0], monster[1]] = -1  #valeur monsters

        relative_treasure_position = (self.treasure_pos - self.hero_pos) / self.grid_size  #distance hero trésor
        monster_distances = [
            np.linalg.norm(self.hero_pos - monster, ord=1) / self.grid_size for monster in self.monsters_pos
        ] # distances monsters hero, ord = 1 pour distance Manhattan
        while len(monster_distances) < 3: # sécurité mais improbable ici
            monster_distances.append(1.0)

        proximity_to_treasure = 1 if np.linalg.norm(self.hero_pos - self.treasure_pos, ord=1) <= 1 else 0 #distance hero tresor

        observation = np.concatenate([
            grid.flatten(), 
            relative_treasure_position,  
            monster_distances,  
            [proximity_to_treasure] 
        ])

        return observation #encode tout ce que l'agent doit savoir pour naviguer dans l'environnement

    def step(self, action):
        """
        Effectue une action et met à jour l'état de l'environnement.
        Args:
            action (int): Action choisie (0=Haut, 1=Bas, 2=Gauche, 3=Droite, 4=Stay).

        Return:
            tuple: (observation, reward, done, info)
        """
        self.step_count += 1 #pour suivre le nombre total d'actions prises par l'agent dans cet épisode
        self.previous_hero_pos = self.hero_pos.copy() #pour comparer les distances

        #déplacement agent
        if action == 0:  
            self.hero_pos[0] = max(0, self.hero_pos[0] - 1)
        elif action == 1:  
            self.hero_pos[0] = min(self.grid_size - 1, self.hero_pos[0] + 1)
        elif action == 2:  
            self.hero_pos[1] = max(0, self.hero_pos[1] - 1)
        elif action == 3:  
            self.hero_pos[1] = min(self.grid_size - 1, self.hero_pos[1] + 1)
        elif action == 4:  
            pass
        
        #déplacement monsters
        for i, monster in enumerate(self.monsters_pos):
            attempts = 0
            while attempts < 10: #10 tentatives, sinon le monstre reste à sa position actuelle
                # pour encourager un mouvement très dynamic et qu'ils ne restent pas tout le temps immobile
                new_monster_pos = monster + np.random.choice([-1, 0, 1], size=2)
                new_monster_pos[0] = np.clip(new_monster_pos[0], 0, self.grid_size - 1) #np.clip pour rester dans les limites de la grille
                new_monster_pos[1] = np.clip(new_monster_pos[1], 0, self.grid_size - 1)
                if (np.sum(np.abs(new_monster_pos - monster)) == 1 and #le monstre se déplace d'une case de 1
                    not np.array_equal(new_monster_pos, self.hero_pos) and #il ne chevauche pas le héros 
                    not np.array_equal(new_monster_pos, self.treasure_pos) and #il ne chevauche pas le trésor
                    np.linalg.norm(new_monster_pos - self.treasure_pos, ord=1) >= 3): #il reste à une distance d’au moins 3 cases du trésor
                    self.previous_monster_positions[i] = monster.copy()
                    monster[:] = new_monster_pos #màj
                    break
                attempts += 1
            else:
                self.previous_monster_positions[i] = monster.copy() #immobile

        reward = -0.1 #mini malus pour qu'il agisse efficacement et ne prenne pas trop de temps a trouver le trésor
        done = False #indicateur de partie terminée
        info = {"is_success": False} #indicateur de succès

        #les récompenses:
        for monster in self.monsters_pos:
            distance_to_monster = np.linalg.norm(self.hero_pos - monster, ord=1)
            if distance_to_monster == 1:  #adjacent a un monstre ; condition d'arret de la partie
                reward = -10 
                done = True
                return self._get_obs(), reward, done, {} 

        for monster in self.monsters_pos:
            distance_to_monster = np.linalg.norm(self.hero_pos - monster, ord=1)
            if distance_to_monster <= 2:  #trop proche d'un monstre
                reward -= 5

        previous_distance = np.linalg.norm(self.previous_hero_pos - self.treasure_pos)
        current_distance = np.linalg.norm(self.hero_pos - self.treasure_pos)
        if current_distance < previous_distance: #s'il s'approche du trésor
            reward += 3
        elif current_distance >= previous_distance: #s'il s'éloigne du trésor
            reward -= 1

        if np.array_equal(self.hero_pos, self.treasure_pos): #s'il atteint le trésor ; condition d'arret victoire
            reward = 30
            done = True
            info = {"is_success": True}

        return self._get_obs(), reward, done, info

    def render(self):
        """
        Affiche un rendu visuel de la grille et des positions actuelles.
        """
        fig, ax = plt.subplots(figsize=(8, 8)) #figure et axe
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(range(self.grid_size)) #grille
        ax.set_yticks(range(self.grid_size))
        ax.grid(True) #affiche les lignes de la grille

        #ajout du héros sur le visuel
        hero_imagebox = OffsetImage(self.hero_img, zoom=0.15)
        ab = AnnotationBbox(hero_imagebox, (self.hero_pos[1], self.grid_size - 1 - self.hero_pos[0]), frameon=False)
        ax.add_artist(ab)

        #ajout du tresor sur le visuel
        treasure_imagebox = OffsetImage(self.treasure_img, zoom=0.1)
        ab = AnnotationBbox(treasure_imagebox, (self.treasure_pos[1], self.grid_size - 1 - self.treasure_pos[0]), frameon=False)
        ax.add_artist(ab)

        #ajout des monstres sur le visuel
        for monster in self.monsters_pos:
            monster_imagebox = OffsetImage(self.monster_img, zoom=0.1)
            ab = AnnotationBbox(monster_imagebox, (monster[1], self.grid_size - 1 - monster[0]), frameon=False)
            ax.add_artist(ab)

        plt.show()