"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""

import os

import ai2thor.controller
import numpy as np
from skimage import transform
from collections import defaultdict

import gym
from gym import error, spaces
from gym.utils import seeding
from gym_ai2thor.image_processing import rgb2gray
from gym_ai2thor.utils import read_config
import gym_ai2thor.tasks

import torch

ALL_POSSIBLE_ACTIONS = [
    'MoveAhead',
    'MoveBack',
    'MoveRight',
    'MoveLeft',
    'LookUp',
    'LookDown',
    'RotateRight',
    'RotateLeft',
    'OpenObject',
    'CloseObject',
    'PickupObject',
    'PutObject'
    # Non-available actions within this wrapper:
    # 'Rotate' is also possible when continuous_movement == True but we don't list it here
    # 'Teleport' and 'TeleportFull' but these shouldn't be allowable actions for an agent
    # These actions below are from ai2thor 1.0 but they're hard to discretize so they're not usable
    # 'DropHandObject', 'ThrowObject', 'ToggleObjectOn', 'ToggleObjectOff', 'MoveHandAhead',
    # 'MoveHandBack', 'MoveHandLeft', 'MoveHandRight', 'MoveHandUp', 'MoveHandDown', 'RotateHand'
]


class AI2ThorEnv(gym.Env):
    """
    Wrapper base class
    """
    def __init__(self, seed=None, config_file='config_files/config_example3.json', config_dict=None):
        """
        :param seed:         (int)   Random seed
        :param config_file:  (str)   Path to environment configuration file. Either absolute or
                                     relative path to the root of this repository.
        :param: config_dict: (dict)  Overrides specific fields from the input configuration file.
        """
        # Loads config settings from file
        self.config = read_config(config_file, config_dict)
        self.scene_id = self.config['scene_id']
        # Randomness settings
        self.np_random = None
        if seed:
            self.seed(seed)

        # Object settings
        # acceptable objects taken from config file.
        self.objects = {}
        if self.config['pickup_put_interaction']:
            self.objects['pickupables'] = self.config['pickup_objects']
            self.objects['receptacles'] = self.config['acceptable_receptacles']

        if self.config['open_close_interaction']:
            self.objects['openables'] = self.config['openable_objects']

        # Action settings
        self.action_names = tuple(ALL_POSSIBLE_ACTIONS.copy())
        # remove open/close and pickup/put actions if respective interaction bool is set to False
        if not self.config['open_close_interaction']:
            # Don't allow opening and closing if set to False
            self.action_names = tuple([action_name for action_name in self.action_names if 'Open'
                                       not in action_name and 'Close' not in action_name])
        if not self.config['pickup_put_interaction']:
            self.action_names = tuple([action_name for action_name in self.action_names if 'Pickup'
                                       not in action_name and 'Put' not in action_name])
        self.action_space = spaces.Discrete(len(self.action_names))
        # rotation settings
        self.continuous_movement = self.config.get('continuous_movement', False)
        if self.continuous_movement:
            self.absolute_rotation = 0.0
            self.rotation_amount = 10.0

        # Image settings
        self.event = None
        channels = 1 if self.config['grayscale'] else 3
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(channels, self.config['resolution'][0],
                                                   self.config['resolution'][1]),
                                            dtype=np.uint8)
        # ai2thor initialise function settings
        self.metadata_last_object_attributes = ['lastObjectPut', 'lastObjectPutReceptacle',
                                                'lastObjectPickedUp', 'lastObjectOpened',
                                                'lastObjectClosed']
        self.cameraY = self.config.get('cameraY', 0.0)
        self.gridSize = self.config.get('gridSize', 0.1)
        # Rendering options. Set segmentation and bounding box options off as default
        self.render_options = defaultdict(lambda: False)
        if 'render_options' in self.config:
            for option, value in self.config['render_options'].items():
                self.render_options[option] = value
        # Create task from config
        try:
            self.task = getattr(gym_ai2thor.tasks, self.config['task']['task_name'])(**self.config)
        except Exception as e:
            raise ValueError('Error occurred while creating task. Exception: {}'.format(e))
        # Start ai2thor
        # headless = False
        # if self.config['point_cloud_model']:
        #     headless = True
        self.controller = ai2thor.controller.Controller(quality="Very Low")

        # self.controller.start()

    def step(self, action, verbose=True):
        if not self.action_space.contains(action):
            raise error.InvalidAction('Action must be an integer between '
                                      '0 and {}!'.format(self.action_space.n))
        action_str = self.action_names[action]
        visible_objects = [obj for obj in self.event.metadata['objects'] if obj['visible']]
        for attribute in self.metadata_last_object_attributes:
            self.event.metadata[attribute] = None

        # if/else statements below for dealing with up to 13 actions
        if action_str.endswith('Object'):  # All interactions end with 'Object'
            # Interaction actions
            interaction_obj, distance = None, float('inf')
            inventory_before = self.event.metadata['inventoryObjects'][0]['objectType'] \
                if self.event.metadata['inventoryObjects'] else []
            if action_str.startswith('Put'):
                closest_receptacle = None
                if self.event.metadata['inventoryObjects']:
                    for obj in visible_objects:
                        # look for closest receptacle to put object from inventory
                        closest_receptacle_to_put_object_in = obj['receptacle'] and \
                                                              obj['distance'] < distance \
                                        and obj['objectType'] in self.objects['receptacles']\
                                        and (not obj['openable'] or (obj['openable'] and obj['isOpen']))
                        if closest_receptacle_to_put_object_in:
                            closest_receptacle = obj
                            distance = closest_receptacle['distance']
                    if closest_receptacle:
                        interaction_obj = closest_receptacle
                        object_to_put = self.event.metadata['inventoryObjects'][0]
                        self.event = self.controller.step(
                                dict(action=action_str,
                                     objectId=object_to_put['objectId'],
                                     receptacleObjectId=interaction_obj['objectId']))
                        self.event.metadata['lastObjectPut'] = object_to_put
                        self.event.metadata['lastObjectPutReceptacle'] = interaction_obj
            elif action_str.startswith('Pickup'):
                closest_pickupable = None
                for obj in visible_objects:
                    # look for closest object to pick up
                    closest_object_to_pick_up = obj['pickupable'] and \
                                                obj['distance'] < distance and \
                                obj['objectType'] in self.objects['pickupables']
                    if closest_object_to_pick_up:
                        closest_pickupable = obj
                if closest_pickupable and not self.event.metadata['inventoryObjects']:
                    interaction_obj = closest_pickupable
                    self.event = self.controller.step(
                        dict(action=action_str, objectId=interaction_obj['objectId']))
                    self.event.metadata['lastObjectPickedUp'] = interaction_obj
            elif action_str.startswith('Open'):
                closest_openable = None
                for obj in visible_objects:
                    # look for closest closed receptacle to open it
                    is_closest_closed_receptacle = obj['openable'] and \
                            obj['distance'] < distance and not obj['isOpen'] and \
                            obj['objectType'] in self.objects['openables']
                    if is_closest_closed_receptacle:
                        closest_openable = obj
                        distance = closest_openable['distance']
                if closest_openable:
                    interaction_obj = closest_openable
                    self.event = self.controller.step(
                        dict(action=action_str, objectId=interaction_obj['objectId']))
                    self.event.metadata['lastObjectOpened'] = interaction_obj
            elif action_str.startswith('Close'):
                closest_openable = None
                for obj in visible_objects:
                    # look for closest opened receptacle to close it
                    is_closest_open_receptacle = obj['openable'] and obj['distance'] < distance \
                                                 and obj['isOpen'] and \
                                                 obj['objectType'] in self.objects['openables']
                    if is_closest_open_receptacle:
                        closest_openable = obj
                        distance = closest_openable['distance']
                if closest_openable:
                    interaction_obj = closest_openable
                    self.event = self.controller.step(
                        dict(action=action_str, objectId=interaction_obj['objectId']))
                    self.event.metadata['lastObjectClosed'] = interaction_obj
            else:
                raise error.InvalidAction('Invalid interaction {}'.format(action_str))
            # print what object was interacted with and state of inventory
            if interaction_obj and verbose:
                inventory_after = self.event.metadata['inventoryObjects'][0]['objectType'] \
                    if self.event.metadata['inventoryObjects'] else []
                if action_str in ['PutObject', 'PickupObject']:
                    inventory_changed_str = 'Inventory before/after: {}/{}.'.format(
                                                            inventory_before, inventory_after)
                    if action_str == 'PutObject':
                        inventory_changed_str += " To Receptacle {}.".format(closest_receptacle["objectType"])
                else:
                    inventory_changed_str = ''
                print('{}: {}. {}'.format(
                    action_str, interaction_obj['objectType'], inventory_changed_str))
        elif action_str.startswith('Rotate'):
            if self.continuous_movement:
                # Rotate action
                if action_str.endswith('Left'):
                    self.absolute_rotation -= self.rotation_amount
                elif action_str.endswith('Right'):
                    self.absolute_rotation += self.rotation_amount
                self.event = self.controller.step(
                    dict(action='Rotate', rotation=self.absolute_rotation))
            else:
                # Do normal RotateLeft/Right command in discrete mode (i.e. 3D GridWorld)
                self.event = self.controller.step(dict(action=action_str))
        elif action_str.startswith('Move') or action_str.startswith('Look'):
            # Move and Look actions
            self.event = self.controller.step(dict(action=action_str))
        else:
            raise NotImplementedError('action_str: {} is not implemented'.format(action_str))

        self.task.step_num += 1

        reward, done = self.task.transition_reward(self.event, action_str)
        info = {}

        return self.get_observation(), reward, done, info

    def get_observation(self):
        if not self.config['point_cloud_model']:
            obs = torch.from_numpy(self.preprocess(self.event.frame)).unsqueeze(0).float()

        else:
            obj_points_feature = self.get_point_cloud(self.event)
            agent_info = self.get_agent_info(self.event)
            obs = (torch.from_numpy(obj_points_feature).unsqueeze(0).float(),
                   torch.from_numpy(agent_info).unsqueeze(0).float())
        return obs

    def preprocess(self, img):
        """
        Compute image operations to generate state representation
        """
        # TODO: replace scikit image with opencv
        img = transform.resize(img, self.config['resolution'], mode='reflect')
        img = img.astype(np.float32)
        if self.observation_space.shape[0] == 1:
            img = rgb2gray(img)
        img = np.moveaxis(img, 2, 0)
        return img

    def get_point_cloud(self, event):
        visible_objects = []
        not_visible_objects = []
        for obj in event.metadata['objects']:
            if obj['visible']:
                visible_objects.append(obj)
            else:
                not_visible_objects.append(obj)
        visible_array = np.array(list(map(lambda obj: [obj['position']['x'], obj['position']['y'], obj['position']['z'], 1],
                                   visible_objects))).transpose()
        not_visible_array = np.array(list(map(lambda obj: [obj['position']['x'], obj['position']['y'], obj['position']['z'], 0],
                                   not_visible_objects))).transpose()
        if visible_objects == []:
            cat_array = not_visible_array
        elif not_visible_objects == []:
            cat_array = visible_array
        else:
            cat_array = np.concatenate([visible_array, not_visible_array], axis=1)
        return cat_array

    def get_agent_info(self, event):
        metadata = event.metadata
        position = [metadata['cameraPosition']['x'], metadata['cameraPosition']['y'], metadata['cameraPosition']['z']]
        rotation_x = metadata['agent']['rotation']['x']
        rotation_y = metadata['agent']['rotation']['y']
        rotation_z = metadata['agent']['rotation']['z']
        assert rotation_x == 0
        assert rotation_z == 0
        look_angle = metadata['agent']['cameraHorizon']
        assert round(look_angle) in [-30, 0, 30, 60]
        if look_angle == 330:
            look_angle -= 360
        return np.array(position + [rotation_y * 2 * np.pi / 360] + [look_angle * 2 * np.pi / 360])


    def reset(self):
        # print('Resetting environment and starting new episode')
        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize', gridSize=self.gridSize,
                                               cameraY=self.cameraY,
                                               renderDepthImage=self.render_options['depth'],
                                               renderClassImage=self.render_options['class'],
                                               renderObjectImage=self.render_options['object'],
                                               continuous=self.continuous_movement))
        self.task.reset()
        return self.get_observation()

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        return seed1

    def close(self):
        self.controller.stop()


if __name__ == '__main__':
    AI2ThorEnv()
