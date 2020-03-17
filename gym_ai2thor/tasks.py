"""
Different task implementations that can be defined inside an ai2thor environment
"""

from gym_ai2thor.utils import InvalidTaskParams


class BaseTask:
    """
    Base class for other tasks to subclass and create specific reward and reset functions
    """
    def __init__(self, config):
        self.task_config = config
        self.max_episode_length = config.get('max_episode_length', 1000)
        # default reward is negative to encourage the agent to move more
        self.movement_reward = config.get('movement_reward', -0.01)
        self.step_num = 0

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """

        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class ExploreAllObjects(BaseTask):
    """
    This task consists of finding all objects in the enviorment.
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.target_objects = kwargs['task'].get('target_objects', {})
        self.discoverd = set()
        # self.never_found = set(self.target_objects.keys())

    def transition_reward(self, state, action_str=None):
        reward, done = self.movement_reward, False
        for obj in state.metadata['objects']:
            assert obj['name'] in self.target_objects
            if obj['visible'] and obj['name'] not in self.discoverd:
                self.discoverd.add(obj['name'])
                x, y, z = obj['position']['x'], obj['position']['y'], obj['position']['z']
                # print("Found {} at {}, {}, {}".format(obj['name'], x, y, z))
                reward += self.target_objects.get(obj['name'], 0)
                # self.never_found.discard(obj['name'])

        if self.max_episode_length and self.step_num >= self.max_episode_length or \
                len(self.discoverd) == len(self.target_objects):
            if len(self.discoverd) == len(self.target_objects):
                print("Used {} steps to find all objects".format(self.step_num))
                reward += 50
            else:
                print('Totally found objects {}/{} with {} steps'.format(len(self.discoverd), len(self.target_objects),
                                                                     self.step_num))
            done = True
            # print(self.never_found)

        return reward, done

    def reset(self):
        self.discoverd = set()
        self.step_num = 0
