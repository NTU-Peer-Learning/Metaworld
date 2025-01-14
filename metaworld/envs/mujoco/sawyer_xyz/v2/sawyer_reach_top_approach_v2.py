import mujoco
import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerReachTopApproachEnvV2(SawyerXYZEnv):
    """SawyerReachEnv.

    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """

    def __init__(self, tasks=None, render_mode=None):
        # goal_low = (-0.1, 0.8, 0.05)
        # goal_high = (0.1, 0.9, 0.3)
        goal_low = (-0.6, 0.35, 0.0)
        goal_high = (0.6, 0.95, 0.1)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_reach_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        reward, reach_dist, in_place = self.compute_reward(action, obs)
        success = float(reach_dist <= 0.05)

        info = {
            "success": success,
            "near_object": reach_dist,
            "grasp_success": 1.0,
            "grasp_reward": reach_dist,
            "in_place_reward": in_place,
            "obj_to_target": reach_dist,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_goal(self):
        return np.random.uniform(
            self._random_reset_space.low[-3:],
            self._random_reset_space.high[-3:],
            size=(3,),
        )

    def _get_pos_objects(self):
        # return self.get_body_com("obj")
        if self._target_pos is not None:
            return self._target_pos
        else:
            return self.get_body_com("obj")

    def _get_quat_objects(self):
        # geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        # return Rotation.from_matrix(geom_xmat).as_quat()
        # random quat
        return Rotation.random().as_quat()

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]
        self._set_obj_xyz(self.obj_init_pos)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.0
        tcp = self.tcp_center
        # obj = obs[4:7]
        # tcp_opened = obs[3]
        target = self._target_pos

        tcp_to_target = np.linalg.norm(tcp - target)
        # obj_to_target = np.linalg.norm(obj - target)

        # top approach
        threshold = 0.12
        # floor is a 3D funnel centered on the door handle
        radius = np.linalg.norm(tcp[:2] - target[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = (
            1.0
            if tcp[2] >= floor
            else reward_utils.tolerance(
                floor - tcp[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid="long_tail",
            )
        )

        # in place
        in_place_margin = np.linalg.norm(self.hand_init_pos - target)
        in_place = reward_utils.tolerance(
            tcp_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        # shake
        # VEL_MARGIN = 0.5
        # joint_vel_norm = np.linalg.norm(self.joint_vel)
        # shake = np.clip(joint_vel_norm, 0, VEL_MARGIN) / VEL_MARGIN

        top_approach_with_in_place = reward_utils.hamacher_product(
            above_floor, in_place
        )
        # in_place_with_shake = reward_utils.hamacher_product(
        #     top_approach_with_in_place, shake
        # )

        return [10 * top_approach_with_in_place, tcp_to_target, in_place]


class TrainReachGoalAsObjv2(SawyerReachTopApproachEnvV2):
    tasks = None

    def __init__(self):
        SawyerReachTopApproachEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestReachGoalAsObjv2(SawyerReachTopApproachEnvV2):
    tasks = None

    def __init__(self):
        SawyerReachTopApproachEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
