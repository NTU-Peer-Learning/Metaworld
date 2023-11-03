import numpy as np
from gymnasium.spaces import Box

import mujoco
from scipy.spatial.transform import Rotation
from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.panda.panda_env import (
    PandaEnv,
    _assert_task_is_set,
)


class PandaGripEnvV2(PandaEnv):
    OBJ_RADIUS = 0.02

    def __init__(self, tasks=None, render_mode=None):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.3, 0.5, 0.02)
        obj_high = (0.3, 0.7, 0.02)
        goal_low = (-0.3, 0.5, 0.00)
        goal_high = (0.3, 0.7, 0.2)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }
        self.goal = np.array([0.5, 0.65, 0.01])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.obj_z_noise = 0

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("panda/panda_sweep_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            object_grasped,
            shake_bonus,
        ) = self.compute_reward(action, obs)

        grasp_success = float(self.touching_main_object and tcp_opened)

        info = {
            "success": float(reward > 0.97),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_reward": object_grasped,
            "grasp_success": grasp_success,
            "in_place_reward": shake_bonus,
            "obj_to_target": -1,
            "unscaled_reward": reward,
        }
        return reward, info

    def _get_quat_objects(self):
        # return self.data.body("obj").xquat
        return Rotation.random().as_quat()

    def _get_pos_objects(self):
        pos = self.data.body("obj").xpos.copy()
        pos[2] += self.obj_z_noise
        return pos

    def _get_pos_goal(self):
        return np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(3,),
        )

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.objHeight = self._get_pos_objects()[2]

        obj_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate((obj_pos[:2], [self.obj_init_pos[-1]]))
        self._target_pos[1] = obj_pos.copy()[1]

        self._set_obj_xyz(self.obj_init_pos)
        self.obj_z_noise = np.random.uniform(-0.005, 0.005)
        self.maxPushDist = np.linalg.norm(
            self.get_body_com("obj")[:-1] - self._target_pos[:-1]
        )
        self.target_reward = 1000 * self.maxPushDist + 1000 * 2

        return self._get_obs()

    def _gripper_caging_reward(self, action, obj_position, obj_radius):
        pad_success_margin = 0.04
        grip_success_margin = obj_radius + 0.01
        x_z_success_margin = 0.001

        tcp = self.tcp_center
        left_pad = self.left_pad
        right_pad = self.right_pad
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_position[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_position[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        right_gripping = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_gripping = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = reward_utils.hamacher_product(right_caging, left_caging)
        y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

        assert y_caging >= 0 and y_caging <= 1

        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_position) + np.array(
            [0.0, -obj_position[1], 0.0]
        )
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
        init_obj_x_z = self.obj_init_pos + np.array([0.0, -self.obj_init_pos[1], 0.0])
        init_tcp_x_z = self.init_tcp + np.array([0.0, -self.init_tcp[1], 0.0])

        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        )
        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        assert right_caging >= 0 and right_caging <= 1
        # gripper_closed = min(max(0, action[-1]), 1)
        # assert gripper_closed >= 0 and gripper_closed <= 1
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        if caging > 0.95:
            gripping = y_gripping
        else:
            gripping = 0.0
        assert gripping >= 0 and gripping <= 1

        caging_and_gripping = (caging + gripping) / 2
        assert caging_and_gripping >= 0 and caging_and_gripping <= 1

        return caging_and_gripping

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        obj[2] -= self.obj_z_noise
        tcp_opened = self.gripper_opened
        # target = self._target_pos

        # obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        # in_place_margin = np.linalg.norm(self.obj_init_pos - target)

        # in_place = reward_utils.tolerance(
        #     obj_to_target,
        #     bounds=(0, _TARGET_RADIUS),
        #     margin=in_place_margin,
        #     sigmoid="long_tail",
        # )

        object_grasped = self._gripper_caging_reward(action, obj, self.OBJ_RADIUS)
        # in_place_and_object_grasped = reward_utils.hamacher_product(
        #     object_grasped, in_place
        # )

        object_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj"),
            object_vel,
            0,
        )
        # ic(object_vel)
        object_vel_norm = np.linalg.norm(object_vel[3:])

        shake_bonus = reward_utils.tolerance(
            object_vel_norm,
            bounds=(0, 0.3),
            margin=0.3,
            sigmoid="long_tail",
        )
        shake_bonus = 1 - shake_bonus

        if object_grasped > 0.95:
            reward = 9 * object_grasped + shake_bonus
        else:
            reward = 9 * object_grasped

        # reward = (2 * object_grasped) + (6 * in_place_and_object_grasped)
        # reward = object_grasped * 10

        # if obj_to_target < _TARGET_RADIUS:
        #     reward = 10.0
        return [reward, tcp_to_obj, tcp_opened, object_grasped, shake_bonus]


class TrainGripv2(PandaGripEnvV2):
    tasks = None

    def __init__(self):
        PandaGripEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestGripv2(PandaGripEnvV2):
    tasks = None

    def __init__(self):
        PandaGripEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
