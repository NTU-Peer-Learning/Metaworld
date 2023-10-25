import numpy as np
from gymnasium.spaces import Box

from metaworld.envs.mujoco.arm_env import ArmEnv
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set

import mujoco


class JacoEnv(ArmEnv):
    _ACTION_DIM = 10
    _QPOS_SPACE = Box(
        np.array(
            [
                -3.14,
                0.82,
                -3.14,
                0.524,
                -3.14,
                1.13,
                -3.14,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
        np.array(
            [
                3.14,
                5.46,
                3.14,
                5.76,
                3.14,
                5.15,
                3.14,
                1.51,
                2,
                1.51,
                2,
                1.51,
                2,
            ]
        ),
        dtype=np.float64,
    )

    def __init__(
        self,
        model_name,
        hand_low=...,
        hand_high=...,
        mocap_low=None,
        mocap_high=None,
        render_mode=None,
    ):
        super().__init__(
            model_name=model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            mocap_low=mocap_low,
            mocap_high=mocap_high,
            render_mode=render_mode,
        )

        self.hand_init_qpos = np.array(
            [0, 3.680, 0.000, 1.170, 0.050, 3.760, 0, 0.5, 0, 0.5, 0, 0.5, 0]
        )
        # self.init_finger_1 = self.get_body_com("jaco_link_finger_1")
        # self.init_finger_2 = self.get_body_com("jaco_link_finger_2")
        # self.init_finger_3 = self.get_body_com("jaco_link_finger_3")

        self.action_space = Box(
            np.array(
                [
                    -30.5,
                    -30.5,
                    -30.5,
                    -30.5,
                    -30.5,
                    -6.8,
                    -6.8,
                    0,
                    0,
                    0,
                ]
            ),
            np.array(
                [
                    30.5,
                    30.5,
                    30.5,
                    30.5,
                    30.5,
                    6.8,
                    6.8,
                    1.51,
                    1.51,
                    1.51,
                ]
            ),
            dtype=np.float64,
        )

    @property
    def tcp_center(self):
        """The COM of the gripper's 3 fingers.

        Returns:
            (np.ndarray): 3-element position
        """
        finger_1, finger_2, finger_3 = (
            self.data.body("pinky_distal"),
            self.data.body("index_distal"),
            self.data.body("thumb_distal"),
        )
        tcp_center = (finger_1.xpos + finger_2.xpos + finger_3.xpos) / 3.0
        return tcp_center

    def set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): 9-element array of actions
        """

        self.do_simulation(action, n_frames=self.frame_skip)

    def gripper_effort_from_action(self, action):
        return np.mean(action[-3:])
