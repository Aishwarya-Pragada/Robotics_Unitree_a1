"""
A1Env: Unitree A1 in MuJoCo, command-conditioned trot.

- Starts standing (settled pose + small noise), so no time wasted learning to stand.
- Observation includes velocity command (vx, vy, yaw_rate) and a gait clock (sin/cos).
- Action = joint position offsets from the standing pose (scaled by action_scale).
- Reward = velocity tracking + trot contact schedule (diagonal pairs) + regularizers.
- Optional domain randomization (randomize=True, OFF by default so the baseline is unchanged):
  per-episode friction, trunk mass, motor strength; random velocity pushes; sensor noise.
  `dr_scale` is the severity knob: 0 = nominal, 1 = training range, >1 = harder than trained.

Foot order everywhere: FR, FL, RR, RL
Trot pairs: (FR, RL) move together, (FL, RR) move together, half a cycle apart.
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
LEGS = ["FR", "FL", "RR", "RL"]
JOINT_TYPES = ["hip", "thigh", "calf"]
FALLBACK_DEFAULT = np.array([0.0, 0.9, -1.8])  # hip, thigh, calf standing pose

# reward weights (named so they are easy to tune / anneal later)
REWARD_WEIGHTS = dict(
    track_lin=1.5,
    track_yaw=0.75,
    gait_contact=1.0,
    lin_vel_z=-2.0,
    ang_vel_xy=-0.05,
    orientation=-2.0,
    base_height=-10.0,
    torque=-1e-4,
    action_rate=-0.01,
    hip_dev=-0.5,
)

# ---- domain randomization (ranges at severity dr_scale = 1) -------------------------------
DR_DEFAULT = dict(
    friction=(0.5, 1.5),         # multiplier on every sliding-friction coefficient (floor + feet)
    trunk_mass=(0.8, 1.5),       # multiplier on trunk mass (payload / battery / sensors)
    kp=(0.8, 1.2),               # multiplier on position-actuator stiffness (motor strength)
    push_vel=0.6,                # max velocity kick in m/s from a random push
    push_interval_s=(2.0, 5.0),  # time between pushes
    obs_noise=1.0,               # multiplier on the sensor-noise std below
)
DR_NOMINAL = dict(friction=1.0, trunk_mass=1.0, kp=1.0)
DR_FLOOR = dict(friction=0.15, trunk_mass=0.3, kp=0.4)    # keep severe settings physically valid
OBS_NOISE_STD = dict(gravity=0.02, ang_vel=0.1, joint_pos=0.01, joint_vel=0.2)


class A1Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path=os.path.join(HERE, "scene.xml"),
        ctrl_dt=0.02,
        episode_len_s=20.0,
        cmd_range=None,            # dict: vx, vy, yaw -> (low, high)
        cmd_resample_s=4.0,
        zero_cmd_prob=0.1,
        gait_freq=2.0,             # Hz
        duty=0.6,                  # stance fraction of each foot
        action_scale=0.25,         # rad
        kp=20.0,                   # only used if actuators are torque motors
        kd=0.5,
        reward_weights=None,
        randomize=False,           # domain randomization + pushes + sensor noise
        dr_scale=1.0,              # severity: 0 nominal, 1 training range, >1 harder
        dr_cfg=None,               # overrides for DR_DEFAULT, e.g. dict(friction=(0.4, 0.4))
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = None
        sim_dt = self.model.opt.timestep
        self.frame_skip = max(1, int(round(ctrl_dt / sim_dt)))
        self.dt = self.frame_skip * sim_dt
        self.max_steps = int(episode_len_s / self.dt)
        self.resample_steps = int(cmd_resample_s / self.dt)

        self.cmd_range = cmd_range or dict(vx=(-0.3, 0.8), vy=(-0.3, 0.3), yaw=(-0.8, 0.8))
        self.zero_cmd_prob = zero_cmd_prob
        self.gait_freq = gait_freq
        self.duty = duty
        self.action_scale = action_scale
        self.kp, self.kd = kp, kd
        self.w = dict(REWARD_WEIGHTS)
        if reward_weights:
            self.w.update(reward_weights)
        self.phase_offsets = np.array([0.0, 0.5, 0.5, 0.0])  # FR, FL, RR, RL

        self.randomize = bool(randomize)
        self.dr_scale = float(dr_scale)
        self.dr_cfg = dict(DR_DEFAULT)
        if dr_cfg:
            self.dr_cfg.update(dr_cfg)

        self._lookup_ids()
        self._store_nominal()
        self._find_default_pose()
        self._settle()

        self.rng = np.random.default_rng()
        self.noise_rng = np.random.default_rng()
        self._push_vel = 0.0
        self._noise_scale = 0.0
        self._push_timer = 10 ** 9
        self.dr_params = dict(friction=1.0, trunk_mass=1.0, kp=1.0, push_vel=0.0, obs_noise=0.0)
        self.fixed_cmd = None
        self.command = np.zeros(3)
        self.phase = 0.0
        self.step_count = 0
        self.last_action = np.zeros(12)

        obs_dim = 3 + 3 + 12 + 12 + 12 + 3 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)

    # ------------------------------------------------------------------ setup
    def _lookup_ids(self):
        m = self.model
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.trunk_id = tid if tid >= 0 else 1

        self.qpos_adr, self.dof_adr, self.act_ids = [], [], []
        for leg in LEGS:
            for jt in JOINT_TYPES:
                name = f"{leg}_{jt}_joint"
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"Joint '{name}' not found. Run check_env.py to see real names.")
                acts = np.where(m.actuator_trnid[:, 0] == jid)[0]
                if len(acts) == 0:
                    raise ValueError(f"No actuator drives joint '{name}'.")
                self.qpos_adr.append(m.jnt_qposadr[jid])
                self.dof_adr.append(m.jnt_dofadr[jid])
                self.act_ids.append(acts[0])
        self.qpos_adr = np.array(self.qpos_adr)
        self.dof_adr = np.array(self.dof_adr)
        self.act_ids = np.array(self.act_ids)

        # position actuators have affine bias (biastype == 1); otherwise torque motors -> own PD
        self.pos_act = bool(np.all(m.actuator_biastype[self.act_ids] == 1))
        self.ctrl_lo = m.actuator_ctrlrange[self.act_ids, 0]
        self.ctrl_hi = m.actuator_ctrlrange[self.act_ids, 1]
        self.ctrl_limited = bool(np.all(m.actuator_ctrllimited[self.act_ids] == 1))

        self.foot_geoms = np.array([self._find_foot_geom(l) for l in LEGS])
        self.foot_of_geom = {int(g): i for i, g in enumerate(self.foot_geoms)}

    def _find_foot_geom(self, leg):
        m = self.model
        for n in (leg, f"{leg}_foot", f"{leg}_foot_geom", f"{leg}_foot_collision"):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            if gid >= 0:
                return gid
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        geoms = np.where(m.geom_bodyid == body)[0]
        spheres = [g for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        return int(spheres[-1] if spheres else geoms[-1])

    def _find_default_pose(self):
        m = self.model
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            self.default_q = m.key_qpos[kid][self.qpos_adr].copy()
            self._start_qpos = m.key_qpos[kid].copy()
        else:
            self.default_q = np.tile(FALLBACK_DEFAULT, 4)
            self._start_qpos = m.qpos0.copy()
            self._start_qpos[:3] = [0, 0, 0.30]
            self._start_qpos[3:7] = [1, 0, 0, 0]
            self._start_qpos[self.qpos_adr] = self.default_q

    # ------------------------------------------------------- domain randomization
    def _store_nominal(self):
        """Remember nominal physics so every episode randomizes around the same baseline."""
        m = self.model
        self._nom = dict(
            friction=m.geom_friction.copy(),
            mass=float(m.body_mass[self.trunk_id]),
            inertia=m.body_inertia[self.trunk_id].copy(),
            gain=m.actuator_gainprm[self.act_ids, 0].copy(),
            bias=m.actuator_biasprm[self.act_ids, 1].copy(),
            kp=self.kp,
            kd=self.kd,
        )
        self._dr_dirty = False

    def set_dr_scale(self, scale):
        """Change severity (takes effect at the next reset). Used for curriculum / severity sweeps."""
        self.dr_scale = float(scale)

    def _scaled_range(self, key):
        lo, hi = self.dr_cfg[key]
        nom, s = DR_NOMINAL[key], self.dr_scale
        lo, hi = nom + s * (lo - nom), nom + s * (hi - nom)
        return max(lo, DR_FLOOR[key]), max(hi, DR_FLOOR[key])

    def _apply_domain_randomization(self):
        """Sample this episode's physics. Must run BEFORE mj_resetData (it may call mj_setConst)."""
        if not self.randomize and not self._dr_dirty:
            return
        m = self.model
        if self.randomize:
            f = self.rng.uniform(*self._scaled_range("friction"))
            ms = self.rng.uniform(*self._scaled_range("trunk_mass"))
            k = self.rng.uniform(*self._scaled_range("kp"))
            self._push_vel = self.dr_cfg["push_vel"] * self.dr_scale
            self._noise_scale = self.dr_cfg["obs_noise"] * self.dr_scale
        else:                                           # randomization was switched off: restore
            f = ms = k = 1.0
            self._push_vel = self._noise_scale = 0.0

        m.geom_friction[:, 0] = self._nom["friction"][:, 0] * f
        m.body_mass[self.trunk_id] = self._nom["mass"] * ms
        m.body_inertia[self.trunk_id] = self._nom["inertia"] * ms
        mujoco.mj_setConst(m, self.data)                # refresh derived mass quantities
        if self.pos_act:                                # scaling gain and bias keeps the target unchanged
            m.actuator_gainprm[self.act_ids, 0] = self._nom["gain"] * k
            m.actuator_biasprm[self.act_ids, 1] = self._nom["bias"] * k
        else:
            self.kp, self.kd = self._nom["kp"] * k, self._nom["kd"] * k

        lo, hi = self.dr_cfg["push_interval_s"]
        self._push_timer = int(self.rng.uniform(lo, hi) / self.dt)
        self._dr_dirty = self.randomize
        self.dr_params = dict(friction=float(f), trunk_mass=float(ms), kp=float(k),
                              push_vel=float(self._push_vel), obs_noise=float(self._noise_scale))

    def _maybe_push(self):
        """Random horizontal velocity kick on the trunk (world frame), like a shove."""
        if self._push_vel <= 0.0:
            return
        self._push_timer -= 1
        if self._push_timer > 0:
            return
        ang = self.rng.uniform(0.0, 2.0 * np.pi)
        mag = self.rng.uniform(0.3, 1.0) * self._push_vel
        self.data.qvel[0] += mag * np.cos(ang)
        self.data.qvel[1] += mag * np.sin(ang)
        lo, hi = self.dr_cfg["push_interval_s"]
        self._push_timer = int(self.rng.uniform(lo, hi) / self.dt)

    def _settle(self):
        """Let the robot settle once to get a physically consistent standing state."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._start_qpos
        for _ in range(int(1.0 / self.model.opt.timestep)):
            self._apply_control(self.default_q)
            mujoco.mj_step(self.model, self.data)
        self.init_qpos = self.data.qpos.copy()
        self.target_height = float(self.data.qpos[2])

    # ---------------------------------------------------------------- helpers
    def _apply_control(self, target):
        if self.pos_act:
            ctrl = target
        else:
            q = self.data.qpos[self.qpos_adr]
            qd = self.data.qvel[self.dof_adr]
            ctrl = self.kp * (target - q) - self.kd * qd
        if self.ctrl_limited:
            ctrl = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[self.act_ids] = ctrl

    def _contacts(self):
        """Returns (foot_contact[4] bool, trunk_touching_ground bool)."""
        foot = np.zeros(4, dtype=bool)
        trunk = False
        f6 = np.zeros(6)
        m = self.model
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == 0:
                other_g, other_b = c.geom2, b2
            elif b2 == 0:
                other_g, other_b = c.geom1, b1
            else:
                continue
            if other_b == self.trunk_id:
                trunk = True
            elif int(other_g) in self.foot_of_geom:
                mujoco.mj_contactForce(m, self.data, i, f6)
                if f6[0] > 1.0:
                    foot[self.foot_of_geom[int(other_g)]] = True
        return foot, trunk

    def _sample_command(self):
        if self.rng.random() < self.zero_cmd_prob:
            cmd = np.zeros(3)
        else:
            cmd = np.array([
                self.rng.uniform(*self.cmd_range["vx"]),
                self.rng.uniform(*self.cmd_range["vy"]),
                self.rng.uniform(*self.cmd_range["yaw"]),
            ])
            if np.linalg.norm(cmd[:2]) < 0.1:
                cmd[:2] = 0.0
            if abs(cmd[2]) < 0.1:
                cmd[2] = 0.0
        self.command = cmd

    def set_command(self, vx, vy, yaw):
        """Fix the command (used by test script and later by the maze waypoint follower)."""
        self.fixed_cmd = np.array([vx, vy, yaw], dtype=np.float64)
        self.command = self.fixed_cmd.copy()

    def clear_command(self):
        self.fixed_cmd = None

    def _obs(self):
        R = self.data.xmat[self.trunk_id].reshape(3, 3)
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        ang_vel = self.data.qvel[3:6]
        q = self.data.qpos[self.qpos_adr] - self.default_q
        qd = self.data.qvel[self.dof_adr]
        if self._noise_scale > 0.0:                     # sensor noise on what the policy sees only
            s, n = self._noise_scale, self.noise_rng.standard_normal
            gravity = gravity + s * OBS_NOISE_STD["gravity"] * n(3)
            ang_vel = ang_vel + s * OBS_NOISE_STD["ang_vel"] * n(3)
            q = q + s * OBS_NOISE_STD["joint_pos"] * n(12)
            qd = qd + s * OBS_NOISE_STD["joint_vel"] * n(12)
        clock = [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)]
        obs = np.concatenate([gravity, ang_vel, q, qd, self.last_action, self.command, clock])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.noise_rng = np.random.default_rng(seed + 10_007)

        self._apply_domain_randomization()
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.init_qpos.copy()
        qpos[0:2] = 0.0
        yaw = self.rng.uniform(-np.pi, np.pi)
        qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        qpos[self.qpos_adr] += self.rng.uniform(-0.05, 0.05, 12)
        self.data.qpos[:] = qpos
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.phase = self.rng.uniform(0.0, 1.0)
        self.step_count = 0
        self.last_action = np.zeros(12)
        if self.fixed_cmd is not None:
            self.command = self.fixed_cmd.copy()
        else:
            self._sample_command()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.default_q + self.action_scale * action
        self._maybe_push()
        for _ in range(self.frame_skip):
            self._apply_control(target)
            mujoco.mj_step(self.model, self.data)

        self.phase = (self.phase + self.gait_freq * self.dt) % 1.0
        self.step_count += 1
        if self.fixed_cmd is None and self.step_count % self.resample_steps == 0:
            self._sample_command()

        # ---- state
        R = self.data.xmat[self.trunk_id].reshape(3, 3)
        v_local = R.T @ self.data.qvel[0:3]
        w_local = self.data.qvel[3:6]
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        z = self.data.qpos[2]
        foot, trunk_hit = self._contacts()
        torque = self.data.actuator_force[self.act_ids]
        q = self.data.qpos[self.qpos_adr]

        # ---- reward terms
        lin_err = np.sum((self.command[:2] - v_local[:2]) ** 2)
        yaw_err = (self.command[2] - w_local[2]) ** 2
        desired = ((self.phase + self.phase_offsets) % 1.0) < self.duty
        hip_idx = [0, 3, 6, 9]
        terms = dict(
            track_lin=np.exp(-lin_err / 0.25),
            track_yaw=np.exp(-yaw_err / 0.25),
            gait_contact=np.mean(foot == desired),
            lin_vel_z=v_local[2] ** 2,
            ang_vel_xy=np.sum(w_local[:2] ** 2),
            orientation=np.sum(gravity[:2] ** 2),
            base_height=(z - self.target_height) ** 2,
            torque=np.sum(torque ** 2),
            action_rate=np.sum((action - self.last_action) ** 2),
            hip_dev=np.sum((q[hip_idx] - self.default_q[hip_idx]) ** 2),
        )
        reward = sum(self.w[k] * v for k, v in terms.items())
        self.last_action = action

        terminated = bool(z < 0.55 * self.target_height or gravity[2] > -0.5 or trunk_hit)
        truncated = self.step_count >= self.max_steps

        info = {f"rew/{k}": float(self.w[k] * v) for k, v in terms.items()}
        info.update(
            foot_contacts=foot.copy(),
            cmd=self.command.copy(),
            v_local=v_local.copy(),
            w_local=w_local.copy(),
            fell=terminated,
        )
        return self._obs(), float(reward), terminated, truncated, info