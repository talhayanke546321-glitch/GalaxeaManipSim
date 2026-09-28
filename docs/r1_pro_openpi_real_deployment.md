# R1 Pro π0.5 real-robot deployment

This deployment keeps model inference on the GPU workstation and runs a small
ROS2 client beside the R1 Pro SDK.  The checkpoint is simulation-trained and is
not validated for autonomous real-robot execution.  Always complete preflight
and shadow review before enabling publishers.

## Architecture

```text
R1 Pro cameras + joint feedback ──ROS2──> real client
                                             │
                                      WebSocket/msgpack
                                             │
                                      GPU policy server
                                             │
                                  15 x 16 absolute actions
                                             │
real client: validate → limit → 0..100 gripper conversion ──ROS2──> mobiman
```

The real client controls only the two seven-joint arms and two grippers.  It
does not command the torso or chassis.  The left arm is held at its measured
pose by default because the bottle-place expert uses the right arm.

## 1. GPU workstation

Start the 10,000-step policy server.  Keep the generated API key available for
the robot-side shell without placing it in command history or source control.

```bash
cd /home/vipuser/robotics/openpi
export OPENPI_API_KEY="$(openssl rand -hex 32)"

./.venv/bin/python scripts/serve_galaxea_r1_pro_bottle.py \
  --host 0.0.0.0 \
  --port 8000
```

Allow TCP port 8000 only from the robot LAN/VPN.  Do not expose this WebSocket
service directly to the public Internet.

After the new GPU is installed and the foreground command has been validated,
an optional systemd template is available at
`openpi/deploy/galaxea-r1-pro-bottle-openpi.service.example`.  It reuses the
API-key environment-file format from `openpi/deploy/galaxea-r1.env.example`.
Install and enable it only after adjusting paths and firewall rules for the
actual workstation.

With the server running, verify one authenticated network inference locally:

```bash
./.venv/bin/python scripts/smoke_test_galaxea_r1_pro_server.py
```

## 2. Robot-side Python environment

The client requires Python 3.10, ROS2 Humble's `rclpy`, and the Galaxea SDK.
Source ROS before activating/running the client.  Do not `pip install` the full
simulation project on the robot; use its source tree plus the lightweight
OpenPI client package.

```bash
source /opt/ros/humble/setup.bash
source /path/to/galaxea-sdk/install/setup.bash

python3 -m venv --system-site-packages ~/venvs/r1pro-openpi
source ~/venvs/r1pro-openpi/bin/activate
python -m pip install numpy pillow opencv-python tyro websockets msgpack
python -m pip install -e /path/to/openpi/packages/openpi-client

export GALAXEA_SKIP_SIM_REGISTRATION=1
export PYTHONPATH=/path/to/GalaxeaManipSim:${PYTHONPATH:-}
export OPENPI_API_KEY='<same value as the GPU workstation>'
```

Use the launch filenames shipped with the installed SDK version.  The required
nodes are the HDAS R1 Pro driver, three camera streams, arm joint tracker and
R1 Pro gripper controller.

## 3. Required ROS2 streams

The client waits for all seven inputs and fails closed if any is missing:

```text
/hdas/feedback_arm_left
/hdas/feedback_arm_right
/hdas/feedback_gripper_left
/hdas/feedback_gripper_right
/hdas/camera_head/left_raw/image_raw_color/compressed
/hdas/camera_wrist_left/color/image_raw/compressed
/hdas/camera_wrist_right/color/image_raw/compressed
```

When execution is explicitly enabled it publishes:

```text
/motion_target/target_joint_state_arm_left
/motion_target/target_joint_state_arm_right
/motion_target/target_position_gripper_left
/motion_target/target_position_gripper_right
```

Check the installed system before proceeding:

```bash
ros2 topic list | sort
ros2 topic hz /hdas/feedback_arm_right
ros2 topic hz /hdas/camera_head/left_raw/image_raw_color/compressed
ros2 topic echo --once /hdas/feedback_gripper_right
```

Gripper feedback and commands must use the SDK convention `0=closed,
100=open`.  The client converts this to and from the checkpoint convention
`0.00=closed, 0.05=open`.

## 4. Preflight (no policy connection, no publishers)

`preflight` is the default mode.  It waits for fresh observations, validates
dimensions/timestamps, converts all three images exactly as training did, and
writes three PNGs plus `preflight.json` for inspection.

```bash
python -m galaxea_sim.scripts.run_pi05_r1_pro_real \
  --mode preflight
```

Artifacts are stored under `real_robot_runs/r1_pro_bottle_place/`.  Verify:

* all three images are correctly oriented RGB views of the real workspace;
* no wrist camera is swapped;
* each arm contains seven finite joint positions in the trained order;
* grippers report values in `[0, 100]`;
* stream age and cross-stream skew pass the configured limits.

`preflight.json` also compares live feedback with the common initial state of
all 120 training demonstrations.  In policy order this state is:

```text
left arm:    [-0.4,  1.3, -0.7, -1.57,  1.3, -0.4, -0.8]
left grip:    0.0 policy units (0 hardware units, closed)
right arm:   [-0.4, -1.3,  0.7, -1.57, -1.3, -0.4,  0.8]
right grip:   0.0 policy units (0 hardware units, closed)
```

Preflight and shadow mode only warn when the live state is outside the default
`0.15 rad` arm / `0.005` policy-gripper envelope.  Execute mode fails before
publishing an action.  Move to the verified start pose using Galaxea's normal
operator workflow; this client deliberately does not implement an automatic
reset motion.

## 5. Shadow inference (policy connected, no publishers)

```bash
python -m galaxea_sim.scripts.run_pi05_r1_pro_real \
  --mode shadow \
  --policy-host <GPU_LAN_IP> \
  --policy-port 8000 \
  --max-steps 300
```

Shadow mode validates server metadata, sends real observations, applies joint
limits and rate limits, converts gripper units, and records every proposed
command to `steps.jsonl`.  It never creates ROS2 action publishers.

Review at least the following before any powered execution:

* no NaN/Inf, stale observation, inference timeout or protocol error;
* left arm remains at the measured pose;
* right-arm signs/order match the SDK;
* the training-initial-state comparison is understood and within tolerance;
* joint-limit clipping is absent or understood;
* predicted motion is consistent with the visible bottle and plate;
* policy latency is below the configured deadline.

## 6. Guarded execution

Execution is intentionally awkward to enable.  It requires both a risk flag
and an exact confirmation phrase because the checkpoint metadata declares
`real_robot_validated=false`.

Before running: clear at least 1.5 m around the robot, keep the physical and
remote emergency stops ready, place the table inside the trained 0.90–1.05 m
range, and start with no bottle in the reachable workspace.

```bash
python -m galaxea_sim.scripts.run_pi05_r1_pro_real \
  --mode execute \
  --policy-host <GPU_LAN_IP> \
  --policy-port 8000 \
  --max-steps 30 \
  --allow-sim-policy-on-real-robot \
  --execution-confirmation R1PRO_SIM_POLICY_EXECUTION_ACKNOWLEDGED
```

Defaults are deliberately conservative: 15 Hz, 10 executed actions per
15-step prediction, `0.02 rad` maximum arm change per step, `0.0025` maximum
policy-gripper change per step, `0.25 rad/s` requested arm velocity, and the
left arm held.  Review these values against the exact installed SDK and robot
configuration rather than increasing them blindly.

On stale data, slow inference, malformed actions, Ctrl+C or another exception,
the client issues a one-shot measured-position hold and exits.  This software
watchdog supplements rather than replaces the robot's physical emergency stop.
The inference deadline is applied directly to the WebSocket receive operation,
so a connected but stalled policy server cannot block the client indefinitely.

## 7. Local tests

The numerical contract is testable without ROS2:

```bash
cd /home/vipuser/robotics/GalaxeaManipSim
/home/vipuser/.conda/envs/galaxea-sim/bin/python -m pytest \
  tests/test_openpi_r1_pro_adapter.py \
  tests/test_openpi_r1_pro_real_adapter.py
```

The GPU policy can also be smoke-tested with the existing simulator before a
robot-side shadow run.  Real ROS topic presence, camera orientation, joint
ordering and powered motion remain hardware acceptance steps and cannot be
completed on the workstation alone.
