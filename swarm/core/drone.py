# ---------------------------------------------------------------------
# Camera helper ─ follow the first drone at ~60 Hz
# ---------------------------------------------------------------------
import numpy as np, pybullet as p, pybullet_data

def track_drone(cli, drone_id) -> None:
    """Keep the PyBullet spectator camera locked on the drone."""
    pos, _ = p.getBasePositionAndOrientation(drone_id,
                                             physicsClientId=cli)
    tg = np.add(pos, [0.0, 0.0, 0.4])                 # look ≈0.4 m above CG
    #tg = pos
    p.resetDebugVisualizerCamera(cameraDistance=1,   # zoom-out
                                 cameraYaw=0,
                                 cameraPitch=-80,       # slight downward tilt
                                 cameraTargetPosition=tg,
                                 physicsClientId=cli)   