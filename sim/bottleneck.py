"""JuPedSim bottleneck scenarios rendered to video frames, with exact ground truth.

A hall (20 m x 9 m) drains through a door of width `door_w` into a short corridor.
Agents start spread out on the left and compress in front of the door.
Frames are streamed (never stored whole) so long runs stay cheap in memory.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import jupedsim as jps
import shapely

PX_PER_M = 30
W_M, H_M = 24.0, 9.0                    # image covers hall + corridor
IMG_W, IMG_H = int(W_M * PX_PER_M), int(H_M * PX_PER_M)
HALL_X = 20.0                           # x of the door wall
# Zone watched in front of the door (metres): x0, y0, x1, y1
ZONE_M = (16.5, 2.0, 20.0, 7.0)
ZONE_AREA = (ZONE_M[2] - ZONE_M[0]) * (ZONE_M[3] - ZONE_M[1])
ZONE_PX = tuple(int(v * PX_PER_M) for v in ZONE_M)


@dataclass
class Scenario:
    door_w: float
    n_agents: int
    seed: int
    duration: float = 150.0
    fps: int = 10


def _geometry(door_w):
    y1, y2 = (H_M - door_w) / 2, (H_M + door_w) / 2
    return shapely.Polygon([(0, 0), (HALL_X, 0), (HALL_X, y1), (W_M, y1), (W_M, y2),
                            (HALL_X, y2), (HALL_X, H_M), (0, H_M)]), y1, y2


def zone_density_true(positions):
    """People per m^2 in the watched zone, from true positions (ground truth)."""
    x0, y0, x1, y1 = ZONE_M
    p = np.asarray(positions).reshape(-1, 2)
    inside = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
    return inside.sum() / ZONE_AREA


def run(sc: Scenario):
    """Yield (t, frame_uint8, true_density) once per video frame."""
    rng = np.random.default_rng(sc.seed)
    area, y1, y2 = _geometry(sc.door_w)
    sim = jps.Simulation(model=jps.CollisionFreeSpeedModel(), geometry=area, dt=0.01)
    exit_id = sim.add_exit_stage([(W_M - 1, y1), (W_M, y1), (W_M, y2), (W_M - 1, y2)])
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))

    start = [(0.5, 0.5), (13.0, 0.5), (13.0, 8.5), (0.5, 8.5)]
    pts = jps.distribute_by_number(polygon=shapely.Polygon(start),
                                   number_of_agents=sc.n_agents,
                                   distance_to_agents=0.45, distance_to_polygon=0.3,
                                   seed=sc.seed)
    sprites = {}
    for p in pts:
        aid = sim.add_agent(jps.CollisionFreeSpeedModelAgentParameters(
            position=p, journey_id=journey_id, stage_id=exit_id,
            desired_speed=float(np.clip(rng.normal(1.3, 0.2), 0.8, 1.8)), radius=0.2))
        sprites[aid] = _make_sprite(rng)

    steps = int(round(1.0 / (sc.fps * sim.delta_time())))
    n_frames = int(sc.duration * sc.fps)
    for k in range(n_frames):
        for _ in range(steps):
            sim.iterate()
        agents = list(sim.agents())
        pos = [a.position for a in agents]
        yield k / sc.fps, _render(agents, sprites), zone_density_true(pos) if pos else 0.0


def _make_sprite(rng, r=6):
    """Textured disc so optical flow has something to track."""
    d = 2 * r + 1
    yy, xx = np.mgrid[:d, :d]
    mask = (xx - r) ** 2 + (yy - r) ** 2 <= r * r
    tex = rng.integers(90, 255, size=(d, d)).astype(np.uint8)
    return tex, mask


def _render(agents, sprites):
    img = np.full((IMG_H, IMG_W), 15, np.uint8)
    for a in agents:
        tex, mask = sprites[a.id]
        r = tex.shape[0] // 2
        cx, cy = int(a.position[0] * PX_PER_M), int(a.position[1] * PX_PER_M)
        x0, y0 = cx - r, cy - r
        if x0 < 0 or y0 < 0 or x0 + tex.shape[1] > IMG_W or y0 + tex.shape[0] > IMG_H:
            continue
        roi = img[y0:y0 + tex.shape[0], x0:x0 + tex.shape[1]]
        roi[mask] = tex[mask]
    return img


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    sc = Scenario(door_w=float(sys.argv[1]), n_agents=int(sys.argv[2]), seed=1, duration=60)
    dens = [d for _, _, d in run(sc)]
    print(f"door {sc.door_w} n {sc.n_agents}: max zone density {max(dens):.2f} p/m2 "
          f"at t={np.argmax(dens)/sc.fps:.1f}s, wall time {time.time()-t0:.1f}s")
