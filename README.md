# mujoco-unitree

Browser-based Unitree robot simulation with MuJoCo WASM + Three.js.

**Live demo:** https://hwkim3330.github.io/mujoco-unitree/

## Features
- MuJoCo WASM physics (mujoco-js 0.0.7)
- Three.js 3D rendering with shadows and camera follow
- CPG (Central Pattern Generator) walking controllers
- Mobile touch controls (joystick + buttons)
- Obstacle spawning and simulation speed control

## Robots
| Robot | Type | Controller | Joints |
|-------|------|-----------|--------|
| **Unitree Go2** | Quadruped | Trot CPG | 12 (3 per leg) |
| **Unitree H1** | Humanoid | Bipedal CPG | 19 (legs + torso + arms) |

Go2 model from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie).
H1 model from [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco).

## Controls

### Desktop
| Key | Action |
|-----|--------|
| `W` `A` `S` `D` | Move |
| `Q` `E` | Rotate |
| `P` | Toggle CPG |
| `R` | Reset pose |
| `F` | Spawn obstacle |
| `[` `]` | Speed down / up |
| `Space` | Pause |
| `C` | Camera follow |

### Mobile
- **Left joystick**: Movement
- **Rotation buttons**: Turn left/right
- **Spawn buttons**: Ball / Box

## Run Locally
```bash
npm install
npm run build
npm run dev
# Open http://localhost:8080
```

## Tech Stack
- [mujoco-js](https://github.com/nicholasgasior/mujoco-js) 0.0.7
- [Three.js](https://threejs.org/) 0.181.0
- esbuild
