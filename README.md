# Autonomous Mobile Robot Navigation & Perception

## Overview
This repository contains the core navigation, planning, and perception pipeline for an autonomous mobile robot (TurtleBot3). The system is developed in **ROS2 (Python)** and integrates LiDAR-based dynamic obstacle avoidance, cost-aware $A^*$ planning, RRT* subpath replanning, and an OpenCV-based vision servoing module.

## Key Features 

### 1. Advanced Path Planning & Dynamic Obstacle Handling (`task2.py`, `task2_bonus.py`)
* **Cost-Aware $A^*$ Planning:** Implemented a custom $A^*$ planner with obstacle-distance weighting. Penalizes paths too close to walls, improving safe path planning
* **Dynamic Obstacle Avoidance:** Designed a LiDAR-based dynamic obstacle filtering pipeline (`STATIC_HIT_TOL`). It distinguishes obstacles from static walls and dynamically updates the occupancy grid in real-time.
* **RRT* Subpath Replanning:** When an obstacle blocks the global path, the system extracts a local segment and uses the **RRT* algorithm** to find a collision-free subpath, merging it seamlessly back into the global plan.

### 2. Autonomous Exploration (`task1.py`)
* **Frontier-Based Exploration:** Built a custom exploration policy using a reachable-mask flood fill algorithm. It autonomously identifies unknown regions and navigates the robot to map the entire environment without human intervention.

### 3. Vision-Based Object Recognition & Servoing (`task3.py`, `red_ball_tracker.py`)
* **Multi-Stage State Machine:** Designed a robust state machine to search, approach, and align with target objects (colored balls).
* **OpenCV Vision Pipeline:** Developed a custom vision pipeline using HSV color space thresholding and morphological operations to handle noise. It detects objects and estimates distance using a pinhole camera model.
* **PID Servoing:** Implemented custom PID controllers to smoothly regulate linear and angular velocities for precise object tracking and alignment.

## Repository Structure & Modules

* `task1.py`: Map processor, reachable mask generation, and autonomous frontier exploration.
* `task2.py` / `task2_bonus.py`: Global $A^*$ planner, LaserScan dynamic obstacle processing, and RRT* local replanning.
* `task3.py`: Waypoint navigation combined with multi-color object detection, coordinate transformations (Camera to Map), and PID-based servoing.
* `auto_navigator.py`: Base implementation of grid-based $A^*$ pathfinding and kinematics.
* `pid_control.py`: Low-level proportional-integral-derivative velocity controller.
* `red_ball_tracker.py`: Standalone vision-tracking module using OpenCV and ROS2 Image transport.

## Tech Stack
* **Framework:** ROS2
* **Languages:** Python
* **Libraries:** OpenCV, NumPy
* **Sensors:** 2D LiDAR, RGB Camera

## Demo / Results
**[Frontier Based Mapping Simulation _ Task_1]**
https://github.com/user-attachments/assets/4bdf2d2c-c24a-47eb-ac11-5e0d5d434f9f
**[A* algorithm with dynamic obstacle avoidance Task_2]**
https://github.com/user-attachments/assets/619038f6-644c-4572-871d-6a2463e50102
**[Ball Finder with OpenCv Camera and Exploration Method]**
https://github.com/user-attachments/assets/bd683b73-ef20-45b0-857b-b7054d8b428c

