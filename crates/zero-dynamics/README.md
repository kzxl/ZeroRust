# zero-dynamics

High-performance Robotics Dynamics and Collaborative Cobot Control for `ZeroRust`.

## Features
- **Zero Dynamic Allocation**: `#![no_std]` capable with stack-only computation and preallocated workspaces.
- **Recursive Newton-Euler (RNEA)**: $O(N)$ inverse dynamics computation for serial manipulators (6-DOF, 7-DOF redundant).
- **Exact Dynamic Terms**: Generates mass matrix $M(q)$, gravity vector $g(q)$, and Coriolis vector $C(q, \dot{q})\dot{q}$.
- **Geometric Jacobian & Manipulability**: Forward velocity mapping, wrench-to-torque transformation, and Yoshikawa manipulability index.
- **Sensorless Collision Detection**: Generalized momentum observer detecting collisions in $< 1\,\text{ms}$ without expensive multi-axis F/T sensors.
- **Impedance & Admittance Control**: Virtual stiffness/damping for peg-in-hole assembly and compliant hand-guiding teaching.
