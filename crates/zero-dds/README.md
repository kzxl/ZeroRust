# zero-dds

Micro-ROS & DDS direct link for `ZeroRust` embedded systems and robotics.

## Features
- **Zero Heap Allocation**: `#![no_std]` capable with in-place buffer serialization and natural alignment padding.
- **OMG CDR Standard**: Complete serializer and deserializer for ROS 2 Little/Big Endian Common Data Representation.
- **Standard ROS 2 Messages**: Builtin support for `builtin_interfaces/Time`, `std_msgs/Header`, `geometry_msgs/Twist`, `geometry_msgs/Vector3`, `geometry_msgs/Quaternion`, and `sensor_msgs/Imu`.
- **Micro-XRCE-DDS Client**: Communicates directly with standard ROS 2 Micro-XRCE Agents over Serial (UART), Ethernet UDP, or CAN bus without needing bulky C++ middleware.
