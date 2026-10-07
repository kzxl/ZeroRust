//! # ZeroDDS
//!
//! Pure Rust, `#![no_std]` capable Micro-ROS and DDS direct link:
//! - OMG CDR (Common Data Representation) zero-allocation serializer and deserializer
//! - Standard ROS 2 message definitions (`std_msgs`, `geometry_msgs`, `sensor_msgs`)
//! - Micro-XRCE-DDS protocol framing and client publisher

#![cfg_attr(not(feature = "std"), no_std)]

pub mod cdr;
pub mod client;
pub mod messages;
pub mod xrce;

pub use cdr::{CdrReader, CdrScheme, CdrWriter};
pub use client::{SessionState, XrceClient};
pub use messages::{Header, Imu, Quaternion, Time, Twist, Vector3};
pub use xrce::{MessageHeader, SubmessageHeader, SubmessageId, WriteDataPayload};
