//! Device abstraction traits and reference CPU implementation.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Vulkan,
    Wgpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeError {
    AllocationFailed,
    InvalidDimensions,
    NullPointer,
    HardwareUnsupported,
    InternalError,
}

/// Abstract hardware compute device.
pub trait ComputeDevice: Send + Sync {
    fn device_type(&self) -> DeviceType;
    fn synchronize(&self) -> Result<(), ComputeError>;
}

/// Reference CPU Compute Device.
pub struct CpuDevice {
    num_threads: usize,
}

impl CpuDevice {
    pub fn new() -> Self {
        #[cfg(feature = "rayon")]
        let threads = rayon::current_num_threads();
        #[cfg(not(feature = "rayon"))]
        let threads = 1;

        Self { num_threads: threads }
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeDevice for CpuDevice {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn synchronize(&self) -> Result<(), ComputeError> {
        Ok(())
    }
}
