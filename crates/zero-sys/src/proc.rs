//! Zero-allocation Linux `/proc` filesystem parser for CPU, Memory, and Network telemetry.

use zero_core::error::{ZeroError, ZeroResult};

/// CPU time ticks parsed from `/proc/stat` line `cpu  ...`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CpuStats {
    /// Time spent in user mode.
    pub user: u64,
    /// Time spent in user mode with low priority (nice).
    pub nice: u64,
    /// Time spent in system mode.
    pub system: u64,
    /// Time spent in the idle task.
    pub idle: u64,
    /// Time waiting for I/O to complete.
    pub iowait: u64,
    /// Time servicing hardware interrupts.
    pub irq: u64,
    /// Time servicing softirqs.
    pub softirq: u64,
    /// Stolen time spent in other operating systems in virtualized environments.
    pub steal: u64,
}

impl CpuStats {
    /// Computes total CPU ticks across all categories.
    pub const fn total(&self) -> u64 {
        self.user
            + self.nice
            + self.system
            + self.idle
            + self.iowait
            + self.irq
            + self.softirq
            + self.steal
    }

    /// Computes non-idle (busy) CPU ticks.
    pub const fn busy(&self) -> u64 {
        self.user + self.nice + self.system + self.irq + self.softirq + self.steal
    }

    /// Computes CPU utilization percentage (0.0 to 100.0) between two sample snapshots.
    pub fn calculate_usage(prev: &Self, curr: &Self) -> f32 {
        let total_delta = curr.total().saturating_sub(prev.total());
        let busy_delta = curr.busy().saturating_sub(prev.busy());

        if total_delta == 0 {
            0.0
        } else {
            (busy_delta as f32 / total_delta as f32) * 100.0
        }
    }

    /// Parses the aggregate `cpu  ...` line from raw `/proc/stat` byte content without heap allocation.
    pub fn parse_stat_bytes(content: &[u8]) -> ZeroResult<Self> {
        let mut lines = content.split(|&b| b == b'\n');
        let cpu_line = lines.next().ok_or(ZeroError::UnexpectedEndOfBuffer)?;

        if !cpu_line.starts_with(b"cpu ") && !cpu_line.starts_with(b"cpu  ") {
            return Err(ZeroError::InvalidArgument);
        }

        let tokens = cpu_line
            .split(|&b| b == b' ' || b == b'\t')
            .filter(|chunk| !chunk.is_empty())
            .skip(1); // Skip the "cpu" literal

        let mut stats = Self::default();
        let mut count = 0;

        for tok in tokens {
            let val = parse_u64_ascii(tok).ok_or(ZeroError::InvalidArgument)?;
            match count {
                0 => stats.user = val,
                1 => stats.nice = val,
                2 => stats.system = val,
                3 => stats.idle = val,
                4 => stats.iowait = val,
                5 => stats.irq = val,
                6 => stats.softirq = val,
                7 => stats.steal = val,
                _ => break,
            }
            count += 1;
        }

        if count >= 4 {
            Ok(stats)
        } else {
            Err(ZeroError::UnexpectedEndOfBuffer)
        }
    }
}

/// System Memory usage statistics parsed from `/proc/meminfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemStats {
    /// Total usable RAM in Kilobytes.
    pub total_kb: u64,
    /// Free physical RAM in Kilobytes.
    pub free_kb: u64,
    /// Estimated memory available for starting new applications in Kilobytes.
    pub available_kb: u64,
    /// Memory used by kernel buffers in Kilobytes.
    pub buffers_kb: u64,
    /// Memory used by the page cache in Kilobytes.
    pub cached_kb: u64,
    /// Total swap space in Kilobytes.
    pub swap_total_kb: u64,
    /// Free swap space in Kilobytes.
    pub swap_free_kb: u64,
}

impl MemStats {
    /// Computes actually used memory in Kilobytes: `total_kb - available_kb`.
    pub const fn used_kb(&self) -> u64 {
        self.total_kb.saturating_sub(self.available_kb)
    }

    /// Computes percentage of memory currently in use (0.0 to 100.0).
    pub fn used_percentage(&self) -> f32 {
        if self.total_kb == 0 {
            0.0
        } else {
            (self.used_kb() as f32 / self.total_kb as f32) * 100.0
        }
    }

    /// Computes percentage of swap currently in use (0.0 to 100.0).
    pub fn swap_used_percentage(&self) -> f32 {
        if self.swap_total_kb == 0 {
            0.0
        } else {
            let used = self.swap_total_kb.saturating_sub(self.swap_free_kb);
            (used as f32 / self.swap_total_kb as f32) * 100.0
        }
    }

    /// Parses `/proc/meminfo` byte slice into strongly typed `MemStats`.
    pub fn parse_meminfo_bytes(content: &[u8]) -> ZeroResult<Self> {
        let mut stats = Self::default();

        for line in content.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            if let Some(pos) = line.iter().position(|&b| b == b':') {
                let key = &line[..pos];
                let rest = &line[pos + 1..];
                // Extract first numeric token
                if let Some(num_bytes) = rest
                    .split(|&b| b == b' ' || b == b'\t')
                    .find(|chunk| !chunk.is_empty())
                {
                    if let Some(val) = parse_u64_ascii(num_bytes) {
                        match key {
                            b"MemTotal" => stats.total_kb = val,
                            b"MemFree" => stats.free_kb = val,
                            b"MemAvailable" => stats.available_kb = val,
                            b"Buffers" => stats.buffers_kb = val,
                            b"Cached" => stats.cached_kb = val,
                            b"SwapTotal" => stats.swap_total_kb = val,
                            b"SwapFree" => stats.swap_free_kb = val,
                            _ => {}
                        }
                    }
                }
            }
        }

        // If MemAvailable was missing on older kernels, approximate: Free + Buffers + Cached
        if stats.available_kb == 0 && stats.total_kb > 0 {
            stats.available_kb = stats.free_kb + stats.buffers_kb + stats.cached_kb;
        }

        if stats.total_kb > 0 {
            Ok(stats)
        } else {
            Err(ZeroError::InvalidArgument)
        }
    }
}

/// Network interface metrics parsed from `/proc/net/dev`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NetInterfaceStats {
    /// Interface name (e.g. "eth0", "ens3", "lo").
    pub name: [u8; 16],
    /// Interface name length.
    pub name_len: usize,
    /// Total received bytes.
    pub rx_bytes: u64,
    /// Total received packets.
    pub rx_packets: u64,
    /// Total transmitted bytes.
    pub tx_bytes: u64,
    /// Total transmitted packets.
    pub tx_packets: u64,
}

impl NetInterfaceStats {
    /// Returns interface name as string slice.
    pub fn name_str(&self) -> &str {
        core::str::from_utf8(&self.name[..self.name_len]).unwrap_or("")
    }

    /// Parses `/proc/net/dev` bytes into a preallocated array of `NetInterfaceStats`.
    ///
    /// Returns number of parsed active network interfaces.
    pub fn parse_net_dev<const N: usize>(content: &[u8], out: &mut [Self; N]) -> usize {
        let mut count = 0;

        for line in content.split(|&b| b == b'\n') {
            if count >= N {
                break;
            }
            if let Some(colon) = line.iter().position(|&b| b == b':') {
                let name_part = line[..colon].trim_ascii();
                let stats_part = &line[colon + 1..];

                let mut tokens = stats_part
                    .split(|&b| b == b' ' || b == b'\t')
                    .filter(|chunk| !chunk.is_empty());

                let rx_bytes = tokens.next().and_then(parse_u64_ascii).unwrap_or(0);
                let rx_packets = tokens.next().and_then(parse_u64_ascii).unwrap_or(0);
                // Skip rx_errs, rx_drop, rx_fifo, rx_frame, rx_compressed, rx_multicast (6 tokens)
                for _ in 0..6 {
                    let _ = tokens.next();
                }
                let tx_bytes = tokens.next().and_then(parse_u64_ascii).unwrap_or(0);
                let tx_packets = tokens.next().and_then(parse_u64_ascii).unwrap_or(0);

                let mut name_buf = [0u8; 16];
                let copy_len = name_part.len().min(16);
                name_buf[..copy_len].copy_from_slice(&name_part[..copy_len]);

                out[count] = Self {
                    name: name_buf,
                    name_len: copy_len,
                    rx_bytes,
                    rx_packets,
                    tx_bytes,
                    tx_packets,
                };
                count += 1;
            }
        }

        count
    }
}

/// Helper function to parse ASCII bytes into u64 without allocating or pulling in std.
fn parse_u64_ascii(bytes: &[u8]) -> Option<u64> {
    if bytes.is_empty() {
        return None;
    }
    let mut acc: u64 = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add((b - b'0') as u64)?;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpu_stat_and_usage() {
        let sample1 = b"cpu  1000 50 300 5000 100 10 5 0\ncpu0 500 25 150 2500 50 5 2 0\n";
        let stats1 = CpuStats::parse_stat_bytes(sample1).unwrap();
        assert_eq!(stats1.user, 1000);
        assert_eq!(stats1.idle, 5000);

        let sample2 = b"cpu  1200 50 350 5100 100 10 5 0\n";
        let stats2 = CpuStats::parse_stat_bytes(sample2).unwrap();

        // Busy delta = (1200 - 1000) + (350 - 300) = 200 + 50 = 250
        // Idle delta = (5100 - 5000) = 100
        // Total delta = 350
        // Usage = 250 / 350 * 100 = 71.428%
        let usage = CpuStats::calculate_usage(&stats1, &stats2);
        assert!((usage - 71.428).abs() < 0.1);
    }

    #[test]
    fn test_parse_meminfo() {
        let sample = b"MemTotal:       16384000 kB\nMemFree:         4096000 kB\nMemAvailable:    8192000 kB\nBuffers:          500000 kB\nCached:          4000000 kB\nSwapTotal:       2048000 kB\nSwapFree:        1024000 kB\n";
        let mem = MemStats::parse_meminfo_bytes(sample).unwrap();
        assert_eq!(mem.total_kb, 16384000);
        assert_eq!(mem.available_kb, 8192000);
        assert_eq!(mem.used_kb(), 8192000); // 16384000 - 8192000
        assert!((mem.used_percentage() - 50.0).abs() < 1e-3);
        assert!((mem.swap_used_percentage() - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_parse_net_dev() {
        let sample = b"Inter-|   Receive                                                |  Transmit\n face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n    lo: 1234567     890    0    0    0     0          0         0  1234567     890    0    0    0     0       0          0\n  eth0: 987654321 45678    0    0    0     0          0         0 123456789 23456    0    0    0     0       0          0\n";
        let mut ifaces = [NetInterfaceStats::default(); 4];
        let count = NetInterfaceStats::parse_net_dev(sample, &mut ifaces);
        assert_eq!(count, 2);
        assert_eq!(ifaces[0].name_str(), "lo");
        assert_eq!(ifaces[0].rx_bytes, 1234567);
        assert_eq!(ifaces[1].name_str(), "eth0");
        assert_eq!(ifaces[1].rx_bytes, 987654321);
    }
}
