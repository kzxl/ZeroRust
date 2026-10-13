//! Cache-aware loop schedulers and grid tiling.

/// Cache-aware 2D grid tiling: divides a 2D matrix into tiles [r0..r1, c0..c1]
/// that fit comfortably within CPU L1/L2 cache capacity.
pub fn tile_2d<F>(rows: usize, cols: usize, block_r: usize, block_c: usize, mut op: F)
where
    F: FnMut(usize, usize, usize, usize),
{
    let mut r = 0;
    while r < rows {
        let r_end = (r + block_r).min(rows);
        let mut c = 0;
        while c < cols {
            let c_end = (c + block_c).min(cols);
            op(r, r_end, c, c_end);
            c += block_c;
        }
        r += block_r;
    }
}

/// Range partitioning utility for parallel workers.
pub fn calculate_chunk_ranges(total: usize, num_workers: usize, min_chunk: usize) -> impl Iterator<Item = (usize, usize)> {
    let chunk_size = ((total + num_workers - 1) / num_workers).max(min_chunk);
    let num_chunks = (total + chunk_size - 1) / chunk_size;

    (0..num_chunks).map(move |i| {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(total);
        (start, end)
    })
}
