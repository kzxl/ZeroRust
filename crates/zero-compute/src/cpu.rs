//! Multi-threaded CPU execution dispatchers.

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Dispatches a range-based operation across the thread pool using Rayon.
pub fn parallel_for_range<F>(total: usize, min_chunk: usize, op: F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    #[cfg(feature = "rayon")]
    {
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((total + num_threads - 1) / num_threads).max(min_chunk);
        let num_chunks = (total + chunk_size - 1) / chunk_size;

        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(total);
            if start < end {
                op(start, end);
            }
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        op(0, total);
    }
}
