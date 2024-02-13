#[cfg(test)]
mod tests;

use std::{alloc::Layout, arch::asm, thread, thread::JoinHandle};

pub struct F32Tensor {
    pub shape: Vec<usize>,
    pub data: *mut f32,
    layout: Layout,
}

impl F32Tensor {
    /// Utility method eliminating footguns assoc. with creating tensors by hand
    pub fn new(shape: Vec<usize>) -> F32Tensor {
        assert!(shape.len() == 2, "Only Shapes of length 2 are supported");
        assert!(
            shape[0] % 16 == 0,
            "Dim 0 {} must be divisible by 16",
            shape[0]
        );
        assert!(
            shape[1] % 16 == 0,
            "Dim 1 {} must be divisible by 16",
            shape[1]
        );
        assert!(shape[0] * shape[1] > 0, "Must have more than 0 elements");

        // AVX512f requires 64 byte alignment
        let layout = Layout::from_size_align(shape[0] * shape[1], 64).unwrap();

        let data = unsafe { std::alloc::alloc(layout) as *mut f32 };

        Self { shape, data, layout }
    }
}

impl std::ops::Index<usize> for F32Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.data.add(index) }
    }
}

impl Drop for F32Tensor {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.data as *mut u8, self.layout);
        }
    }
}

/// Wrapper for *mut f32 that allows unsafe mutability from multiple threads.
#[derive(Copy)]
struct F32Buffer(*mut f32);
unsafe impl Send for F32Buffer {}
unsafe impl Sync for F32Buffer {}
impl Clone for F32Buffer {
    fn clone(self: &Self) -> F32Buffer {
        F32Buffer(self.0)
    }
}

impl F32Buffer {
    fn add(self: &Self, i: usize) -> *mut f32 {
        unsafe {
            self.0.offset(i as isize)
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn sgemm(a: &F32Tensor, b: &F32Tensor, c: &F32Tensor) {
    assert!(
        a.shape[1] == b.shape[0],
        "Tensor A Shape {:#?} is not compatible with Tensor B Shape {:#?}",
        a.shape,
        b.shape
    );
    assert!(
        a.shape[0] == c.shape[0] && b.shape[1] == c.shape[1],
        "Output tensor `c` has shape {:?}, but should have shape {}, {}",
        c.shape,
        a.shape[0],
        b.shape[1]
    );

    let m = a.shape[0];
    let n = a.shape[1];
    let k = b.shape[1];

    let block_size = 16;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            if k / block_size >= 8 {
                unsafe {
                    sgemm_avx512_parallel(a.data, b.data, c.data, m, n, k, block_size);
                }
            } else {
                unsafe {
                    sgemm_avx512_serial(a.data, b.data, c.data, m, n, k, block_size);
                }
            }
        } else if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            unsafe {
                sgemm_avx(a.data, b.data, c.data, m, n, k, block_size);
            }
        } else {
            panic!("Please run on a machine with AVX or AVX512f support");
        }
    }
}

/// Parallelized SIMD SGEMM with AVX512f
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn sgemm_avx512_parallel(
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) {
    let cols_per_thread = k / 8;
    let a = F32Buffer(a);
    let b = F32Buffer(b);
    let c = F32Buffer(c);

    (0..k)
        .into_iter()
        .step_by(cols_per_thread)
        .map(|t_col_start| {
            thread::spawn(move || {
                let t_col_end = t_col_start + cols_per_thread;
                for col_block in (t_col_start..t_col_end).step_by(block_size) {
                    for row in 0..m {
                        for tile in (0..n).step_by(block_size) {
                            asm!("vmovups zmm0, [{}]", in(reg) c.add(row * k + col_block));
                            for tile_col in 0..block_size {
                                asm!(
                                    "vbroadcastss zmm1, [{0}]",
                                    "vmovups zmm2, [{1}]",
                                    "vfmadd231ps zmm0, zmm1, zmm2",
                                    in(reg) a.add(row * n + tile + tile_col),
                                    in(reg) b.add((tile + tile_col) * k + col_block),
                                );
                            }
                            asm!("vmovups [{}], zmm0", in(reg) c.add(row * k + col_block));
                        }
                    }
                }
            })
        })
        .collect::<Vec<JoinHandle<()>>>()
        .into_iter()
        .for_each(|h| h.join().unwrap());
}

/// Serial SIMD SGEMM with AVX512f
#[inline(always)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn sgemm_avx512_serial(
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) {
    for col_block in (0..k).step_by(block_size) {
        for row in 0..m {
            for tile in (0..n).step_by(block_size) {
                asm!("vmovups zmm0, [{}]", in(reg) c.add(row * k + col_block));
                for tile_col in 0..block_size {
                    asm!(
                        "vbroadcastss zmm1, [{0}]",
                        "vmovups zmm2, [{1}]",
                        "vfmadd231ps zmm0, zmm1, zmm2",
                        in(reg) a.add(row * n + tile + tile_col),
                        in(reg) b.add((tile + tile_col) * k + col_block),
                    )
                }
                asm!("vmovups [{}], zmm0", in(reg) c.add(row * k + col_block));
            }
        }
    }
}

/// Serial SIMD SGEMM with AVX + FMA
#[inline(always)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn sgemm_avx(
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) {
    for col_block in (0..k).step_by(block_size) {
        for row in 0..m {
            for tile in (0..n).step_by(block_size) {
                asm!(
                    "vmovups ymm0, [{0}]",
                    "vmovups ymm1, [{1}]",
                    in(reg) c.add(row * k + col_block),
                    in(reg) c.add(row * k + col_block + 8),
                );
                for tile_col in 0..block_size {
                    asm!(
                        "vbroadcastss ymm2, [{0}]",
                        "vmovups ymm3, [{1}]",
                        "vfmadd231ps ymm0, ymm3, ymm2",
                        "vbroadcastss ymm2, [{0}]",
                        "vmovups ymm3, [{2}]",
                        "vfmadd231ps ymm1, ymm3, ymm5",
                        in(reg) a.add(row * n + tile + tile_col),
                        in(reg) b.add((tile + tile_col) * k + col_block),
                        in(reg) b.add((tile + tile_col) * k + col_block + 8),
                    );
                }
                asm!(
                    "vmovups [{0}], ymm0",
                    "vmovups [{1}], ymm1",
                    in(reg) c.add(row * k + col_block),
                    in(reg) c.add(row * k + col_block + 8),
                );
            }
        }
    }
}
