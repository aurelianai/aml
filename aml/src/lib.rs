#[cfg(test)]
mod tests;

use std::arch::asm;
use std::{thread, thread::JoinHandle};

pub struct F32Tensor<'a> {
    pub shape: Vec<usize>,
    data: &'a [f32],
}

impl<'a> F32Tensor<'a> {
    /// Utility method eliminating footguns assoc. with creating tensors by hand
    pub fn new(data: &'a [f32], shape: Vec<usize>) -> F32Tensor<'a> {
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
        assert!(
            data.len() == shape.iter().fold(1, |acc, next| acc * next),
            "Data of Length {} doesn't work for shape {:#?}",
            data.len(),
            shape
        );

        Self { shape, data }
    }
}

#[derive(Copy)]
struct F32MutBuffer(*mut f32);

unsafe impl Sync for F32MutBuffer {}
unsafe impl Send for F32MutBuffer {}
impl Clone for F32MutBuffer {
    fn clone(&self) -> Self {
        F32MutBuffer(self.0)
    }
}

impl F32MutBuffer {
    #[inline(always)]
    /// This will += `v` at index `i` in the underlying buffer
    unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) += v
    }

    #[inline(always)]
    unsafe fn add(self, i: usize) -> *mut f32 {
        self.0.add(i)
    }
}

#[derive(Copy)]
struct F32Buffer(*const f32);

unsafe impl Sync for F32Buffer {}
unsafe impl Send for F32Buffer {}
impl Clone for F32Buffer {
    fn clone(&self) -> Self {
        F32Buffer(self.0)
    }
}

impl F32Buffer {
    #[inline(always)]
    unsafe fn add(self, i: usize) -> *const f32 {
        self.0.add(i)
    }
}


pub fn sgemm(a: &F32Tensor, a_t: bool, b: &F32Tensor, b_t: bool, c: &mut Vec<f32>) {
    assert!(!a_t && !b_t, "Transposes are not supported yet");
    assert!(
        a.shape[1] == b.shape[0],
        "Tensor A Shape {:#?} is not compatible with Tensor B Shape {:#?}",
        a.shape,
        b.shape
    );
    assert!(
        a.shape[0] * b.shape[1] == c.len(),
        "Output buffer `c` has size {}, but should have {} * {}",
        c.len(),
        a.shape[0],
        b.shape[1]
    );

    let m = a.shape[0];
    let n = a.shape[1];
    let p = b.shape[1];

    for i in 0..m {
        for j in 0..p {
            for k in 0..n {
                c[i * p + j] += a.data[i * n + k] * b.data[k * p + j];
            }
        }
    }
}

pub fn sgemm_tiled(a: &F32Tensor, a_t: bool, b: &F32Tensor, b_t: bool, c: &mut Vec<f32>) {
    assert!(!a_t && !b_t, "Transposes are not supported yet");
    assert!(
        a.shape[1] == b.shape[0],
        "Tensor A Shape {:#?} is not compatible with Tensor B Shape {:#?}",
        a.shape,
        b.shape
    );
    assert!(
        a.shape[0] * b.shape[1] == c.len(),
        "Output buffer `c` has size {}, but should have {} * {}",
        c.len(),
        a.shape[0],
        b.shape[1]
    );

    let m = a.shape[0];
    let n = a.shape[1];
    let p = b.shape[1];

    let block_size = 16;

    for col_block in (0..p).step_by(block_size) {
        for row in 0..m {
            for tile in (0..n).step_by(block_size) {
                for tile_row in 0..block_size {
                    for el in 0..block_size {
                        unsafe {
                            *c.get_unchecked_mut(row * p + col_block + el) +=
                                *a.data.get_unchecked(row * n + tile + tile_row)
                                    * *b.data
                                        .get_unchecked(tile * p + tile_row * p + col_block + el);
                        }
                    }
                }
            }
        }
    }
}

pub fn sgemm_tiled_par(a: &F32Tensor, a_t: bool, b: &F32Tensor, b_t: bool, c: &mut Vec<f32>) {
    assert!(!a_t && !b_t, "Transposes are not supported yet");
    assert!(
        a.shape[1] == b.shape[0],
        "Tensor A Shape {:#?} is not compatible with Tensor B Shape {:#?}",
        a.shape,
        b.shape
    );
    assert!(
        a.shape[0] * b.shape[1] == c.len(),
        "Output buffer `c` has size {}, but should have {} * {}",
        c.len(),
        a.shape[0],
        b.shape[1]
    );

    let m = a.shape[0];
    let n = a.shape[1];
    let p = b.shape[1];

    let block_size = 16;
    let c_ptr = F32MutBuffer(c.as_mut_ptr());

    if p / 16 < 4 {
        println!("Small Matrix, so it will not run in parallel");
        sgemm_tiled(a, a_t, b, b_t, c);
    } else {
        let cols_per_thread = p / 4;

        thread::scope(|s| {
            for thread_col_start in (0..p).step_by(cols_per_thread) {
                s.spawn(move || {
                    for col_block in
                        (thread_col_start..thread_col_start + cols_per_thread).step_by(16)
                    {
                        for row in 0..m {
                            for tile in (0..n).step_by(block_size) {
                                for tile_row in 0..block_size {
                                    for el in 0..block_size {
                                        let c_index = row * p + col_block + el;
                                        unsafe {
                                            c_ptr.set(
                                                c_index,
                                                *a.data.get_unchecked(row * n + tile + tile_row)
                                                    * *b.data.get_unchecked(
                                                        tile * p + tile_row * p + col_block + el,
                                                    ),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn sgemm_tiled_simd(a: &F32Tensor, a_t: bool, b: &F32Tensor, b_t: bool, c: &mut Vec<f32>) {
    assert!(!a_t && !b_t, "Transposes are not supported yet");
    assert!(
        a.shape[1] == b.shape[0],
        "Tensor A Shape {:#?} is not compatible with Tensor B Shape {:#?}",
        a.shape,
        b.shape
    );
    assert!(
        a.shape[0] * b.shape[1] == c.len(),
        "Output buffer `c` has size {}, but should have {} * {}",
        c.len(),
        a.shape[0],
        b.shape[1]
    );

    let m = a.shape[0];
    let n = a.shape[1];
    let p = b.shape[1];

    let block_size = 16;

    let a_ptr = a.data.as_ptr();
    let b_ptr = b.data.as_ptr();
    let c_ptr = c.as_mut_ptr();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            if p / block_size >= 8 {
                unsafe {
                    sgemm_avx512_parallel(a_ptr, b_ptr, c_ptr, m, n, p, block_size);
                }
            } else {
                unsafe {
                    sgemm_avx512_serial(a_ptr, b_ptr, c_ptr, m, n, p, block_size);
                }
            }
        } else if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            unsafe {
                sgemm_avx(a_ptr, b_ptr, c_ptr, m, n, p, block_size);
            }
        } else {
            panic!("AVX not detected");
        }
    }
}

/// Parallelized SIMD SGEMM with AVX512f
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn sgemm_avx512_parallel(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    p: usize,
    block_size: usize,
) {
    let cols_per_thread = p / 8;

    let a_buf = F32Buffer(a_ptr);
    let b_buf = F32Buffer(b_ptr);
    let c_buf = F32MutBuffer(c_ptr);

    (0..p)
        .into_iter()
        .step_by(cols_per_thread)
        .map(|t_col_start| {
            thread::spawn(move || {
                let t_col_end = t_col_start + cols_per_thread;
                for col_block in (t_col_start..t_col_end).step_by(block_size) {
                    for row in 0..m {
                        for tile in (0..n).step_by(block_size) {
                            asm!("vmovups zmm0, [{}]", in(reg) c_buf.add(row * p + col_block));
                            for tile_col in 0..block_size {
                                asm!(
                                    "vbroadcastss zmm1, [{0}]",
                                    "vmovups zmm2, [{1}]",
                                    "vfmadd231ps zmm0, zmm1, zmm2",
                                    in(reg) a_buf.add(row * n + tile + tile_col),
                                    in(reg) b_buf.add((tile + tile_col) * p + col_block),
                                )
                            }
                            asm!("vmovups [{}], zmm0", in(reg) c_buf.add(row * p + col_block));
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
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    p: usize,
    block_size: usize,
) {
    for col_block in (0..p).step_by(block_size) {
        for row in 0..m {
            for tile in (0..n).step_by(block_size) {
                asm!("vmovups zmm0, [{}]", in(reg) c_ptr.add(row * p + col_block));
                for tile_col in 0..block_size {
                    asm!(
                        "vbroadcastss zmm1, [{0}]",
                        "vmovups zmm2, [{1}]",
                        "vfmadd231ps zmm0, zmm1, zmm2",
                        in(reg) a_ptr.add(row * n + tile + tile_col),
                        in(reg) b_ptr.add((tile + tile_col) * p + col_block),
                    )
                }
                asm!("vmovups [{}], zmm0", in(reg) c_ptr.add(row * p + col_block));
            }
        }
    }
}

/// Serial SIMD SGEMM with AVX + FMA
#[inline(always)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn sgemm_avx(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    p: usize,
    block_size: usize,
) {
    for col_block in (0..p).step_by(block_size) {
        for row in 0..m {
            for tile in (0..n).step_by(block_size) {
                asm!(
                    "vmovups ymm0, [{0}]",
                    "vmovups ymm1, [{1}]",
                    in(reg) c_ptr.add(row * p + col_block),
                    in(reg) c_ptr.add(row * p + col_block + 8),
                );
                for tile_col in 0..block_size {
                    asm!(
                        "vbroadcastss ymm2, [{0}]",
                        "vmovups ymm3, [{1}]",
                        "vfmadd231ps ymm0, ymm3, ymm2",
                        "vbroadcastss ymm2, [{0}]",
                        "vmovups ymm3, [{2}]",
                        "vfmadd231ps ymm1, ymm3, ymm5",
                        in(reg) a_ptr.add(row * n + tile + tile_col),
                        in(reg) b_ptr.add((tile + tile_col) * p + col_block),
                        in(reg) b_ptr.add((tile + tile_col) * p + col_block + 8),
                    );
                }
                asm!(
                    "vmovups [{0}], ymm0",
                    "vmovups [{1}], ymm1",
                    in(reg) c_ptr.add(row * p + col_block),
                    in(reg) c_ptr.add(row * p + col_block + 8),
                );
            }
        }
    }
}

// Column Major Functions

pub fn sgemm_cm(a: &F32Tensor, b: &F32Tensor, c: &mut Vec<f32>) {
    assert!(
        a.shape[1] == b.shape[0],
        "Tensor A Shape {:#?} is not compatible with Tensor B Shape {:#?}",
        a.shape,
        b.shape
    );
    assert!(
        a.shape[0] * b.shape[1] == c.len(),
        "Output buffer `c` has size {}, but should have {} * {}",
        c.len(),
        a.shape[0],
        b.shape[1]
    );

    let m = a.shape[0];
    let n = a.shape[1];
    let p = b.shape[1];

    for i in 0..m {
        for j in 0..p {
            for k in 0..n {
                c[i * p + j] += a.data[i * n + k] * b.data[j * n + k];
            }
        }
    }
}
