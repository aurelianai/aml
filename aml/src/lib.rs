#[cfg(test)]
mod tests;

use std::arch::asm;
use std::thread;

pub struct F32Tensor<'a> {
    pub shape: Vec<usize>,
    data: &'a [f32],
}

impl<'a> F32Tensor<'a> {
    /// Utility Method eliminating footguns assoc. with creating tensors by hand
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
struct F32Buffer(*mut f32);

unsafe impl Sync for F32Buffer {}
unsafe impl Send for F32Buffer {}
impl Clone for F32Buffer {
    fn clone(&self) -> Self {
        F32Buffer(self.0)
    }
}

impl F32Buffer {
    #[inline(always)]
    /// This will += `v` at index `i` in the underlying buffer
    unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) += v
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
    let c_ptr = F32Buffer(c.as_mut_ptr());

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

/// The other functions are counterintuitively far more vectorized that this one
/// The loops are agressively unrolled in a way that would be hard for me to match.
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
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            unsafe {
                sgemm_avx(a_ptr, b_ptr, c_ptr, m, n, p, block_size);
            }
        } else {
            panic!("AVX is not enabled");
        }
    }
}

/*
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn sgemm_avx512(
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

                )
            }
        }
    }
}
*/

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
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

                // ymm0 + ymm1 = c
                asm!("vmovups ymm0, [{}]", in(reg) c_ptr.add(row * p + col_block));
                asm!("vmovups ymm1, [{}]", in(reg) c_ptr.add(row * p + col_block + 8));

                // ymm4 = a_val
                asm!("movups xmm5, [{}]", in(reg) a_ptr.add(row * n + tile));
                asm!("movups xmm6, [{}]", in(reg) a_ptr.add(row * n + tile + 4));
                asm!("movups xmm7, [{}]", in(reg) a_ptr.add(row * n + tile + 8));
                asm!("movups xmm8, [{}]", in(reg) a_ptr.add(row * n + tile + 12));

                // ymm2 + ymm3 = b
                for tile_col in 0..4 {
                    asm!("vbroadcastss ymm4, xmm5");
                    asm!("vmovups ymm2, [{}]", in(reg) b_ptr.add((tile + tile_col) * p + col_block));
                    asm!("vmovups ymm3, [{}]", in(reg) b_ptr.add((tile + tile_col) * p + col_block + 8));
                    asm!("vfmadd231ps ymm0, ymm2, ymm4");
                    asm!("vfmadd231ps ymm1, ymm3, ymm4");
                    asm!("shufps xmm5, xmm5, 57");
                }

                for tile_col in 4..8 {
                    asm!("vbroadcastss ymm4, xmm6");
                    asm!("vmovups ymm2, [{}]", in(reg) b_ptr.add((tile + tile_col) * p + col_block));
                    asm!("vmovups ymm3, [{}]", in(reg) b_ptr.add((tile + tile_col) * p + col_block + 8));
                    asm!("vfmadd231ps ymm0, ymm2, ymm4");
                    asm!("vfmadd231ps ymm1, ymm3, ymm4");
                    asm!("shufps xmm6, xmm6, 57");
                }

                for tile_col in 8..12 {
                    asm!("vbroadcastss ymm4, xmm7");
                    asm!("vmovups ymm2, [{}]", in(reg) b_ptr.add((tile + tile_col) * p + col_block));
                    asm!("vmovups ymm3, [{}]", in(reg) b_ptr.add((tile + tile_col) * p + col_block + 8));
                    asm!("vfmadd231ps ymm0, ymm2, ymm4");
                    asm!("vfmadd231ps ymm1, ymm3, ymm4");
                    asm!("shufps xmm7, xmm7, 57");
                }

                for tile_col in 12..16 {
                    asm!("vbroadcastss ymm8, xmm3");
                    asm!("vmovups ymm2, [{}]", in(reg) b_ptr.add((tile + tile_col) * p + col_block));
                    asm!("vmovups ymm3, [{}]", in(reg) b_ptr.add((tile + tile_col) * p + col_block + 8));
                    asm!("vfmadd231ps ymm0, ymm2, ymm4");
                    asm!("vfmadd231ps ymm1, ymm3, ymm4");
                    asm!("shufps xmm8, xmm8, 57");
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
