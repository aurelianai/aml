#[cfg(test)]
mod tests;

#[cfg(target_arch = "x86")]
use core::arch::x86;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use std::thread;

pub struct F32Tensor<'a> {
    shape: Vec<usize>,
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
    /* 
    pub fn to_col_major(self: &Self) -> F32Tensor {
        let mut new_data: Vec<f32> = Vec::new();
        
        let p = self.shape[0];
        let n = self.shape[1];
        
        for row_offset in 0..p  {
            for row in (0..n * p).step_by(p) {
                new_data.push(self.data[row + row_offset]);
            }
        }
        
        F32Tensor::new(&Vec::from(new_data), self.shape)
    }
    */
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

    if is_x86_feature_detected!("avx") {
        for col_block in (0..p).step_by(block_size) {
            for row in 0..m {
                for tile in (0..n).step_by(block_size) {
                    unsafe {
                        let mut c_1 = _mm256_loadu_ps(c_ptr.add(row * p + col_block));
                        let mut c_2 = _mm256_loadu_ps(c_ptr.add(row * p + col_block + 8));
                        for tile_col in 0..block_size {
                            let b_1 = _mm256_loadu_ps(
                                b_ptr.add(tile * p + tile_col * p + col_block),
                            );
                            let b_2 = _mm256_loadu_ps(
                                b_ptr.add(tile * p + tile_col * p + col_block + 8),
                            );

                            let a_val = _mm256_broadcast_ss(&*a_ptr.add(row * n + tile + tile_col));

                            c_1 = _mm256_add_ps(
                                c_1,
                                _mm256_mul_ps(a_val, b_1),
                            );
                            c_2 = _mm256_add_ps(
                                c_2,
                                _mm256_mul_ps(a_val, b_2),
                            );

                        }
                        _mm256_storeu_ps(c.as_mut_ptr().add(row * p + col_block), c_1);
                        _mm256_storeu_ps(c.as_mut_ptr().add(row * p + col_block + 8), c_2);
                    }
                }
            }
        }
    } else {
        for col_block in (0..p).step_by(block_size) {
            for row in 0..m {
                for tile in (0..n).step_by(block_size) {
                    for tile_row in 0..block_size {
                        for el in 0..block_size {
                            c[row * p + col_block + el] = a.data[row * n + tile + tile_row]
                                * b.data[tile * p + tile_row * p + col_block + el];
                        }
                    }
                }
            }
        }
    }
}

// Column Major Functions

/// WIP, just a copy of regular function
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
                c[i * p + j] += a.data[i * n + k] * b.data[k * p + j];
            }
        }
    }
}