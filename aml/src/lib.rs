#[cfg(test)]
mod tests;

pub struct F32Tensor<'a> {
    shape: Vec<usize>,
    data: &'a [f32],
}

impl<'a> F32Tensor<'a> {
    /// Utility Method eliminating footguns assoc. with creating tensors by hand
    pub fn new(data: &'a [f32], shape: Vec<usize>) -> F32Tensor<'a> {
        assert!(shape.len() == 2, "Only Shapes of length 2 are supported");
        assert!(
            data.len() == shape.iter().fold(1, |acc, next| acc * next),
            "Data of Length {} doesn't work for shape {:#?}",
            data.len(),
            shape
        );

        Self { shape, data }
    }
}

pub fn sgemm(a: &F32Tensor, a_t: bool, b: &F32Tensor, b_t: bool, c: &mut Vec<f32>) {
    assert!(
        a.shape[1] == b.shape[0],
        "Tensor A Shape {:#?} is not compatible with Tensor B Shape {:#?}",
        a.shape,
        b.shape
    );
    assert!(!a_t && !b_t, "Transposes are not supported yet");

    let m = a.shape[0];
    let n = a.shape[1];
    let p = b.shape[1];

    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                c[i * p + k] += a.data[i * n + j] * b.data[j * p + k];
            }
        }
    }
}
