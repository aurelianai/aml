mod tests;

use half::f16;

/// Compressed representation of f32/f16 tensor in 4 bits.
///
/// See how scales and zeros are used in dot
pub struct I4Tensor<'a> {
    /// scale values for tensor
    pub scales: &'a [f16],
    /// Zero values for block quantization
    pub zeros: &'a [i8],
    /// data, called nibbles becuase the values are packed in pairs as a u8
    pub nibbles: &'a [i8],
    /// number of i4 values per scale/zero value value
    pub block_size: usize,
    pub shape: Vec<usize>,
}

impl I4Tensor<'_> {
    /// method to prevent you from creating invalid tensors
    pub fn new<'a>(
        scales: &'a [f16],
        zeros: &'a [i8],
        nibbles: &'a [i8],
        shape: Vec<usize>,
    ) -> I4Tensor<'a> {
        assert!(scales.len() == zeros.len() * 2);
        assert!(shape.iter().fold(1 as usize, |a, v| a * v) == nibbles.len() * 2);

        let block_size = nibbles.len() * 2 / scales.len();

        I4Tensor {
            scales,
            zeros,
            nibbles,
            block_size,
            shape,
        }
    }

    /// Get row of I4Tensor (m, n) --> I4Tensor (n, ) without copying.
    pub fn get_row(self: &Self, i: usize) -> I4Tensor {
        assert!(self.shape.len() == 2);

        let n = self.shape[1];
        let blocks = n / self.block_size;

        I4Tensor::new(
            &self.scales[i * blocks..(i + 1) * blocks],
            &self.zeros[(i * blocks) / 2..((i + 1) * blocks) / 2],
            &self.nibbles[(i * n) / 2..((i + 1) * n) / 2],
            vec![n],
        )
    }
}

pub struct F16Tensor {
    pub values: Vec<f16>,
    pub shape: Vec<usize>,
}

impl F16Tensor {
    pub fn new(values: Vec<f16>, shape: Vec<usize>) -> F16Tensor {
        assert!(values.len() == shape.iter().fold(1 as usize, |a, v| a * v));

        F16Tensor { values, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> F16Tensor {
        let n_elements = shape.iter().fold(1 as usize, |a, v| a * v);
        let values: Vec<f16> = vec![f16::from_f32(0f32); n_elements];

        F16Tensor { values, shape }
    }

    pub fn reshape(self: &mut Self, new_shape: Vec<usize>) {
        assert!(self.values.len() == new_shape.iter().fold(1 as usize, |a, v| a * v));
        self.shape = new_shape;
    }
}

/// Dot product between an F16 Tensor and a I4 tensor
///
/// Expects `a` (n, ) and `b` (n, )
pub fn qdot(a: &F16Tensor, b: &I4Tensor) -> f32 {
    assert!(a.shape.len() == 1);
    assert!(b.shape.len() == 1);
    assert!(a.shape[0] == b.shape[0]);

    let mut acc = 0f32;

    let num_blocks = (b.nibbles.len() * 2) / b.block_size;

    for block in 0..num_blocks {
        let scale = b.scales[block].to_f32();
        let zero = match block % 2 {
            0 => b.zeros[block / 2] >> 4,
            _ => (b.zeros[block / 2] << 4) >> 4,
        };

        let mut b_idx = 0;
        let mut a_idx = 0;
        loop {
            if b_idx * 2 == b.block_size {
                break;
            }

            let b_nibble = b.nibbles[(block * b.block_size + b_idx) / 2];
            let b1 = scale * ((b_nibble >> 4) - zero) as f32;
            let b2 = scale * (((b_nibble << 4) >> 4) - zero) as f32;

            let a1 = a.values[block * b.block_size + a_idx].to_f32();
            let a2 = a.values[block * b.block_size + a_idx + 1].to_f32();

            acc += b1 * a1 + b2 * a2;

            b_idx += 1;
            a_idx += 2;
        }
    }

    acc
}

/// Dot product between `a` (F16) and each row of `b` (I4)
///
/// I4 (m, n) @ F16 (n,) --> F16 (m,)
pub fn qgemv(a: &F16Tensor, b: &I4Tensor) -> F16Tensor {
    assert!(a.shape.len() == 1);
    assert!(b.shape.len() == 2);
    assert!(
        a.shape[0] == b.shape[1],
        "a is not the same length as b rows"
    );

    let mut out = Vec::with_capacity(b.shape[0]);

    for row_idx in 0..b.shape[0] {
        let b_row = b.get_row(row_idx);
        let b_row_dot = qdot(&a, &b_row);
        out.push(f16::from_f32(b_row_dot));
    }

    F16Tensor::new(out, vec![b.shape[0]])
}

/// Matrix Muliply between an `F16Tensor` and an `I4Tensor`. Result stored in `c` (F16)
///
/// I4(m, n) @ F16(k, n)**T --> F16(m, k)
pub fn qgemm(a: &F16Tensor, a_transpose: bool, b: &I4Tensor, b_transpose: bool, c: &mut F16Tensor) {
    assert!(
        a.shape.len() == 2,
        "`a` must have 2 dimensions. Found {}.",
        a.shape.len()
    );
    assert!(
        b.shape.len() == 2,
        "`b` must have 2 dimensions. Found {}.",
        b.shape.len()
    );

    let out_shape = vec![
        match a_transpose {
            true => a.shape[1],
            false => a.shape[0],
        },
        match b_transpose {
            true => b.shape[0],
            false => b.shape[1],
        },
    ];

    if a_transpose && b_transpose {
        assert!(
            a.shape[0] == b.shape[1],
            "Inner dimensions {}, {} do not match",
            a.shape[0],
            b.shape[1]
        );
    } else if a_transpose {
        assert!(
            a.shape[0] == b.shape[0],
            "Inner dimensions {}, {} do not match",
            a.shape[0],
            b.shape[0]
        );
    } else if b_transpose {
        assert!(
            a.shape[1] == b.shape[1],
            "Inner dimensions {}, {} do not match",
            a.shape[1],
            b.shape[1]
        );
    } else {
        assert!(
            a.shape[1] == b.shape[0],
            "Inner dimensions {}, {} do not match",
            a.shape[1],
            b.shape[0]
        );
    }

    assert!(
        out_shape == c.shape,
        "`c` has the wrong shape. Expected {:?}, found {:?}.",
        out_shape,
        c.shape
    );

    if a_transpose && b_transpose {
        todo!();
    } else if a_transpose {
        todo!();
    } else if b_transpose {
        todo!();
    } else {
        todo!();
    }
}
