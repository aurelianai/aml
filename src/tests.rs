#[cfg(test)]
use crate::qdot;
#[cfg(test)]
use crate::qgemv;
#[cfg(test)]
use crate::F16Tensor;
#[cfg(test)]
use crate::I4Tensor;
#[cfg(test)]
use half::f16;

#[test]
pub fn dot_correctness_sm() {
    let zeros: Vec<i8> = vec![0x0F];
    let nibbles: Vec<i8> = vec![0xAAu8 as i8, 0x36];
    let scales: Vec<f16> = vec![3f32, 5f32].iter().map(|v| f16::from_f32(*v)).collect();
    let a = I4Tensor::new(&scales, &zeros, &nibbles, vec![4]);

    let values: Vec<f16> = vec![0f32, 4f32, 5f32, 8f32]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let b = F16Tensor::new(values, vec![4]);

    // Values: [-6,-6, 3, 6]
    // Scales: [3, 5]
    // Zeros: [0, -1]
    // Scaled, Shifted Values: [-18, -18, 20, 35]
    // b: [0, 4, 5, 8]
    // Dot with b: 0 -72 + 100 + 280 = 308

    let dot = qdot(&b, &a);

    assert!(dot as i32 == 308);
}

#[test]
pub fn qgemv_test_correctness_sm() {
    let a = F16Tensor::new(
        vec![0f32, 4f32, 5f32, 8f32]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect(),
        vec![4],
    );

    let scales: Vec<f16> = vec![3f32, 5f32, 3f32, 5f32, 3f32, 5f32]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let zeros: Vec<i8> = vec![0x0F, 0x0F, 0x0F];
    let nibbles: Vec<i8> = vec![0xAAu8 as i8, 0x36, 0xAAu8 as i8, 0x36, 0xAAu8 as i8, 0x36];
    let b = I4Tensor::new(&scales, &zeros, &nibbles, vec![3, 4]);

    let actual = qgemv(&a, &b);

    let expected: Vec<i32> = vec![308, 308, 308];
    for i in 0..expected.len() {
        assert!(actual.values[i].to_f32() as i32 == 308)
    }
}
