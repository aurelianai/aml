use aml::{sgemm, F32Tensor};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::{thread_rng, Rng};
use std::hint::black_box;

pub fn sgemm_benches(cri: &mut Criterion) {
    let mut rng = thread_rng();

    // Small Square Matrices 512 x 512
    let mut a_data_sq_sm: Vec<f32> = vec![0.0; 512 * 512];
    a_data_sq_sm.iter_mut().for_each(|v| *v = rng.gen());
    let a_sq_sm = F32Tensor::new(vec![512, 512]);

    let mut b_data_sq_sm: Vec<f32> = vec![0.0; 512 * 512];
    b_data_sq_sm.iter_mut().for_each(|v| *v = rng.gen());
    let b_sq_sm = F32Tensor::new(vec![512, 512]);

    // Medium Square Matrices 1024 x 1024
    let mut a_data_sq_md: Vec<f32> = vec![0.0; 1024 * 1024];
    a_data_sq_md.iter_mut().for_each(|v| *v = rng.gen());
    let a_sq_md = F32Tensor::new(vec![1024, 1024]);

    let mut b_data_sq_md: Vec<f32> = vec![0.0; 1024 * 1024];
    b_data_sq_md.iter_mut().for_each(|v| *v = rng.gen());
    let b_sq_md = F32Tensor::new(vec![1024, 1024]);

    // Large Square Matrices 2048 x 2048
    let mut a_data_sq_lg: Vec<f32> = vec![0.0; 2048 * 2048];
    a_data_sq_lg.iter_mut().for_each(|v| *v = rng.gen());
    let a_sq_lg = F32Tensor::new(vec![2048, 2048]);

    let mut b_data_sq_lg: Vec<f32> = vec![0.0; 2048 * 2048];
    b_data_sq_lg.iter_mut().for_each(|v| *v = rng.gen());
    let b_sq_lg = F32Tensor::new(vec![2048, 2048]);

    // Declare Benches
    for (a, b) in [(a_sq_sm, b_sq_sm), (a_sq_md, b_sq_md), (a_sq_lg, b_sq_lg)].iter() {
        let mut c: Vec<f32> = vec![0.0; a.shape[0] * b.shape[1]];
        cri.bench_function(
            &format!("{:?} x {:?} (Naive)", a.shape, b.shape),
            |bencher| bencher.iter(|| sgemm(black_box(&a), black_box(&b), black_box(&mut c))),
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = sgemm_benches
}
criterion_main!(benches);
