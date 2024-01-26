use aml::{sgemm, F32Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{thread_rng, Rng};

pub fn sgemm_benches(cri: &mut Criterion) {
    let mut rng = thread_rng();

    let mut a_data: Vec<f32> = vec![0.0; 4096 * 1024];
    a_data.iter_mut().for_each(|v| *v = rng.gen());
    let a = F32Tensor::new(&a_data, vec![1024, 4096]);

    let mut b_data: Vec<f32> = vec![0.0; 4096 * 512];
    b_data.iter_mut().for_each(|v| *v = rng.gen());
    let b = F32Tensor::new(&b_data, vec![4096, 512]);

    let mut c: Vec<f32> = vec![0.0; 512 * 1024];

    cri.bench_function("Large Rectangular Matrices", |bencher| {
        bencher.iter(|| {
            sgemm(
                black_box(&a),
                black_box(false),
                black_box(&b),
                black_box(false),
                black_box(&mut c),
            )
        })
    });
}

criterion_group!(benches, sgemm_benches);
criterion_main!(benches);
