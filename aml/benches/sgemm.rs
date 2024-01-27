use aml::{sgemm, sgemm_tiled, F32Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{thread_rng, Rng};

pub fn sgemm_benches(cri: &mut Criterion) {
    let mut rng = thread_rng();

    let mut a_data: Vec<f32> = vec![0.0; 1024 * 256];
    a_data.iter_mut().for_each(|v| *v = rng.gen());
    let a = F32Tensor::new(&a_data, vec![1024, 256]);

    let mut b_data: Vec<f32> = vec![0.0; 256 * 128];
    b_data.iter_mut().for_each(|v| *v = rng.gen());
    let b = F32Tensor::new(&b_data, vec![256, 128]);

    let mut c: Vec<f32> = vec![0.0; 1024 * 128];

    cri.bench_function("Medium Rectangular Matrices", |bencher| {
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

    let mut a_data_tiled: Vec<f32> = vec![0.0; 1024 * 256];
    a_data_tiled.iter_mut().for_each(|v| *v = rng.gen());
    let a_tiled = F32Tensor::new(&a_data_tiled, vec![1024, 256]);

    let mut b_data_tiled: Vec<f32> = vec![0.0; 256 * 128];
    b_data_tiled.iter_mut().for_each(|v| *v = rng.gen());
    let b_tiled = F32Tensor::new(&b_data_tiled, vec![256, 128]);

    let mut c_tiled: Vec<f32> = vec![0.0; 1024 * 128];

    cri.bench_function("Medium Rectangular Matrices (Tiled)", |bencher| {
        bencher.iter(|| {
            sgemm_tiled(
                black_box(&a_tiled),
                black_box(false),
                black_box(&b_tiled),
                black_box(false),
                black_box(&mut c_tiled),
            )
        })
    });

    let mut a_data_tiled_lg: Vec<f32> = vec![0.0; 4096 * 1024];
    a_data_tiled_lg.iter_mut().for_each(|v| *v = rng.gen());
    let a_tiled_lg = F32Tensor::new(&a_data_tiled_lg, vec![4096, 1024]);

    let mut b_data_tiled_lg: Vec<f32> = vec![0.0; 1024 * 512];
    b_data_tiled_lg.iter_mut().for_each(|v| *v = rng.gen());
    let b_tiled_lg = F32Tensor::new(&b_data_tiled_lg, vec![1024, 512]);

    let mut c_tiled_lg: Vec<f32> = vec![0.0; 4096 * 512];

    cri.bench_function("Medium Rectangular Matrices (Tiled)", |bencher| {
        bencher.iter(|| {
            sgemm_tiled(
                black_box(&a_tiled_lg),
                black_box(false),
                black_box(&b_tiled_lg),
                black_box(false),
                black_box(&mut c_tiled_lg),
            )
        })
    });   

}

criterion_group!(benches, sgemm_benches);
criterion_main!(benches);
