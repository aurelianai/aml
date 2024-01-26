use crate::{sgemm, F32Tensor};
use float_cmp::approx_eq;

#[test]
fn sgemm_trivial_case() {
    let a_data = vec![
        -113.0, -97.0, -70.0, 19.0, -73.0, -1.0, 115.0, 37.0, -26.0, 53.0,
    ];
    let a = F32Tensor::new(&a_data, vec![2, 5]);
    let b_data = vec![
        -82.0, 23.0, 82.0, -73.0, -68.0, 122.0, 12.0, -44.0, 30.0, 4.0, -35.0, -116.0, -32.0,
        -101.0, 121.0,
    ];
    let b = F32Tensor::new(&b_data, vec![5, 3]);

    let mut c = vec![0.0; 6];

    sgemm(&a, false, &b, false, &mut c);

    let c_expected = [17919.0, 13785.0, -34237.0, -9669.0, -13914.0, 24487.0];

    for i in 0..6 {
        assert!(approx_eq!(f32, c[i], c_expected[i]));
    }
}

/* This should go into benchmarks. Takes forever
fn locate_test_file(file: &str) -> String {
    let mut test_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    test_file_path.push("resources");
    test_file_path.push(file);

    String::from(
        test_file_path
            .to_str()
            .expect("path could not be converted to string"),
    )
}

#[test]
pub fn sgemm_correctness() {
    let mut arr1_fp = File::open(locate_test_file("f32-arr1.bin")).expect("1 not found");
    let mut arr2_fp = File::open(locate_test_file("f32-arr2.bin")).expect("2 not found");
    let mut arr3_fp = File::open(locate_test_file("f32-arr3.bin")).expect("3 not found");

    let mut arr1 = vec![0.0; 4096 * 1024];
    arr1_fp
        .read_f32_into::<BigEndian>(&mut arr1)
        .expect("Error Reading F32s from `f32-arr1.bin`");

    let mut arr2 = vec![0.0; 4096 * 512];
    arr2_fp
        .read_f32_into::<BigEndian>(&mut arr2)
        .expect("Error Reading F32s from `f32-arr2.bin`");

    let mut arr3_expected = vec![0.0; 512 * 1024];
    arr3_fp
        .read_f32_into::<BigEndian>(&mut arr3_expected)
        .expect("Error Reading F32s from `f32-arr3.bin`");

    let a = F32Tensor::new(&mut arr1, vec![1024, 4096]);
    let b = F32Tensor::new(&mut arr2, vec![4096, 512]);
    let mut c = vec![0.0; 512 * 1024];

    sgemm(&a, false, &b, false, &mut c);

    for i in 0..(1024 * 512) {
        assert!(approx_eq!(f32, c[i], arr3_expected[i]));
    }
}
*/
