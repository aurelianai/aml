## AML
A pure rust implementation of useful quantized BLAS operations for transformer/diffuser inference on the CPU.

1. Rust is safer, easier to build, and easier to read than C. Pure rust is also far easier to use as a dependency in rust projects.
2. Dynamic Optimization. For easier downstream use, the binary reacts to the available hardware automatically. This will result is *slightly* inferior performance due to extra jumps in the generated asm. 
3. I want to learn more about how LLM's work and what is holding back performance on CPUs.
