# `rs-gavisunk`
Rust port of [GAVISUNK](https://github.com/pdishuck/GAVISUNK)

> [!NOTE]
> Major WIP.

### Why?
Unnecessary/slow steps in original implementation.
* Count kmers with `jellyfish` and then map back with `mrsfast` to get positions.
* Phased ONT reads requirement.
* Pre-compiled Nim code.
* Slow ONT read annotation.

### Usage
```bash
cargo build --release
./target/release/rs-gavisunk
```
