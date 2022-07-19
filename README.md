# Skia backed by Vulkan in Rust

This is a minimalist library for drawing GPU accelerated graphics in Skia using Rust.
The crate uses [skia-safe](https://github.com/rust-skia/rust-skia), [The Ash Vulkan bindings in Rust](https://github.com/ash-rs/ash)
along with dependencies on [winit](https://github.com/rust-windowing/winit) to facilitate an easy to use interface to get started
drawing with Skia.

This library depends on no abstraction layers between `skia-safe` and `ash`, and gives easy control over
what commands are being executed.

The library supports drawing to multiple windows, and defaults to a MSAA sample count of 4, which I have found
to look a lot more pleasant than using no sampling.

See the `example.rs` file for how to use the library to start drawing.
