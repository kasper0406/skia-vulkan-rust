use ::skia_vulkan;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit::platform::run_return::EventLoopExtRunReturn;

use log::{ info };

fn main() {
    env_logger::init();

    let vulkan: skia_vulkan::VulkanInstance = skia_vulkan::VulkanInstance::new();

    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Skia - Test")
        .build(&event_loop)
        .unwrap();

    let mut skia_renderer = skia_vulkan::WindowRenderer::construct(&vulkan, &window);

    let start_time = std::time::Instant::now();

    let fps_report_interval = std::time::Duration::from_secs_f32(1.0);
    let mut frame_counter = 0;
    let mut last_fps_report_time = std::time::Instant::now();
    event_loop.run_return(|event, _window, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                frame_counter += 1;
                if last_fps_report_time.elapsed() > fps_report_interval {
                    let fps = (frame_counter as f32) / last_fps_report_time.elapsed().as_secs_f32();
                    info!["FPS = {}", fps];
                    last_fps_report_time = std::time::Instant::now();
                    frame_counter = 0;
                }

                skia_renderer.draw(window.inner_size().into(), &|canvas: &mut skia_safe::Canvas| {
                    canvas.clear(skia_safe::Color4f::new(1.0, 1.0, 1.0, 1.0));
                    let paint = skia_safe::Paint::new(skia_safe::Color4f::new(1.0, 0.0, 0.0, 1.0), None);

                    let degrees = ((360.0f32 / 5.0f32) * start_time.elapsed().as_secs_f32()) % 360.0;
                    canvas.rotate(degrees, Some(skia_safe::Point::new(300.0, 300.0)));

                    canvas.draw_circle(skia_safe::Point::new(200.0, 200.0), 100.0, &paint);
                });
            },
            _ => ()
        }
    });
}
