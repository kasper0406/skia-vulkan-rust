use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use vulkano::sync::{self, FenceSignalFuture, FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use skia_safe::gpu::{BackendRenderTarget, DirectContext, SurfaceOrigin};
use vulkano::{ Handle, VulkanObject, SynchronizedVulkanObject };
use vulkano::image::traits::ImageAccess;

use log::{ info, warn, error };

use ash::vk;
// use ash::version::InstanceV1_0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        "
    }
}

struct SelectedQueueFamilies<'a> {
    graphics: Option<QueueFamily<'a>>,
    present: Option<QueueFamily<'a>>,
}

pub fn select_physical_device<'a>(
    instance: &'a Arc<Instance>,
    surface: Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (PhysicalDevice<'a>, (QueueFamily<'a>, usize)) {
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families().enumerate()
                .find(|(_, q)| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|(idx, q)| (p, (q, idx)))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("no device available");
    (physical_device, queue_family)
}

fn find_queue_family<'a, T>(device: &'a PhysicalDevice, surface: &'a Surface<T>) -> SelectedQueueFamilies<'a> {
    let mut graphics = None;
    let mut present = None;

    let queue_families = device.queue_families();
    for queue_family in queue_families {
        if queue_family.supports_graphics() {
            graphics = Some(queue_family);
        }
        if queue_family.supports_surface(surface).unwrap() {
            present = Some(queue_family)
        }
        if graphics.is_some() && present.is_some() {
            break;
        }
    }

    SelectedQueueFamilies { graphics, present }
}

fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain<Window>>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: DontCare,
                format: swapchain.image_format(),  // set the format the same as the swapchain
                samples: 1,
            },
            resolve: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),  // set the format the same as the swapchain
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {},
            resolve: [resolve],
        }
    )
    .unwrap()
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .multisample_state(vulkano::pipeline::graphics::multisample::MultisampleState {
            rasterization_samples: 1,
            ..Default::default()
        })
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
}

fn get_command_buffers(
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    skia_image: Arc<dyn ImageAccess>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            let destination = framebuffer.attachments()[0].image();
            warn!["Destination: {:?}", destination.inner()];
            warn!["Skia image: {:?}", skia_image.inner()];

            builder
                .begin_render_pass(
                    framebuffer.clone(),
                    SubpassContents::Inline,
                    vec![[0.0, 0.0, 1.0, 1.0].into()],
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap()
                .copy_image(skia_image.clone(), [0,0,0], 0, 0, destination, [0,0,0], 0, 0, [0,0,0], 1)
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

unsafe fn skia_get_proc(
    vulkan_entry: &ash::Entry,
    instance_fns: &ash::vk::InstanceFnV1_0,
    of: skia_safe::gpu::vk::GetProcOf) -> Option<unsafe extern "system" fn()>
{
    match of {
        skia_safe::gpu::vk::GetProcOf::Instance(instance, name) => {
            let ash_instance = vk::Instance::from_raw(instance as _);
            // entry.get_instance_proc_addr(ash_instance, name)
            vulkan_entry.get_instance_proc_addr(ash_instance, name)
        }
        skia_safe::gpu::vk::GetProcOf::Device(device, name) => {
            let ash_device = vk::Device::from_raw(device as _);
            instance_fns.get_device_proc_addr(ash_device, name)
        }
    }
}

fn get_image_from_skia_texture(texture: &skia_safe::gpu::BackendTexture) -> vk::Image {
    unsafe { std::mem::transmute(texture.vulkan_image_info().unwrap().image) }
}

fn main() {
    env_logger::init();

    // let vulkan_loader = vulkano::instance::loader::auto_loader().unwrap();
    let vulkan_entry = unsafe { ash::Entry::load().unwrap() };

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    }).expect("failed to create instance");

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, (queue_family, queue_family_index)) =
        select_physical_device(&instance, surface.clone(), &device_extensions);
    info!["Running on {}", physical_device.properties().device_name];

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions), // new
            ..Default::default()
        },
    ).expect("failed to create device");

    let queue = queues.next().unwrap();

    let get_proc = |of| unsafe {
        let instance_fns = &instance.fns().v1_0;
        match skia_get_proc(&vulkan_entry, instance_fns, of) {
            Some(f) => f as _,
            None => {
                error!("resolve of {} failed", of.name().to_str().unwrap());
                std::ptr::null()
            }
        }
    };

    let skia_backend = unsafe {
        skia_safe::gpu::vk::BackendContext::new(
            instance.internal_object().as_raw() as _,
            physical_device.internal_object().as_raw() as _,
            device.internal_object().as_raw() as _,
            (
                queue.internal_object_guard().as_raw() as _,
                queue_family_index,
            ),
            &get_proc
        )
    };

    let mut skia_context = skia_safe::gpu::DirectContext::new_vulkan(&skia_backend, None).unwrap();

    let (mut swapchain, images) = {
        let capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let dimensions = surface.window().inner_size();
        let composite_alpha = capabilities.supported_composite_alpha.iter().next().unwrap();
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: capabilities.min_image_count,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::color_attachment(),
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap()
    };
    info!["Using {} images", images.len()];

    let render_pass = get_render_pass(device.clone(), swapchain.clone());
    let framebuffers = get_framebuffers(&images, render_pass.clone());

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            [
                Vertex { position: [ -1.0, 1.0 ] },
                Vertex { position: [ 1.0, -1.0 ] },
                Vertex { position: [ -1.0, -1.0 ] },
                Vertex { position: [ -1.0, 1.0 ] },
                Vertex { position: [ 1.0, -1.0 ] },
                Vertex { position: [ 1.0, 1.0 ] },
            ]
            .iter()
            .cloned()
        ).unwrap();

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: surface.window().inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );




    let dimensions: (i32, i32) = surface.window().inner_size().into();
    // let skia_image_info = skia_safe::ImageInfo::new_n32_premul((dimensions.0, dimensions.1), None);


    let skia_test_image = vulkano::image::attachment::AttachmentImage::with_usage(
        device.clone(),
        [dimensions.0 as u32, dimensions.1 as u32],
        // swapchain.image_format()
        vulkano::format::Format::B8G8R8A8_UNORM,
        ImageUsage {
            color_attachment: true,
            transfer_source: true,
            .. ImageUsage::none()
        },
    ).unwrap();
    warn!["Image layout = {:?}", skia_test_image.initial_layout_requirement()];
    unsafe { skia_test_image.layout_initialized() };

    /*
    let skia_render_target = skia_safe::gpu::BackendRenderTarget::new_vulkan(dimensions, Some(1), &skia_safe::gpu::vk::ImageInfo {
        image: skia_test_image.inner().image.internal_object().as_raw() as _,
        layout: skia_safe::gpu::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        format: skia_safe::gpu::vk::Format::B8G8R8A8_UNORM,
        sample_count: 1,
        level_count: 1,
        current_queue_family: queue_family_index as u32,
        .. skia_safe::gpu::vk::ImageInfo::default()
    });

    let mut skia_surface = skia_safe::Surface::from_backend_render_target(
        &mut skia_context,
        &skia_render_target,
        skia_safe::gpu::SurfaceOrigin::TopLeft,
        skia_safe::ColorType::BGRA8888,
        skia_safe::ColorSpace::new_srgb_linear(),
        None, // Surface props
    ).unwrap(); */

    /*
    let mut skia_surface = skia_safe::Surface::new_render_target(
        &mut skia_context,
        skia_safe::Budgeted::Yes,
        &skia_image_info,
        None, // Sample count
        skia_safe::gpu::SurfaceOrigin::TopLeft,
        None, // Surface props
        false
    ).unwrap();

    let skia_texture = skia_surface
            .get_backend_texture(skia_safe::surface::BackendHandleAccess::FlushRead)
            .as_ref().unwrap().clone();
    
    let skia_image = get_image_from_skia_texture(&skia_texture); */

    // let mut canvas = skia_surface.canvas();
    // canvas.clear(skia_safe::Color4f::new(0.0, 1.0, 0.0, 1.0));

    // skia_surface.flush_and_submit();




    let mut command_buffers = get_command_buffers(
        device.clone(),
        queue.clone(),
        pipeline,
        &framebuffers,
        vertex_buffer.clone(),
        skia_test_image,
    );

    let mut window_resized = false;
    let mut recreate_swapchain = false;
    
    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_index = 0;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                window_resized = true;
            }
            Event::MainEventsCleared => {
                if window_resized || recreate_swapchain {
                    recreate_swapchain = false;
    
                    let new_dimensions = surface.window().inner_size();
    
                    let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                        image_extent: new_dimensions.into(),
                        ..swapchain.create_info()
                    }) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };
                    swapchain = new_swapchain;
                    let new_framebuffers = get_framebuffers(&new_images, render_pass.clone());
    
                    if window_resized {
                        window_resized = false;
    
                        viewport.dimensions = new_dimensions.into();
                        let new_pipeline = get_pipeline(
                            device.clone(),
                            vs.clone(),
                            fs.clone(),
                            render_pass.clone(),
                            viewport.clone(),
                        );
                        /*
                        command_buffers = get_command_buffers(
                            device.clone(),
                            queue.clone(),
                            new_pipeline,
                            &new_framebuffers,
                            vertex_buffer.clone(),
                            skia_test_image,
                        ); */

                        panic!["Implement this!"];
                    }
                }

                let (image_index, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                // wait for the fence related to this image to finish (normally this would be the oldest fence)
                if let Some(image_fence) = &fences[image_index] {
                    image_fence.wait(None).unwrap();
                }

                let previous_future = match fences[previous_fence_index].clone() {
                    // Create a NowFuture
                    None => {
                        let mut now = sync::now(device.clone());
                        now.cleanup_finished();
    
                        now.boxed()
                    }
                    // Use the existing FenceSignalFuture
                    Some(fence) => fence.boxed(),
                };

                let future = previous_future
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffers[image_index].clone())
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_index)
                    .then_signal_fence_and_flush();
                
                fences[image_index] = match future {
                    Ok(value) => Some(Arc::new(value)),
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        None
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        None
                    }
                };
    
                previous_fence_index = image_index;
            },
            _ => (),
        }
    });
}
