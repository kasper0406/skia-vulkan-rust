use std::sync::Arc;
use std::ffi::{ CStr, CString };

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use skia_safe::gpu::{BackendRenderTarget, DirectContext, SurfaceOrigin};

use log::{ info, warn, error };

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use ash::vk::{
    KhrGetPhysicalDeviceProperties2Fn, KhrPortabilityEnumerationFn, KhrPortabilitySubsetFn,
};

use ash::vk;
use ash::vk::Handle;
use ash::vk::PhysicalDeviceType;
use ash::prelude::*;

/*
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);
*/

/*
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
} */

/*
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
} */

/*
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
*/

/*
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
*/

/*
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
} */

/*
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
*/

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
            (instance_fns.get_device_proc_addr)(ash_device, name)
        }
    }
}

fn get_image_from_skia_texture(texture: &skia_safe::gpu::BackendTexture) -> vk::Image {
    unsafe { std::mem::transmute(texture.vulkan_image_info().unwrap().image) }
}

fn create_instance(application_name: String, extensions: &[*const std::os::raw::c_char], entry: &ash::Entry) -> VkResult<ash::Instance> {
    // TODO(knielsen): Support custom extensions
    /*
    let enabled_extensions_cstr: Vec<CString> = vec![
        "VK_KHR_surface",
        "VK_MVK_macos_surface",
        "VK_KHR_get_physical_device_properties2",
        "VK_KHR_get_surface_capabilities2"
    ].into_iter().map(|extension_name| CString::new(extension_name).unwrap()).collect();
    let enabled_extensions_cstr_raw: Vec<*const std::os::raw::c_char> = enabled_extensions_cstr.iter()
            .map(|cstr| cstr.as_ptr())
            .collect(); */

    let enabled_layer_names: Vec<*const std::os::raw::c_char> = vec![];

    let application_name_cstr = CString::new(application_name.clone()).unwrap();
    let engine_name_cstr = CString::new(application_name.clone()).unwrap();

    let appinfo = vk::ApplicationInfo::builder()
            .application_name(&application_name_cstr)
            .application_version(ash::vk::make_api_version(0, 0, 0, 0))
            .engine_name(&engine_name_cstr)
            .engine_version(ash::vk::make_api_version(0, 0, 0, 0))
            .api_version(vk::make_api_version(0, 1, 1, 0))
            .build();

    let vulkan_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&extensions)
            .flags(ash::vk::InstanceCreateFlags::empty())
            .build();

    unsafe { entry.create_instance(&vulkan_create_info, None) }
}

struct PhysicalDevice {
    handle: ash::vk::PhysicalDevice,
    name: String,
    device_type: PhysicalDeviceType,
}

fn select_physical_device(instance: &ash::Instance, surface_loader: &ash::extensions::khr::Surface, surface: ash::vk::SurfaceKHR) -> Option<(PhysicalDevice, usize)> {
    unsafe {
        let physical_devices = instance.enumerate_physical_devices().unwrap();
        physical_devices.into_iter()
            .filter_map(|physical_device| {
                let physical_device_properties = instance.get_physical_device_properties(physical_device);
                let name = CStr::from_ptr(physical_device_properties.device_name.as_ptr()).to_str().unwrap();
                info!["Considering physical device {}", name];

                let queue_family_properties = instance.get_physical_device_queue_family_properties(physical_device);
                queue_family_properties.iter().enumerate()
                        .filter_map(|(queue_family_index, queue_info)| {
                            let supports_graphics = queue_info.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                            let supports_surface = surface_loader.get_physical_device_surface_support(
                                        physical_device, queue_family_index as u32, surface).unwrap();
                            
                            if supports_graphics && supports_surface {
                                let device = PhysicalDevice {
                                    handle: physical_device,
                                    device_type: physical_device_properties.device_type,
                                    name: String::from(name),
                                };
                                return Some((device, queue_family_index))
                            }
                            None
                        })
                        .next()
            })
            .min_by_key(|(physical_device, _queue_family_index)| match physical_device.device_type {
                PhysicalDeviceType::DISCRETE_GPU => 0,
                PhysicalDeviceType::INTEGRATED_GPU => 1,
                PhysicalDeviceType::VIRTUAL_GPU => 2,
                PhysicalDeviceType::CPU => 3,
                PhysicalDeviceType::OTHER => 4,
                _ => 99,
            })
    }
}

fn create_device(instance: &ash::Instance, physical_device: &PhysicalDevice, queue_family_index: usize) -> ash::Device {
    let device_extension_names_raw = [
        Swapchain::name().as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        KhrPortabilitySubsetFn::name().as_ptr(),
    ];
    let features = vk::PhysicalDeviceFeatures {
        // shader_clip_distance: 1,
        ..Default::default()
    };
    let priorities = [1.0];

    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index as u32)
        .queue_priorities(&priorities)
        .build();

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&device_extension_names_raw)
        .enabled_features(&features)
        .build();

    unsafe {
        instance.create_device(physical_device.handle, &device_create_info, None)
    }.unwrap()
}

fn get_required_extensions(window: &Window) -> Vec<*const std::os::raw::c_char> {
    let mut extensions = vec![];
    for window_extension in ash_window::enumerate_required_extensions(&window).unwrap() {
        extensions.push(*window_extension);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // extensions.push(KhrPortabilityEnumerationFn::name().as_ptr());
        // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
        extensions.push(KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
    }

    extensions
}

fn create_swapchain(
    instance: &ash::Instance, 
    physical_device: &PhysicalDevice,
    device: &ash::Device,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    window_dimensions: (i32, i32)) -> (vk::SwapchainKHR, Vec<vk::Image>, Swapchain)
{
    unsafe {
        let swapchain_loader = Swapchain::new(&instance, &device);

        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(physical_device.handle, surface)
            .unwrap();
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // TODO(knielsen): Query this dynamically
        let desired_image_count = 2;

        let surface_format = surface_loader
            .get_physical_device_surface_formats(physical_device.handle, surface)
            .unwrap()[0];

        let surface_capabilities = surface_loader
            .get_physical_device_surface_capabilities(physical_device.handle, surface)
            .unwrap();
        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D {
                width: window_dimensions.0 as u32,
                height: window_dimensions.1 as u32,
            },
            _ => surface_capabilities.current_extent,
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .build();

        let swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None).unwrap();

        let images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
        /*
        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image)
                    .build();
                device.create_image_view(&create_view_info, None).unwrap()
            })
            .collect(); */

        (swapchain, images, swapchain_loader)
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Skia - Test")
        .build(&event_loop)
        .unwrap();
    let dimensions: (i32, i32) = window.inner_size().into();

    let vulkan_entry = unsafe { ash::Entry::load().unwrap() };
    let physical_device_extensions = get_required_extensions(&window);
    let instance = create_instance(String::from("skia-app"), &physical_device_extensions, &vulkan_entry).unwrap();

    let surface = unsafe { ash_window::create_surface(&vulkan_entry, &instance, &window, None) }.unwrap();
    let surface_loader = Surface::new(&vulkan_entry, &instance);

    let (physical_device, queue_family_index) = select_physical_device(&instance, &surface_loader, surface).unwrap();
    info!["Selected physical device {} with queue family index {}", physical_device.name, queue_family_index];

    let device = create_device(&instance, &physical_device, queue_family_index);

    let present_queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

    let (swapchain, swapchain_images, swapchain_loader) = create_swapchain(
        &instance, &physical_device, &device, &surface_loader, surface, dimensions);


    // Renderpass
    let renderpass_attachments = [
        vk::AttachmentDescription {
            format: vk::Format::B8G8R8A8_UNORM, // TODO(knielsen): Fetch from swapchain
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        },
    ];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let subpass = vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_refs)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build();

    let renderpass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&renderpass_attachments)
        .subpasses(&[subpass])
        .build();

    let renderpass = unsafe { device.create_render_pass(&renderpass_create_info, None) }.unwrap();

    /*
    let framebuffers: Vec<vk::Framebuffer> = swapchain_images.iter()
        .map(|&present_image_view| {
            let framebuffer_attachments = [present_image_view];
            let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(&framebuffer_attachments)
                .width(dimensions.0 as u32)
                .height(dimensions.1 as u32)
                .layers(1)
                .build();

            unsafe { device.create_framebuffer(&frame_buffer_create_info, None) }.unwrap()
        })
        .collect(); */

    const MAX_FRAMES_IN_FLIGHT: u32 = 2;
    let (command_buffers, command_buffer_fences) = {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index as u32)
            .build();

        let pool = unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .build();
        
        let command_buffers = unsafe { device.allocate_command_buffers(&command_buffer_allocate_info).unwrap() };

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();
        
        let fences: Vec<_> = command_buffers.iter()
            .map(|_command_buffer| unsafe { device.create_fence(&fence_create_info, None).unwrap() })
            .collect();

        (command_buffers, fences)
    };

    // Skia
    let get_proc = |of| unsafe {
        match skia_get_proc(&vulkan_entry, instance.fp_v1_0(), of) {
            Some(f) => f as _,
            None => {
                error!("resolve of {} failed", of.name().to_str().unwrap());
                std::ptr::null()
            }
        }
    };

    let skia_backend = unsafe {
        skia_safe::gpu::vk::BackendContext::new(
            instance.handle().as_raw() as _,
            physical_device.handle.as_raw() as _,
            device.handle().as_raw() as _,
            (
                present_queue.as_raw() as _,
                queue_family_index,
            ),
            &get_proc
        )
    };

    let mut skia_context = skia_safe::gpu::DirectContext::new_vulkan(&skia_backend, None).unwrap();

    let skia_image_info = skia_safe::ImageInfo::new_n32_premul(dimensions, None);
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
    
    let skia_image = get_image_from_skia_texture(&skia_texture);

    let mut canvas = skia_surface.canvas();
    canvas.clear(skia_safe::Color4f::new(1.0, 1.0, 1.0, 1.0));
    let paint = skia_safe::Paint::new(skia_safe::Color4f::new(1.0, 0.0, 0.0, 1.0), None);
    canvas.draw_circle(skia_safe::Point::new(200.0, 200.0), 100.0, &paint);

    skia_surface.flush_and_submit();

    let (image_available_semaphores, render_finished_semaphores) = {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let image_available_semaphores: Vec<_> = command_buffers.iter()
            .map(|_command_buffer| unsafe { device.create_semaphore(&semaphore_create_info, None) }.unwrap())
            .collect();
        let render_finished_semaphorese: Vec<_> = command_buffers.iter()
            .map(|_command_buffer| unsafe { device.create_semaphore(&semaphore_create_info, None) }.unwrap())
            .collect();

        (image_available_semaphores, render_finished_semaphorese)
    };

    /*
    let graphics_pipeline = {
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: dimensions.0 as f32,
            height: dimensions.1 as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Extent2D {
            width: dimensions.0 as u32,
            height: dimensions.1 as u32,
        }.into()];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .scissors(&scissors)
            .viewports(&viewports)
            .build();
        
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .build();

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];
        let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states)
            .build();

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder()
                .dynamic_states(&dynamic_state)
                .build();
        
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .build();
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_create_info, None).unwrap() };
        
        let graphics_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .viewport_state(&viewport_state_info)
            .multisample_state(&multisample_state_info)
            .color_blend_state(&color_blend_state_info)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(renderpass)
            .build();

        unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_pipeline_info], None).unwrap()[0]
        }
    }; */

    let mut command_buffer_index: u32 = 0;
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
                // window_resized = true;
            }
            Event::MainEventsCleared => {
                let command_buffer = command_buffers[command_buffer_index as usize];
                let command_buffer_fence = command_buffer_fences[command_buffer_index as usize];
                let image_available_semaphore = image_available_semaphores[command_buffer_index as usize];
                let render_finished_semaphore = render_finished_semaphores[command_buffer_index as usize];

                let (present_index, _) = unsafe {
                    swapchain_loader.acquire_next_image(swapchain, std::u64::MAX, image_available_semaphore, vk::Fence::null())
                }.unwrap();
                let image = swapchain_images[present_index as usize];

                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                ];

                /*
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(renderpass)
                    .framebuffer(framebuffers[present_index as usize])
                    .render_area(vk::Extent2D {
                        width: dimensions.0 as u32,
                        height: dimensions.1 as u32,
                    }.into())
                    .clear_values(&clear_values)
                    .build(); */

                unsafe {
                    // Wait and reset
                    device.wait_for_fences(&[command_buffer_fence], true, std::u64::MAX).unwrap();
                    device.reset_fences(&[command_buffer_fence]).unwrap();

                    device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES).unwrap();
                    
                    // Send commands
                    let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                        .build();
                    device.begin_command_buffer(command_buffer, &command_buffer_begin_info).unwrap();

                    // device.cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                    // device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline);
                    // device.cmd_end_render_pass(command_buffer);
                    
                    let skia_barrier = vk::ImageMemoryBarrier::builder()
                        .image(skia_image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::NONE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .subresource_range(vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .build())
                        .build();

                    device.cmd_pipeline_barrier(command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(), &[], &[], &[skia_barrier]);

                    let barrier_test = vk::ImageMemoryBarrier::builder()
                        .image(image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .subresource_range(vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .build())
                        .build();

                    device.cmd_pipeline_barrier(command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(), &[], &[], &[barrier_test]);
                    
                    device.cmd_copy_image(command_buffer, skia_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[
                                vk::ImageCopy::builder()
                                    .src_offset(vk::Offset3D::default())
                                    .src_subresource(vk::ImageSubresourceLayers::builder()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .layer_count(1)
                                        .build())
                                    .dst_offset(vk::Offset3D::default())
                                    .dst_subresource(vk::ImageSubresourceLayers::builder()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .layer_count(1)
                                        .build())
                                    .extent(vk::Extent3D::builder()
                                        .width(dimensions.0 as u32)
                                        .height(dimensions.1 as u32)
                                        .depth(1)
                                        .build())
                                    .build()
                            ]);

                    let barrier = vk::ImageMemoryBarrier::builder()
                        .image(image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .subresource_range(vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .build())
                        .build();

                    device.cmd_pipeline_barrier(command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(), &[], &[], &[barrier]);
                    
                    device.end_command_buffer(command_buffer).unwrap();
                    
                    // Submit
                    let submit_info = vk::SubmitInfo::builder()
                        .wait_semaphores(&[image_available_semaphore])
                        // .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                        .command_buffers(&[command_buffer])
                        .signal_semaphores(&[render_finished_semaphore])
                        .build();
                    
                    device.queue_submit(present_queue, &[submit_info], command_buffer_fence).unwrap();
                };

                let wait_semaphores = [render_finished_semaphore];
                let swapchains = [swapchain];
                let image_indices = [present_index];
                let present_info = vk::PresentInfoKHR::builder()
                    .wait_semaphores(&wait_semaphores) // &base.rendering_complete_semaphore)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices)
                    .build();

                unsafe { swapchain_loader.queue_present(present_queue, &present_info) }.unwrap();

                command_buffer_index = (command_buffer_index + 1) % MAX_FRAMES_IN_FLIGHT;
            },
            _ => (),
        }
    });


    // Make sure the vulkan_entry and instance is valid all the way through program execution
    // drop(get_proc);
    // drop(instance);
    // drop(vulkan_entry);

    /*

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
    */
}
