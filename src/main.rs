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

    let skia_targets: Vec<_> = {
        let skia_image_info = skia_safe::ImageInfo::new(
            dimensions,
            skia_safe::ColorType::BGRA8888,
            skia_safe::AlphaType::Premul,
            None);

        (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_frame_index| {
                let mut skia_surface = skia_safe::Surface::new_render_target(
                    &mut skia_context,
                    skia_safe::Budgeted::Yes,
                    &skia_image_info,
                    Some(8), // Sample count
                    skia_safe::gpu::SurfaceOrigin::TopLeft,
                    None, // Surface props
                    false
                ).unwrap();

                let skia_texture = skia_surface
                    .get_backend_texture(skia_safe::surface::BackendHandleAccess::FlushRead)
                    .as_ref().unwrap().clone();
                let skia_image = get_image_from_skia_texture(&skia_texture);

                (skia_surface, skia_image)
            })
            .collect()
    };

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

    let mut degrees = 0.0;

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

                let (skia_surface_ref, skia_image) = &skia_targets[command_buffer_index as usize];
                let mut skia_surface = skia_surface_ref.clone();

                let (present_index, _) = unsafe {
                    swapchain_loader.acquire_next_image(swapchain, std::u64::MAX, image_available_semaphore, vk::Fence::null())
                }.unwrap();
                let image = swapchain_images[present_index as usize];

                unsafe {
                    // Wait and reset
                    device.wait_for_fences(&[command_buffer_fence], true, std::u64::MAX).unwrap();
                    device.reset_fences(&[command_buffer_fence]).unwrap();

                    device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES).unwrap();
                    
                    // Send commands
                    let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                        .build();
                    device.begin_command_buffer(command_buffer, &command_buffer_begin_info).unwrap();
                    
                    let mut canvas = skia_surface.canvas();
                    canvas.clear(skia_safe::Color4f::new(1.0, 1.0, 1.0, 1.0));
                    let paint = skia_safe::Paint::new(skia_safe::Color4f::new(1.0, 0.0, 0.0, 1.0), None);
                    canvas.draw_circle(skia_safe::Point::new(200.0, 200.0), 100.0, &paint);
                    canvas.rotate(degrees, Some(skia_safe::Point::new(300.0, 300.0)));
                    degrees = (degrees + 0.01) % 360.0;

                    skia_surface.flush_and_submit();

                    let skia_barrier = vk::ImageMemoryBarrier::builder()
                        .image(*skia_image)
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
                    
                    device.cmd_copy_image(command_buffer, *skia_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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

                    let skia_finish_copy_barrier = vk::ImageMemoryBarrier::builder()
                        .image(*skia_image)
                        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .subresource_range(vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .build())
                        .build();

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
                        vk::DependencyFlags::empty(), &[], &[], &[barrier, skia_finish_copy_barrier]);

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
}
