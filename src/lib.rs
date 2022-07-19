use std::marker::PhantomData;
use std::sync::Arc;
use std::ffi::{ CStr, CString };
use std::sync::Mutex;

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

unsafe fn skia_get_proc_impl(
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

fn select_physical_device(instance: &ash::Instance, surface_loader: &ash::extensions::khr::Surface, surfaces: &[ash::vk::SurfaceKHR]) -> Option<(PhysicalDevice, usize)> {
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
                            let supports_all_surfaces = surfaces.iter()
                                .all(|surface| surface_loader.get_physical_device_surface_support(
                                        physical_device, queue_family_index as u32, *surface).unwrap());

                            if supports_graphics && supports_all_surfaces {
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

enum WindowManager {
    Windows,
    Wayland,
    Xlib,
    Android,
    Macos,
    Ios
}

impl WindowManager {
    fn get_platform_window_manager() -> WindowManager {
        #[cfg(target_os = "windows")]
        return WindowManager::Windows;

        #[cfg(
            all(
                feature = "wayland",
                any(
                    target_os = "linux",
                    target_os = "dragonfly",
                    target_os = "freebsd",
                    target_os = "netbsd",
                    target_os = "openbsd"
                )
            )
        )]
        return WindowManager::Wayland;

        #[cfg(
            all(
                feature = "x11",
                any(
                    target_os = "linux",
                    target_os = "dragonfly",
                    target_os = "freebsd",
                    target_os = "netbsd",
                    target_os = "openbsd"
                )
            )
        )]
        return WindowManager::X11;

        #[cfg(any(target_os = "android"))]
        return WindowManager::Android;

        #[cfg(any(target_os = "macos"))]
        return WindowManager::Macos;

        #[cfg(any(target_os = "ios"))]
        return WindowManager::Ios;
    }
}

pub fn enumerate_required_extensions() -> Vec<*const std::os::raw::c_char> {
    match WindowManager::get_platform_window_manager() {
        WindowManager::Windows => {
            const WINDOWS_EXTS: [*const std::os::raw::c_char; 2] = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::khr::Win32Surface::name().as_ptr(),
            ];
            WINDOWS_EXTS.to_vec()
        }
        WindowManager::Wayland => {
            const WAYLAND_EXTS: [*const std::os::raw::c_char; 2] = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::khr::WaylandSurface::name().as_ptr(),
            ];
            WAYLAND_EXTS.to_vec()
        }
        WindowManager::Xlib => {
            const XLIB_EXTS: [*const std::os::raw::c_char; 2] = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::khr::XlibSurface::name().as_ptr(),
            ];
            XLIB_EXTS.to_vec()
        }
        WindowManager::Android => {
            const ANDROID_EXTS: [*const std::os::raw::c_char; 2] = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::khr::AndroidSurface::name().as_ptr(),
            ];
            ANDROID_EXTS.to_vec()
        }
        WindowManager::Macos => {
            const MACOS_EXTS: [*const std::os::raw::c_char; 3] = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::ext::MetalSurface::name().as_ptr(),
                KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
            ];
            MACOS_EXTS.to_vec()
        },
        WindowManager::Ios => {
            const IOS_EXTS: [*const std::os::raw::c_char; 3] = [
                ash::extensions::khr::Surface::name().as_ptr(),
                ash::extensions::ext::MetalSurface::name().as_ptr(),
                KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
            ];
            IOS_EXTS.to_vec()
        }
    }
}

fn get_required_extensions() -> Vec<*const std::os::raw::c_char> {
    let mut extensions = vec![];
    for window_extension in enumerate_required_extensions() {
        extensions.push(window_extension);
    }

    extensions
}

fn create_swapchain(
    instance: &ash::Instance, 
    physical_device: &PhysicalDevice,
    device: &ash::Device,
    swapchain_loader: &Swapchain,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    window_dimensions: (i32, i32)) -> (vk::SwapchainKHR, Vec<vk::Image>)
{
    unsafe {
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

        (swapchain, images)
    }
}

struct CommandBufferResources {
    command_buffer: ash::vk::CommandBuffer,
    available_fence: ash::vk::Fence,
    image_available_semaphore: ash::vk::Semaphore,
    rendering_finished_semaphore: ash::vk::Semaphore,

    skia_surface: skia_safe::Surface,
    skia_image: ash::vk::Image,
}

pub struct VulkanInstance<'a> {
    vulkan_entry: Arc<ash::Entry>,
    instance: Arc<ash::Instance>,
    surface_loader: ash::extensions::khr::Surface,

    skia_get_proc: Box<dyn skia_safe::gpu::vk::GetProc + 'a>,
}

impl<'a> VulkanInstance<'a> {
    pub fn new() -> VulkanInstance<'a> {
        let vulkan_entry = Arc::new(unsafe { ash::Entry::load().unwrap() });
        let physical_device_extensions = get_required_extensions();
        let instance = Arc::new(
            create_instance(String::from("skia-app"), &physical_device_extensions, &vulkan_entry).unwrap());

        let vulkan_entry_clone = vulkan_entry.clone();
        let instance_clone = instance.clone();
        let skia_get_proc: Box<dyn skia_safe::gpu::vk::GetProc + 'a> = Box::new(move |of| unsafe {
            match skia_get_proc_impl(&vulkan_entry_clone, &instance_clone.fp_v1_0(), of) {
                Some(f) => f as _,
                None => {
                    error!("resolve of {} failed", of.name().to_str().unwrap());
                    std::ptr::null()
                }
            }
        });

        let surface_loader = Surface::new(&vulkan_entry, &instance);

        VulkanInstance {
            vulkan_entry,
            instance,
            surface_loader,

            skia_get_proc,
        }
    }
}

pub struct StaticWindowsResources<'a> {
    vulkan: &'a VulkanInstance<'a>,
    physical_device: PhysicalDevice,
    device: ash::Device,
    present_queue: ash::vk::Queue,
    queue_family_index: usize,
    swapchain_loader: Swapchain,

    // TODO(knielsen): Is there a way to get rid of this mutex?
    //                 It should not be too bad, as the resources are only used for swapchain re-creation
    skia_context: Arc<Mutex<skia_safe::gpu::DirectContext>>,

    window_resources: std::collections::HashMap<winit::window::WindowId, StaticWindowResources<'a>>,
}

impl<'a> StaticWindowsResources<'a> {
    pub fn construct(vulkan: &'a VulkanInstance<'a>, windows: &'a [&Window]) -> StaticWindowsResources<'a> {
        let vulkan_entry = vulkan.vulkan_entry.clone();
        let instance = vulkan.instance.clone();

        let surfaces: Vec<_> = windows.iter()
            .map(|window| unsafe { ash_window::create_surface(&vulkan_entry, &instance, &window, None) }.unwrap())
            .collect();

        let (physical_device, queue_family_index) = select_physical_device(&instance, &vulkan.surface_loader, &surfaces).unwrap();
        info!["Selected physical device {} with queue family index {}", physical_device.name, queue_family_index];

        let device = create_device(&instance, &physical_device, queue_family_index);
        let swapchain_loader = Swapchain::new(&instance, &device);

        let present_queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        // Skia
        let skia_backend: skia_safe::gpu::vk::BackendContext<'a> = unsafe {
            skia_safe::gpu::vk::BackendContext::new(
                instance.handle().as_raw() as _,
                physical_device.handle.as_raw() as _,
                device.handle().as_raw() as _,
                (
                    present_queue.as_raw() as _,
                    queue_family_index,
                ),
                &(vulkan.skia_get_proc)
            )
        };

        let window_resources: std::collections::HashMap<_, _> = std::iter::zip(windows.iter(), surfaces.into_iter())
            .map(|(window, surface)| {
                let static_window_resources = StaticWindowResources::create(window, &device, queue_family_index as u32, surface);
                (window.id(), static_window_resources)
            })
            .collect();

        let skia_context: skia_safe::gpu::DirectContext = skia_safe::gpu::DirectContext::new_vulkan(
            &skia_backend, None).unwrap();

        StaticWindowsResources {
            vulkan,

            physical_device,
            device,
            present_queue,

            skia_context: Arc::new(Mutex::new(skia_context)),

            window_resources,
            queue_family_index,

            swapchain_loader,
        }
    }

    fn for_window(self: &Self, window: &Window) -> &StaticWindowResources<'a> {
        self.window_resources.get(&window.id()).expect("Tried to use a window that was not registered for Skia!")
    }
}

struct StaticWindowResources<'a> {
    window: &'a Window,
    command_buffers: Vec<vk::CommandBuffer>,
    command_buffer_fences: Vec<vk::Fence>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    surface: ash::vk::SurfaceKHR,
    queue_family_index: u32,
}

struct DynamicWindowResources<'a, T: 'a> {
    swapchain: ash::vk::SwapchainKHR,
    swapchain_images: Vec<ash::vk::Image>,
    current_dimensions: (u32, u32),
    command_buffers: Vec<CommandBufferResources>,

    phantom: PhantomData<&'a T>,
}

impl<'a> StaticWindowResources<'a> {
    fn create(window: &'a Window, device: &ash::Device, queue_family_index: u32, surface: ash::vk::SurfaceKHR) -> StaticWindowResources<'a> {
        let (command_buffers, command_buffer_fences) = {
            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index)
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

        StaticWindowResources {
            window,
            surface,
            command_buffers,
            command_buffer_fences,
            image_available_semaphores,
            render_finished_semaphores,
            queue_family_index,
        }
    }
}

impl<'a, T> DynamicWindowResources<'a, T> {
    fn create(static_resources: &StaticWindowsResources<'a>, window: &Window, sample_count: usize) -> DynamicWindowResources<'a, T> {
        let vulkan = static_resources.vulkan;
        let instance = &vulkan.instance;
        let physical_device = &static_resources.physical_device;
        let device = &static_resources.device;
        let swapchain_loader = &static_resources.swapchain_loader;
        let surface_loader = &vulkan.surface_loader;
        let window_dimension = window.inner_size();

        let mut skia_context = static_resources.skia_context.lock().unwrap();

        let skia_targets: Vec<_> = {
            let skia_image_info = skia_safe::ImageInfo::new(
                skia_safe::ISize::new(window_dimension.width as i32, window_dimension.height as i32),
                skia_safe::ColorType::BGRA8888,
                skia_safe::AlphaType::Premul,
                None);

            (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_frame_index| {
                    let mut skia_surface = skia_safe::Surface::new_render_target(
                        &mut skia_context,
                        skia_safe::Budgeted::Yes,
                        &skia_image_info,
                        Some(sample_count),
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

        let window_resources = static_resources.for_window(window);
        let surface = window_resources.surface;

        let (swapchain, swapchain_images) = create_swapchain(
            &instance, &physical_device, &device, &swapchain_loader, &surface_loader, surface, window_dimension.into());

        // TODO(knielsen): Construct this in a nicer way
        let command_buffers = (0..MAX_FRAMES_IN_FLIGHT as usize)
            .map(|index| {
                CommandBufferResources {
                    command_buffer: window_resources.command_buffers[index],
                    available_fence: window_resources.command_buffer_fences[index],
                    image_available_semaphore: window_resources.image_available_semaphores[index],
                    rendering_finished_semaphore: window_resources.render_finished_semaphores[index],
                    skia_surface: skia_targets[index].0.clone(),
                    skia_image: skia_targets[index].1,
                }
            })
            .collect();
        
        DynamicWindowResources {
            swapchain,
            swapchain_images,
            current_dimensions: window_dimension.into(),
            command_buffers,

            phantom: PhantomData,
        }
    }
}

pub struct WindowRenderer<'a> {
    window: &'a Window,
    static_resources: &'a StaticWindowsResources<'a>,
    static_window_resources: &'a StaticWindowResources<'a>,
    dynamic_resources: DynamicWindowResources<'a, ()>,

    sample_count: usize,

    should_recreate_swapchain: bool,
    command_buffer_index: usize,
}

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
impl<'a> WindowRenderer<'a> {
    pub fn construct(static_resources: &'a StaticWindowsResources<'a>, window: &'a Window) -> WindowRenderer<'a> {
        let sample_count = 8;
        let dynamic_resources = DynamicWindowResources::create(static_resources, &window, sample_count);
        let static_window_resources = static_resources.for_window(&window);

        WindowRenderer {
            window,
            static_resources,
            static_window_resources,
            dynamic_resources,

            sample_count,

            should_recreate_swapchain: false,
            command_buffer_index: 0,
        }
    }

    pub fn draw(self: &mut Self, window_dimensions: (u32, u32), user_draw: &dyn Fn(&mut skia_safe::Canvas) -> ()) -> () {
        let device = &self.static_resources.device;
        
        if self.should_recreate_swapchain || self.dynamic_resources.current_dimensions != window_dimensions {
            info!["Re-creating the swapchain!"];
            unsafe { device.device_wait_idle() }.unwrap();
            self.dynamic_resources = DynamicWindowResources::create(self.static_resources, self.window, self.sample_count);
            self.should_recreate_swapchain = false;
        }

        let swapchain_loader = &self.static_resources.swapchain_loader;
        let present_queue = self.static_resources.present_queue;
        let swapchain = self.dynamic_resources.swapchain;
        let swapchain_images = &self.dynamic_resources.swapchain_images;

        let command_buffer_resources = &self.dynamic_resources.command_buffers[self.command_buffer_index];

        let mut skia_surface = command_buffer_resources.skia_surface.clone();

        unsafe {
            device.wait_for_fences(&[command_buffer_resources.available_fence], true, std::u64::MAX)
        }.unwrap();

        let (present_index, suboptimal) = unsafe {
            swapchain_loader.acquire_next_image(swapchain, std::u64::MAX, command_buffer_resources.image_available_semaphore, vk::Fence::null())
        }.unwrap();
        if suboptimal {
            self.should_recreate_swapchain = true;
            return;
        }

        unsafe {
            device.reset_fences(&[command_buffer_resources.available_fence]).unwrap();

            device.reset_command_buffer(command_buffer_resources.command_buffer, vk::CommandBufferResetFlags::RELEASE_RESOURCES).unwrap();
            
            // Send commands
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .build();
            device.begin_command_buffer(command_buffer_resources.command_buffer, &command_buffer_begin_info).unwrap();

            // Issue Skia draw commands
            let canvas = skia_surface.canvas();
            canvas.save();
            user_draw(canvas);
            canvas.restore();

            skia_surface.flush_and_submit();

            let skia_barrier = vk::ImageMemoryBarrier::builder()
                .image(command_buffer_resources.skia_image)
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

            let image = swapchain_images[present_index as usize];
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

            device.cmd_pipeline_barrier(command_buffer_resources.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[], &[skia_barrier, barrier_test]);
            
            device.cmd_copy_image(command_buffer_resources.command_buffer, command_buffer_resources.skia_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
                                .width(self.dynamic_resources.current_dimensions.0)
                                .height(self.dynamic_resources.current_dimensions.1)
                                .depth(1)
                                .build())
                            .build()
                    ]);

            let skia_finish_copy_barrier = vk::ImageMemoryBarrier::builder()
                .image(command_buffer_resources.skia_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(if self.sample_count == 1 {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL
                })
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

            device.cmd_pipeline_barrier(command_buffer_resources.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[], &[barrier, skia_finish_copy_barrier]);

            device.end_command_buffer(command_buffer_resources.command_buffer).unwrap();
            
            // Submit
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&[command_buffer_resources.image_available_semaphore])
                // .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                .command_buffers(&[command_buffer_resources.command_buffer])
                .signal_semaphores(&[command_buffer_resources.rendering_finished_semaphore])
                .build();
            
            device.queue_submit(present_queue, &[submit_info], command_buffer_resources.available_fence).unwrap();
        };

        let wait_semaphores = [command_buffer_resources.rendering_finished_semaphore];
        let swapchains = [swapchain];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores) // &base.rendering_complete_semaphore)
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .build();

        unsafe { swapchain_loader.queue_present(present_queue, &present_info) }.unwrap();

        self.command_buffer_index = (self.command_buffer_index + 1) % self.dynamic_resources.command_buffers.len();
    }
}
