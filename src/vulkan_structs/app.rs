use winit::window::{Window};
use anyhow::{anyhow, Result};
use log::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::Version;
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;
use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, PhysicalDeviceFeatures, KhrSwapchainExtension};

use crate::vulkan_structs::app_data::AppData;
use crate::vulkan_structs::queue_family_indices::QueueFamilyIndices;
use crate::vulkan_structs::suitability_error::SuitabilityError;
use crate::vulkan_structs::swapchain_support::{SwapchainSupport};

const PORTABILITY_MACOS_VERSION: Version = Version::new(1,3,216);
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
const MAX_FRAME_IN_FLIGHT: usize = 2;


#[derive(Clone, Debug)]
pub struct App{
    entry: Entry,
    instance: Instance,
    data: AppData,
    pub device: Device,
    frame: usize,
    pub resized: bool,
}

impl App {
    //public functions
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        // create the instance and make it so that it prints fucking error messages
        let instance = Self::create_instance(window, &entry, &mut data)?;
        // generate the surface to draw on
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        // pick a graphics card that suits the program
        Self::pick_physical_device(&instance, &mut data)?;
        let device = Self::create_logical_device(&entry, &instance, &mut data)?;
        // create the swapchain
        Self::create_swapchain(window, &instance, &device, &mut data)?;
        //image view
        Self::create_swapchain_image_views(&device, &mut data)?;
        //render pass
        Self::create_render_pass(&instance, &device, &mut data)?;
        //pipeline
        Self::create_pipeline(&device, &mut data)?;
        //framebuffers
        Self::create_framebuffers(&device, &mut data)?;
        //commnad pool
        Self::create_command_pool(&instance, &device, &mut data)?;
        Self::create_command_buffers(&device, &mut data)?;
        //sync
        Self::create_sync_objects(&device, &mut data)?;
        Ok(Self { entry, instance, data, device, frame: 0, resized: false})
    }
    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        info!("uhm excuse me what the actual fuck?");
        let in_flight_fence = self.data.in_flight_fences[self.frame];
        self.device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?;
        //acquire the image index and in case recreate the swapchain
        info!("is the problem here?");
        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphore[self.frame],
            vk::Fence::null(),
        );
        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !self.data.images_in_flight[image_index].is_null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index]],
                true,
                u64::MAX
            )?;
        }
        self.data.images_in_flight[image_index] = self.data.in_flight_fences[self.frame];
        let wait_semaphores = &[self.data.image_available_semaphore[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphore[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);
        self.device.reset_fences(&[in_flight_fence])?;
        self.device.queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;
        //presentation
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);
        let result = self.device.queue_present_khr(self.data.presentation_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        info!("code arrived here?");
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }
        self.device.queue_present_khr(self.data.presentation_queue, &present_info)?;
        self.frame = (self.frame + 1) % MAX_FRAME_IN_FLIGHT;
        Ok(())
    }
    pub unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.data.in_flight_fences.iter().for_each(|f| self.device.destroy_fence(*f, None));
        self.data.render_finished_semaphore.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.image_available_semaphore.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }
    //private functions

    // ===============================
    // ||                           || // the instance allow us to initialize the program,
    // ||         INSTANCE          || // basically telling it on which os is running and what it has to do to initialize
    // ||                           ||
    // ===============================
    unsafe fn create_instance(
        window: &Window,
        entry: &Entry,
        data: &mut AppData,
    ) -> Result<Instance>{

        let application_info = vk::ApplicationInfo::builder()
            .application_name(b"vulkan w rust POGGERS")
            .application_version(vk::make_version(1,0,0))
            .engine_name(b"no engine\0")
            .engine_version(vk::make_version(1,0,0))
            .api_version(vk::make_version(1,0,0));
        //validation layers
        let available_layers = entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();
        if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
            return Err(anyhow!("Validation layer requested but not supported"));
        }
        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };
        //Extensions
        let mut extensions = vk_window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();
        if VALIDATION_ENABLED {
            extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        }
        // fuck you macos
        let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            info!("Enabling extensions for macos portability");
            extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
            extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let mut info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(flags);
        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .user_callback(Some(debug_callback));
        if VALIDATION_ENABLED {
            info = info.push_next(&mut debug_info);
        }
        let instance = entry.create_instance(&info, None)?;
        if VALIDATION_ENABLED {
            data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
        }
        Ok(instance)
    }
    // ===============================
    // ||                           ||
    // ||      PHYSICAL DEVICE      || // we choose the graphics card (in this case the first one that suits us)
    // ||                           ||
    // ===============================
    unsafe fn pick_physical_device(
        instance: &Instance,
        data: &mut AppData,
    ) -> Result<()>{
        for physical_device in instance.enumerate_physical_devices()? {
            let properties = instance.get_physical_device_properties(physical_device);
            if let Err(error) = Self::check_physical_device(instance, data, physical_device) {
                warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
            } else {
                info!("Selected physical device (`{}`).", properties.device_name);
                data.physical_device = physical_device;
                return Ok(());
            }
        }
        Err(anyhow!("Failed to find suitable physical device (god damn it)"))
    }
    unsafe fn check_physical_device(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()>{
        QueueFamilyIndices::get(instance, data, physical_device)?;
        Self::check_physical_device_extensions(instance, physical_device)?;
        Ok(())
    }
    // ===============================
    // ||                           ||
    // ||      LOGICAL DEVICE       || // to interface with the graphics card
    // ||                           ||
    // ===============================
    unsafe fn create_logical_device(
        entry: &Entry,
        instance: &Instance,
        data: &mut AppData,
    ) -> Result<Device>{
        let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

        let mut unique_indices = HashSet::new();
        unique_indices.insert(indices.graphics);
        unique_indices.insert(indices.presentation);

        let queue_priorities = &[1.0];
        let queue_infos = unique_indices.iter().map(|i|{
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        }).collect::<Vec<_>>();
        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let mut extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|n| n.as_ptr())
            .collect::<Vec<_>>();

        if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
        }
        let features = PhysicalDeviceFeatures::builder();
        let info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);
        let device = instance.create_device(data.physical_device, &info, None)?;
        data.graphics_queue = device.get_device_queue(indices.graphics, 0);
        data.presentation_queue = device.get_device_queue(indices.presentation, 0);
        Ok(device)
    }
    // ===============================
    // ||                           ||
    // ||         SWAPCHAIN         || // series of images to present
    // ||                           ||
    // ===============================
    unsafe fn check_physical_device_extensions(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()> {
        // we check if all the extensions are there
        let extensions = instance
            .enumerate_device_extension_properties(physical_device, None)?
            .iter()
            .map(|e| e.extension_name)
            .collect::<HashSet<_>>();
        if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
            info!("YOOO, WE GOT THE SWAPCHAIN SUPPORT LETS GOOOOOOO");
            Ok(()) // here we basically check if there is the swapchain extension
            // DEVICE_EXTENSIONS only contains the name of the Swapchain extension so by doing this we can crash the program if the
            // graphics card does not support the swapchain
        }
        else {
            Err(anyhow!(SuitabilityError("Missing required device extensions.")))
        }
    }

    unsafe fn create_swapchain(
        window: &Window,
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
        let support = SwapchainSupport::get(instance, data, data.physical_device)?;

        let surface_format = Self::get_swapchain_surface_format(&support.formats);
        let presentation_mode = Self::get_swapchain_present_mode(&support.presentation_mode);
        let extent = Self::get_swapchain_extent(window, support.capabilities);

        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count != 0 && image_count > support.capabilities.max_image_count {
            image_count = support.capabilities.max_image_count;
        }

        let mut queue_family_indices = vec![];
        let image_sharing_mode = if indices.graphics != indices.presentation{
            queue_family_indices.push(indices.graphics);
            queue_family_indices.push(indices.presentation);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        // now we gotta fill in the struct >:(
        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(data.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)// always 1 unless it's 3D
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(presentation_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());
        data.swapchain = device.create_swapchain_khr(&info, None)?;
        data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
        data.swapchain_format = surface_format.format;
        data.swapchain_extent = extent;
        info!("problem here?");
        Ok(())
    }

    unsafe fn recreate_swapchain( // in case of resize
        &mut self,
        window: &Window
    ) -> Result<()>{
        self.device.device_wait_idle()?;
        Self::destroy_swapchain(self);
        Self::create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        Self::create_swapchain_image_views(&self.device, &mut self.data)?;
        Self::create_render_pass(&self.instance, &self.device, &mut self.data)?;
        Self::create_pipeline(&self.device, &mut self.data)?;
        Self::create_framebuffers(&self.device, &mut self.data)?;
        Self::create_command_buffers(&self.device, &mut self.data)?;
        self.data.images_in_flight.resize(self.data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }
    unsafe fn destroy_swapchain(&mut self) {
        self.device.free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

    fn get_swapchain_surface_format(
        formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        formats
            .iter()
            .cloned()
            .find(|f|{
                f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            }).unwrap_or_else(|| formats[0])
    }
    fn get_swapchain_present_mode(
        present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR{
        //we check if the MAILBOX presentation mode is available
        /*
        vk::PresentModeKHR::MAILBOX – This is another variation of the second mode.
        Instead of blocking the application when the queue is full, the images that are already queued are simply replaced with the newer ones.
        This mode can be used to render frames as fast as possible while still avoiding tearing,
        resulting in fewer latency issues than standard vertical sync. This is commonly known as "triple buffering",
        although the existence of three buffers alone does not necessarily mean that the framerate is unlocked.
         */
        // if there isn't the mailbox mode we choose FIFO
        // which basically means more latency and less updated frames but also less power usage
        present_modes
            .iter()
            .cloned()
            .find(|m| *m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }
    fn get_swapchain_extent(
        window: &Window,
        capabilities: vk::SurfaceCapabilitiesKHR,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            info!("WE GOT THE MAX VALUE BABY LETS GOOOO");
            capabilities.current_extent
        } else {
            let size = window.inner_size();
            let clamp = |min: u32, max:  u32, v: u32| min.max(max.min(v)); //what the fuck?
            vk::Extent2D::builder()
                .width(clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                    size.width,
                ))
                .height(clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                    size.height,
                ))
                .build()
            //dont ask me what this code means i have no fucking clue
            //probably it's useful to resize the window?
            //idfk
        }
    }

    // ===============================
    // ||                           ||
    // ||        IMAGE VIEW         || // the view of the image wtf am I going on about
    // ||                           ||
    // ===============================
    unsafe fn create_swapchain_image_views(
        device: &Device,
        data: &mut AppData
    ) -> Result<()>{
        data.swapchain_image_views = data.swapchain_images
            .iter()
            .map(|i|{
                let components = vk::ComponentMapping::builder()
                    .r(vk::ComponentSwizzle::IDENTITY)
                    .g(vk::ComponentSwizzle::IDENTITY)
                    .b(vk::ComponentSwizzle::IDENTITY)
                    .a(vk::ComponentSwizzle::IDENTITY);
                // no mipmapping ;-;
                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                //if we were working in a 3d ambient we would create a swapchain with multiple layers
                let info = vk::ImageViewCreateInfo::builder()
                    .image(*i)
                    .view_type(vk::ImageViewType::_2D)
                    .format(data.swapchain_format)
                    .components(components)
                    .subresource_range(subresource_range);
                device.create_image_view(&info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }
    // ===============================
    // ||                           ||
    // ||        THE PIPELINE       ||// to draw shit I guess
    // ||                           ||
    // ===============================
    unsafe fn create_pipeline(
        device: &Device,
        data: &mut AppData
    ) -> Result<()>{
        //assembling shaders
        let vert = include_bytes!("../../shaders/vert.spv");
        let frag = include_bytes!("../../shaders/frag.spv");
        let vert_shader_module = Self::create_shader_module(device, &vert[..])?;
        let frag_shader_module = Self::create_shader_module(device, &frag[..])?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0");
        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0");
        // vertex shit idfk what this is
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder(); // since we hardcoded the vertexes in the fucking shader we leave it as default
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST) //gotta experiment with this >:D
            .primitive_restart_enable(false);
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(data.swapchain_extent.width as f32)
            .height(data.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D{x: 0, y:0 })
            .extent(data.swapchain_extent);
        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL) // gotta play with this >:D
            .line_width(1.0) // gotta play with this too
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE) // see what counter clockwise does
            .depth_bias_enable(false);
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::_1);
        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        let attachments = &[attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);
        let dynamic_states = &[
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::LINE_WIDTH,
        ];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(dynamic_states);
        let layout_info = vk::PipelineLayoutCreateInfo::builder();
        data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;
        let stages = &[vert_stage, frag_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(data.pipeline_layout)
            .render_pass(data.render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null()) // optional
            .base_pipeline_index(-1); // optional
        data.pipeline = device.create_graphics_pipelines(
            vk::PipelineCache::null(), &[info], None
        )?.0[0];
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
        Ok(())
    }
    unsafe fn create_shader_module(
        device: &Device,
        bytecode: &[u8],
    ) -> Result<vk::ShaderModule>{
        let bytecode = Vec::<u8>::from(bytecode);
        let (prefix, code, suffix) = bytecode.align_to::<u32>();
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(anyhow!("Shader bytecode is not properly aligned."));
        }
        let info = vk::ShaderModuleCreateInfo::builder()
            .code_size(bytecode.len())
            .code(code);
        Ok(device.create_shader_module(&info, None)?)
    }
    // ===============================
    // ||                           ||
    // ||       RENDER PASSES       ||// basically debug :D
    // ||                           ||
    // ===============================
    unsafe fn create_render_pass(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(data.swapchain_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachments = &[color_attachment_ref];
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments);
        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        //initialize render passes
        let attachments = &[color_attachment];
        let subpasses = &[subpass];
        let dependencies = &[dependency];
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);
        data.render_pass = device.create_render_pass(&info, None)?;
        Ok(())
    }
    // ===============================
    // ||                           ||
    // ||       FRAMEBUFFERS        || // I dont fucking know
    // ||                           ||
    // ===============================
    unsafe fn create_framebuffers(
        device: &Device,
        data: &mut AppData,
    ) -> Result<()>{
        data.framebuffers = data
            .swapchain_image_views
            .iter()
            .map(|i|{
                let attachments = &[*i];
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(data.render_pass)
                    .attachments(attachments)
                    .width(data.swapchain_extent.width)
                    .height(data.swapchain_extent.height)
                    .layers(1);
                device.create_framebuffer(&create_info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }
    // ===============================
    // ||                           ||
    // ||       COMMAND POOL        || // I dont fucking know
    // ||                           ||
    // ===============================
    unsafe fn create_command_pool(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()>{
        let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
        let info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::empty()) // optional since it's empty
            .queue_family_index(indices.graphics);
        data.command_pool = device.create_command_pool(&info, None)?;

        Ok(())
    }
    unsafe fn create_command_buffers(
        device: &Device,
        data: &mut AppData,
    ) -> Result<()>{
        info!("creating command buffers");
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(data.framebuffers.len() as u32);
        data.command_buffers = device.allocate_command_buffers(&allocate_info)?;
        for (i, command_buffer) in data.command_buffers.iter().enumerate() {
            let inheritance = vk::CommandBufferInheritanceInfo::builder();
            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::empty()) //optional
                .inheritance_info(&inheritance); //optional
            device.begin_command_buffer(*command_buffer, &info)?;
            let render_area = vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(data.swapchain_extent);
            let color_clear_value = vk::ClearValue{
                color: vk::ClearColorValue{
                    float32: [0.0, 0.0, 0.0, 1.0],//background color :D
                },
            };
            let clear_values = &[color_clear_value];
            let info = vk::RenderPassBeginInfo::builder()
                .render_pass(data.render_pass)
                .framebuffer(data.framebuffers[i])
                .render_area(render_area)
                .clear_values(clear_values);
            device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
            device.cmd_draw(*command_buffer, 3, 1, 0, 0); // this is like opengl draw triangles or whatever
            device.cmd_end_render_pass(*command_buffer);
            device.end_command_buffer(*command_buffer)?;
        }
        Ok(())
    }
    // ===============================
    // ||                           ||
    // ||         SEMAPHORES        ||
    // ||                           ||
    // ===============================
    unsafe fn create_sync_objects(
        device: &Device,
        data: &mut AppData,
    ) -> Result<()>{
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..MAX_FRAME_IN_FLIGHT{
            data.image_available_semaphore.push(device.create_semaphore(&semaphore_info, None)?);
            data.render_finished_semaphore.push(device.create_semaphore(&semaphore_info, None)?);
            data.in_flight_fences.push(device.create_fence(&fence_info, None)?);
        }
        data.images_in_flight = data.swapchain_images
            .iter()
            .map(|_|vk::Fence::null())
            .collect();
        Ok(())
    }
}
// ===============================
// ||                           ||
// ||     VALIDATION LAYERS     ||// basically debug :D
// ||                           ||
// ===============================
extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}
