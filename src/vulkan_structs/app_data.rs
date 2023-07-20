use vulkanalia::prelude::v1_0::*;
#[derive(Clone, Debug, Default)]
pub struct AppData{
    //Debug
    pub messenger: vk::DebugUtilsMessengerEXT,
    //surface
    pub surface: vk::SurfaceKHR,
    //Physical Device / Logical Device
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub presentation_queue: vk::Queue,
    //Swapchain
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    // PIPELINE LAYOUT TO USE UNIFORMS
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    //framebuffers
    pub framebuffers: Vec<vk::Framebuffer>,
    //command pool
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    //semaphores
    pub image_available_semaphore: Vec<vk::Semaphore>,
    pub render_finished_semaphore: Vec<vk::Semaphore>,
    //fences
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}