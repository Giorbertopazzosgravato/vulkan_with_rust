use anyhow::{Result};
use vulkanalia::{Instance, vk};
use vulkanalia::vk::{KhrSurfaceExtension};
use crate::vulkan_structs::app_data::AppData;

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub presentation_mode: Vec<vk::PresentModeKHR>
}
impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self{
            capabilities: instance.get_physical_device_surface_capabilities_khr(
                physical_device, data.surface
            )?,
            formats: instance.get_physical_device_surface_formats_khr(
                physical_device, data.surface
            )?,
            presentation_mode: instance
                .get_physical_device_surface_present_modes_khr(
                    physical_device, data.surface
                )?,
        })
    }
}