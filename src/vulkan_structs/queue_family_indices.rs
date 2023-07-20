use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtension;

use crate::vulkan_structs::app_data::AppData;
use crate::vulkan_structs::suitability_error::SuitabilityError;

pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub presentation: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance
            .get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))//we look for the first one that can handle graphics queues
            .map(|i| i as u32);

        let mut presentation = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(physical_device, index as u32, data.surface)? {
                presentation = Some(index as u32);
                break;
            }
        }
        if let (Some(graphics), Some(presentation)) = (graphics, presentation) {
            Ok(Self { graphics, presentation } )
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families")))
        }
    }
}
