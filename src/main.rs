#![allow(dead_code, unused_variables, clippy::too_many_arguments, clippy::unnecessary_wraps)]
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder};
use anyhow::{Result};

use vulkanalia::prelude::v1_0::*;

use crate::vulkan_structs::app::App;

mod vulkan_structs;
mod questions;

fn main() -> Result<()> {
    pretty_env_logger::init();
    //Window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("vulkan tutorial rust")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    //App

    let mut app = unsafe {App::create(&window)?};
    let mut destroying = false;
    event_loop.run(move |event, _, control_flow|{
        *control_flow = ControlFlow::Poll;
        match event {
           Event::MainEventsCleared if ! destroying =>
               unsafe {app.render(&window)}.unwrap(),
           Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => {
               destroying = true;
               *control_flow = ControlFlow::Exit;
               unsafe {app.device.device_wait_idle().unwrap();}
               unsafe {app.destroy();}
           }
            Event::WindowEvent {event: WindowEvent::Resized(_), ..} => app.resized = true,
           _ => {}
        }
    });
}
