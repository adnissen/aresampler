//! Windows app icon fetching using Shell32 and GDI

use std::os::windows::ffi::OsStrExt;
use std::path::Path;

use image::codecs::png::PngEncoder;
use image::{ImageBuffer, ImageEncoder, Rgba};
use windows::core::PCWSTR;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Gdi::{
    CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, GetDC, ReleaseDC, SelectObject,
    BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS,
};
use windows::Win32::UI::Shell::ExtractIconExW;
use windows::Win32::UI::WindowsAndMessaging::{DestroyIcon, DrawIconEx, GetIconInfo, DI_NORMAL};

const ICON_SIZE: u32 = 32;

/// Fetches the application icon as PNG bytes for a given executable path.
/// Returns None if the icon cannot be fetched.
pub fn get_app_icon_png(exe_path: &Path) -> Option<Vec<u8>> {
    unsafe {
        // Extract icon from executable
        let hicon = extract_icon_from_exe(exe_path)?;

        // Convert HICON to RGBA pixels
        let pixels = hicon_to_rgba(hicon, ICON_SIZE)?;

        // Clean up icon handle
        let _ = DestroyIcon(hicon);

        // Encode as PNG
        rgba_to_png(pixels, ICON_SIZE, ICON_SIZE)
    }
}

unsafe fn extract_icon_from_exe(
    exe_path: &Path,
) -> Option<windows::Win32::UI::WindowsAndMessaging::HICON> {
    // Convert path to wide string (null-terminated)
    let path_wide: Vec<u16> = exe_path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let mut large_icons = [windows::Win32::UI::WindowsAndMessaging::HICON::default()];

    let count = ExtractIconExW(
        PCWSTR::from_raw(path_wide.as_ptr()),
        0, // First icon index
        Some(large_icons.as_mut_ptr()),
        None, // Don't need small icons
        1,
    );

    if count > 0 && !large_icons[0].is_invalid() {
        Some(large_icons[0])
    } else {
        None
    }
}

unsafe fn hicon_to_rgba(
    hicon: windows::Win32::UI::WindowsAndMessaging::HICON,
    size: u32,
) -> Option<Vec<u8>> {
    // Get icon info for cleanup later
    let mut icon_info = windows::Win32::UI::WindowsAndMessaging::ICONINFO::default();
    if GetIconInfo(hicon, &mut icon_info).is_err() {
        return None;
    }

    // Create device context
    let hdc_screen = GetDC(HWND::default());
    let hdc_mem = CreateCompatibleDC(hdc_screen);

    // Setup BITMAPINFO for 32-bit BGRA
    let bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: size as i32,
            biHeight: -(size as i32), // Negative = top-down DIB
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };

    // Create DIB section
    let mut bits_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let hbitmap = CreateDIBSection(hdc_mem, &bmi, DIB_RGB_COLORS, &mut bits_ptr, None, 0);

    let hbitmap = match hbitmap {
        Ok(bmp) => bmp,
        Err(_) => {
            cleanup_icon_info(&icon_info);
            let _ = DeleteDC(hdc_mem);
            let _ = ReleaseDC(HWND::default(), hdc_screen);
            return None;
        }
    };

    // Select bitmap into DC and draw icon
    let old_bitmap = SelectObject(hdc_mem, hbitmap);
    let _ = DrawIconEx(
        hdc_mem,
        0,
        0,
        hicon,
        size as i32,
        size as i32,
        0,
        None,
        DI_NORMAL,
    );

    // Copy pixel data
    let pixel_count = (size * size) as usize;
    let mut pixels = vec![0u8; pixel_count * 4];
    std::ptr::copy_nonoverlapping(bits_ptr as *const u8, pixels.as_mut_ptr(), pixels.len());

    // Convert BGRA to RGBA
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2); // Swap B and R
    }

    // Cleanup GDI objects
    SelectObject(hdc_mem, old_bitmap);
    let _ = DeleteObject(hbitmap);
    let _ = DeleteDC(hdc_mem);
    let _ = ReleaseDC(HWND::default(), hdc_screen);
    cleanup_icon_info(&icon_info);

    Some(pixels)
}

unsafe fn cleanup_icon_info(icon_info: &windows::Win32::UI::WindowsAndMessaging::ICONINFO) {
    if !icon_info.hbmColor.is_invalid() {
        let _ = DeleteObject(icon_info.hbmColor);
    }
    if !icon_info.hbmMask.is_invalid() {
        let _ = DeleteObject(icon_info.hbmMask);
    }
}

fn rgba_to_png(pixels: Vec<u8>, width: u32, height: u32) -> Option<Vec<u8>> {
    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, pixels)?;

    let mut png_bytes = Vec::new();
    let encoder = PngEncoder::new(&mut png_bytes);
    encoder
        .write_image(img.as_raw(), width, height, image::ExtendedColorType::Rgba8)
        .ok()?;

    Some(png_bytes)
}
