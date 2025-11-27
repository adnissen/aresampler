//! macOS app icon fetching using NSWorkspace

use objc2::rc::Retained;
use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep, NSImage, NSWorkspace};
use objc2_foundation::{NSDictionary, NSSize, NSString};

/// Fetches the application icon as PNG bytes for a given bundle identifier.
/// Returns None if the icon cannot be fetched (app not found, conversion fails, etc.)
pub fn get_app_icon_png(bundle_id: &str) -> Option<Vec<u8>> {
    unsafe {
        let workspace = NSWorkspace::sharedWorkspace();
        let bundle_id_ns = NSString::from_str(bundle_id);

        // Get the application URL from bundle ID
        let app_url = workspace.URLForApplicationWithBundleIdentifier(&bundle_id_ns)?;
        let app_path = app_url.path()?;

        // Get the icon for the application
        let icon: Retained<NSImage> = workspace.iconForFile(&app_path);

        // Set desired size (32x32 for good quality at 16-24px display)
        let size = NSSize {
            width: 32.0,
            height: 32.0,
        };
        icon.setSize(size);

        // Convert NSImage to PNG data via TIFF -> BitmapRep -> PNG
        let tiff_data = icon.TIFFRepresentation()?;
        let bitmap_rep = NSBitmapImageRep::imageRepWithData(&tiff_data)?;

        // Create an empty dictionary for properties (typed correctly)
        let props: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::new();
        let png_data =
            bitmap_rep.representationUsingType_properties(NSBitmapImageFileType::PNG, &props)?;

        // Get the bytes from NSData
        Some(png_data.to_vec())
    }
}
