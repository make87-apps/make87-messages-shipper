use make87::encodings::{Encoder, ProtobufEncoder};
use make87_messages::core::Header;
use make87_messages::google::protobuf::Timestamp;
use make87_messages::image::compressed::ImageJpeg;
use make87_messages::image::uncompressed::{image_raw_any, ImageNv12, ImageRawAny, ImageRgb888, ImageRgba8888, ImageYuv420, ImageYuv422, ImageYuv444};
use make87_messages::text::PlainText;
use regex::Regex;
use std::collections::HashMap;
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};
use turbojpeg::{Compressor, Image, PixelFormat, YuvImage, Subsamp};

fn timestamp_to_secs_f64(ts: &Timestamp) -> f64 {
    ts.seconds as f64 + (ts.nanos as f64 / 1_000_000_000.0)
}

fn get_current_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn ensure_leading_slash(entity_path: String) -> String {
    if entity_path.starts_with('/') {
        entity_path
    } else {
        format!("/{}", entity_path)
    }
}

fn process_header_and_set_time(header: &Option<Header>, rec: &rerun::RecordingStream) -> (String, f64) {
    let (entity_path, header_time) = match header {
        Some(header) => {
            let time = header.timestamp
                .map(|ts| timestamp_to_secs_f64(&ts))
                .unwrap_or_else(get_current_timestamp_secs);
            (ensure_leading_slash(header.entity_path.clone()), time)
        }
        None => ("/".to_string(), get_current_timestamp_secs()),
    };

    rec.set_timestamp_secs_since_epoch("header_time", header_time);
    (entity_path, header_time)
}

pub trait MessageHandler {
    fn handle_message(&self, sample: &zenoh::sample::Sample, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>>;
}

pub struct TextPlainTextHandler {
    encoder: ProtobufEncoder<PlainText>,
}

impl TextPlainTextHandler {
    pub fn new() -> Self {
        Self {
            encoder: ProtobufEncoder::<PlainText>::new(),
        }
    }
}

impl MessageHandler for TextPlainTextHandler {
    fn handle_message(&self, sample: &zenoh::sample::Sample, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        rec.log(
            entity_path,
            &rerun::TextDocument::new(message_decoded.body)
        )?;
        Ok(())
    }
}

pub struct ImageCompressedJpegHandler {
    encoder: ProtobufEncoder<ImageJpeg>,
}

impl ImageCompressedJpegHandler {
    pub fn new() -> Self {
        Self {
            encoder: ProtobufEncoder::<ImageJpeg>::new(),
        }
    }
}

impl MessageHandler for ImageCompressedJpegHandler {
    fn handle_message(&self, sample: &zenoh::sample::Sample, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);
        rec.log(
            entity_path,
            &rerun::EncodedImage::new(message_decoded.data).with_media_type(rerun::MediaType::from("image/jpeg"))
        )?;
        Ok(())
    }
}

// Trait for handling different image formats
trait ImageFormatHandler {
    fn log_to_rerun(&self, entity_path: String, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>>;
    fn get_format_name(&self) -> &'static str;
}

// Individual format handlers
struct Yuv420Handler<'a> {
    data: &'a ImageYuv420,
}

impl<'a> ImageFormatHandler for Yuv420Handler<'a> {
    fn log_to_rerun(&self, entity_path: String, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        // Convert YUV420 directly to JPEG (more efficient than YUV->RGB->JPEG)
        let jpeg_data = convert_yuv420_to_jpeg(self.data)?;

        rec.log(
            entity_path,
            &rerun::EncodedImage::new(jpeg_data).with_media_type(rerun::MediaType::from("image/jpeg"))
        )?;
        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "YUV420 (JPEG compressed)"
    }
}

// YUV420 to RGB conversion using yuvutils-rs
fn yuv420_to_rgb_with_yuvutils(yuv_data: &[u8], width: usize, height: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let y_size = width * height;
    let uv_size = y_size / 4;
    let expected_total_size = y_size + 2 * uv_size;

    if yuv_data.len() < expected_total_size {
        return Err(format!("Insufficient YUV420 data: expected {}, got {}", expected_total_size, yuv_data.len()).into());
    }

    let y_plane = &yuv_data[0..y_size];
    let u_plane = &yuv_data[y_size..y_size + uv_size];
    let v_plane = &yuv_data[y_size + uv_size..y_size + 2 * uv_size];

    let yuv_planar = yuvutils_rs::YuvPlanarImage {
        y_plane,
        y_stride: width as u32,
        u_plane,
        u_stride: (width / 2) as u32,
        v_plane,
        v_stride: (width / 2) as u32,
        width: width as u32,
        height: height as u32,
    };

    let mut rgb_data = vec![0u8; width * height * 3];

    // Use Limited range with Bt709 matrix (most common for modern video)
    yuvutils_rs::yuv420_to_rgb(
        &yuv_planar,
        &mut rgb_data,
        width as u32 * 3,
        yuvutils_rs::YuvRange::Limited,
        yuvutils_rs::YuvStandardMatrix::Bt709,
    ).map_err(|e| format!("YUV420 to RGB conversion failed: {:?}", e))?;

    Ok(rgb_data)
}

struct Rgb888Handler<'a> {
    data: &'a ImageRgb888,
}

impl<'a> ImageFormatHandler for Rgb888Handler<'a> {
    fn log_to_rerun(&self, entity_path: String, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        // Convert RGB888 to JPEG for bandwidth efficiency
        let jpeg_data = convert_rgb888_to_jpeg(self.data)?;

        rec.log(
            entity_path,
            &rerun::EncodedImage::new(jpeg_data).with_media_type(rerun::MediaType::from("image/jpeg"))
        )?;
        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "RGB888 (JPEG compressed)"
    }
}

struct Rgba8888Handler<'a> {
    data: &'a ImageRgba8888,
}

impl<'a> ImageFormatHandler for Rgba8888Handler<'a> {
    fn log_to_rerun(&self, entity_path: String, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        // Convert RGBA8888 to JPEG for bandwidth efficiency
        let jpeg_data = convert_rgba8888_to_jpeg(self.data)?;

        rec.log(
            entity_path,
            &rerun::EncodedImage::new(jpeg_data).with_media_type(rerun::MediaType::from("image/jpeg"))
        )?;
        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "RGBA8888 (JPEG compressed)"
    }
}

// Helper function to handle any image format
fn handle_image_format(handler: &dyn ImageFormatHandler, entity_path: String, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
    log::info!("Processing {} image", handler.get_format_name());
    handler.log_to_rerun(entity_path, rec)
}

// Handler for composite ImageRawAny messages
pub struct ImageRawAnyHandler {
    encoder: ProtobufEncoder<ImageRawAny>,
}

impl ImageRawAnyHandler {
    pub fn new() -> Self {
        Self {
            encoder: ProtobufEncoder::<ImageRawAny>::new(),
        }
    }
}

impl MessageHandler for ImageRawAnyHandler {
    fn handle_message(&self, sample: &zenoh::sample::Sample, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        // Handle the one-of field properly
        match &message_decoded.image {
            Some(image_raw_any::Image::Rgb888(rgb888)) => {
                let handler = Rgb888Handler { data: rgb888 };
                handle_image_format(&handler, entity_path, rec)?;
            }
            Some(image_raw_any::Image::Rgba8888(rgba8888)) => {
                let handler = Rgba8888Handler { data: rgba8888 };
                handle_image_format(&handler, entity_path, rec)?;
            }
            Some(image_raw_any::Image::Yuv420(yuv420)) => {
                let handler = Yuv420Handler { data: yuv420 };
                handle_image_format(&handler, entity_path, rec)?;
            }
            Some(image_raw_any::Image::Yuv422(_yuv422)) => {
                // TODO: Add YUV422 handler
                log::warn!("YUV422 format not yet implemented");
            }
            Some(image_raw_any::Image::Yuv444(_yuv444)) => {
                // TODO: Add YUV444 handler
                log::warn!("YUV444 format not yet implemented");
            }
            Some(image_raw_any::Image::Nv12(_nv12)) => {
                // TODO: Add NV12 handler
                log::warn!("NV12 format not yet implemented");
            }
            None => {
                return Err("No image format found in ImageRawAny message".into());
            }
        }

        Ok(())
    }
}

// Individual format message handlers (for when you receive specific formats directly)
pub struct ImageYuv420Handler {
    encoder: ProtobufEncoder<ImageYuv420>,
}

impl ImageYuv420Handler {
    pub fn new() -> Self {
        Self {
            encoder: ProtobufEncoder::<ImageYuv420>::new(),
        }
    }
}

impl MessageHandler for ImageYuv420Handler {
    fn handle_message(&self, sample: &zenoh::sample::Sample, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        let handler = Yuv420Handler { data: &message_decoded };
        handle_image_format(&handler, entity_path, rec)
    }
}

pub struct ImageRgb888Handler {
    encoder: ProtobufEncoder<ImageRgb888>,
}

impl ImageRgb888Handler {
    pub fn new() -> Self {
        Self {
            encoder: ProtobufEncoder::<ImageRgb888>::new(),
        }
    }
}

impl MessageHandler for ImageRgb888Handler {
    fn handle_message(&self, sample: &zenoh::sample::Sample, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        let handler = Rgb888Handler { data: &message_decoded };
        handle_image_format(&handler, entity_path, rec)
    }
}


pub struct ImageRgba8888Handler {
    encoder: ProtobufEncoder<ImageRgba8888>,
}

impl ImageRgba8888Handler {
    pub fn new() -> Self {
        Self {
            encoder: ProtobufEncoder::<ImageRgba8888>::new(),
        }
    }
}

impl MessageHandler for ImageRgba8888Handler {
    fn handle_message(&self, sample: &zenoh::sample::Sample, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        let handler = Rgba8888Handler { data: &message_decoded };
        handle_image_format(&handler, entity_path, rec)
    }
}


type HandlerFactory = fn() -> Box<dyn MessageHandler>;

pub struct MessageTypeRegistry {
    handlers: HashMap<&'static str, HandlerFactory>,
}

impl MessageTypeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            handlers: HashMap::new(),
        };

        // Register message types with their corresponding handlers
        registry.register("text-PlainText", || Box::new(TextPlainTextHandler::new()));
        registry.register("image-compressed-ImageJPEG", || Box::new(ImageCompressedJpegHandler::new()));

        // Register composite and individual image format handlers
        registry.register("image-uncompressed-ImageRawAny", || Box::new(ImageRawAnyHandler::new()));
        registry.register("image-uncompressed-ImageYUV420", || Box::new(ImageYuv420Handler::new()));
        registry.register("image-uncompressed-ImageRGB888", || Box::new(ImageRgb888Handler::new()));
        registry.register("image-uncompressed-ImageRGBA8888", || Box::new(ImageRgba8888Handler::new()));

        registry
    }

    fn register(&mut self, message_type: &'static str, factory: HandlerFactory) {
        self.handlers.insert(message_type, factory);
    }

    pub fn create_handler_from_topic_key(&self, topic_key: &str) -> Option<Box<dyn MessageHandler>> {
        let message_type = self.extract_message_type_from_topic_key(topic_key)?;
        let factory = self.handlers.get(message_type)?;
        Some(factory())
    }

    fn extract_message_type_from_topic_key<'a>(&self, topic_key: &'a str) -> Option<&'a str> {
        let re = Regex::new(r".*/.*/.*/make87_messages-([^/]+)/.*").ok()?;
        re.captures(topic_key)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str())
    }
}

// JPEG conversion functions for all image formats
fn convert_rgb888_to_jpeg(rgb888: &ImageRgb888) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut compressor = Compressor::new()?;
    let pixels = rgb888.data.as_slice();
    let width = rgb888.width as usize;
    let height = rgb888.height as usize;
    let pitch = width * 3;
    let image = Image {
        pixels,
        width,
        pitch,
        height,
        format: PixelFormat::RGB,
    };
    Ok(compressor.compress_to_vec(image)?)
}

fn convert_rgba8888_to_jpeg(rgba8888: &ImageRgba8888) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut compressor = Compressor::new()?;
    let pixels = rgba8888.data.as_slice();
    let width = rgba8888.width as usize;
    let height = rgba8888.height as usize;
    let pitch = width * 4;
    let image = Image {
        pixels,
        width,
        pitch,
        height,
        format: PixelFormat::RGBA,
    };
    Ok(compressor.compress_to_vec(image)?)
}

fn convert_yuv420_to_jpeg(yuv420: &ImageYuv420) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut compressor = Compressor::new()?;
    let width = yuv420.width as usize;
    let height = yuv420.height as usize;
    let yuv_data = yuv420.data.as_slice();
    let yuv_image = YuvImage {
        pixels: yuv_data,
        width,
        align: 1,
        height,
        subsamp: Subsamp::Sub2x2, // YUV420
    };
    Ok(compressor.compress_yuv_to_vec(yuv_image)?)
}

fn convert_yuv422_to_jpeg(yuv422: &ImageYuv422) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut compressor = Compressor::new()?;
    let width = yuv422.width as usize;
    let height = yuv422.height as usize;
    let yuv_data = yuv422.data.as_slice();
    let yuv_image = YuvImage {
        pixels: yuv_data,
        width,
        align: 1,
        height,
        subsamp: Subsamp::Sub2x1, // YUV422
    };
    Ok(compressor.compress_yuv_to_vec(yuv_image)?)
}

fn convert_yuv444_to_jpeg(yuv444: &ImageYuv444) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut compressor = Compressor::new()?;
    let width = yuv444.width as usize;
    let height = yuv444.height as usize;
    let yuv_data = yuv444.data.as_slice();
    let yuv_image = YuvImage {
        pixels: yuv_data,
        width,
        align: 1,
        height,
        subsamp: Subsamp::None, // YUV444 - no subsampling
    };
    Ok(compressor.compress_yuv_to_vec(yuv_image)?)
}

fn convert_nv12_to_jpeg(nv12: &ImageNv12) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut compressor = Compressor::new()?;
    let width = nv12.width as usize;
    let height = nv12.height as usize;
    let nv12_data = nv12.data.as_slice();

    // NV12 format: Y plane followed by interleaved UV plane
    let y_size = width * height;
    let uv_size = y_size / 2; // UV plane is half the size (2x2 subsampling)

    if nv12_data.len() < y_size + uv_size {
        return Err(format!("NV12 data too small: expected {}, got {}", y_size + uv_size, nv12_data.len()).into());
    }

    // Create planar YUV420 data
    let mut yuv420_data = Vec::with_capacity(y_size + uv_size);

    // Copy Y plane as-is
    yuv420_data.extend_from_slice(&nv12_data[0..y_size]);

    // Convert interleaved UV to separate U and V planes
    let uv_plane = &nv12_data[y_size..y_size + uv_size];

    // Extract U components (even indices in UV plane)
    for i in (0..uv_size).step_by(2) {
        yuv420_data.push(uv_plane[i]);
    }

    // Extract V components (odd indices in UV plane)
    for i in (1..uv_size).step_by(2) {
        yuv420_data.push(uv_plane[i]);
    }

    let yuv_image = YuvImage {
        pixels: yuv420_data.as_slice(),
        width,
        align: 1,
        height,
        subsamp: Subsamp::Sub2x2, // YUV420 (converted from NV12)
    };
    Ok(compressor.compress_yuv_to_vec(yuv_image)?)
}

// Universal conversion function for ImageRawAny (handles all formats)
fn convert_to_jpeg(image_raw_any: &ImageRawAny) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut compressor = Compressor::new()?;

    use image_raw_any::Image as RawImageVariant;

    let jpeg_data = match &image_raw_any.image {
        Some(RawImageVariant::Rgb888(rgb888)) => {
            let pixels = rgb888.data.as_slice();
            let width = rgb888.width as usize;
            let height = rgb888.height as usize;
            let pitch = width * 3;
            let image = Image {
                pixels,
                width,
                pitch,
                height,
                format: PixelFormat::RGB,
            };
            compressor.compress_to_vec(image)?
        }
        Some(RawImageVariant::Rgba8888(rgba8888)) => {
            let pixels = rgba8888.data.as_slice();
            let width = rgba8888.width as usize;
            let height = rgba8888.height as usize;
            let pitch = width * 4;
            let image = Image {
                pixels,
                width,
                pitch,
                height,
                format: PixelFormat::RGBA,
            };
            compressor.compress_to_vec(image)?
        }
        Some(RawImageVariant::Yuv420(yuv420)) => {
            let width = yuv420.width as usize;
            let height = yuv420.height as usize;
            let yuv_data = yuv420.data.as_slice();
            let yuv_image = YuvImage {
                pixels: yuv_data,
                width,
                align: 1,
                height,
                subsamp: Subsamp::Sub2x2, // YUV420
            };
            compressor.compress_yuv_to_vec(yuv_image)?
        }
        Some(RawImageVariant::Yuv422(yuv422)) => {
            let width = yuv422.width as usize;
            let height = yuv422.height as usize;
            let yuv_data = yuv422.data.as_slice();
            let yuv_image = YuvImage {
                pixels: yuv_data,
                width,
                align: 1,
                height,
                subsamp: Subsamp::Sub2x1, // YUV422
            };
            compressor.compress_yuv_to_vec(yuv_image)?
        }
        Some(RawImageVariant::Yuv444(yuv444)) => {
            let width = yuv444.width as usize;
            let height = yuv444.height as usize;
            let yuv_data = yuv444.data.as_slice();
            let yuv_image = YuvImage {
                pixels: yuv_data,
                width,
                align: 1,
                height,
                subsamp: Subsamp::None, // YUV444
            };
            compressor.compress_yuv_to_vec(yuv_image)?
        }
        Some(RawImageVariant::Nv12(nv12)) => {
            let width = nv12.width as usize;
            let height = nv12.height as usize;
            let nv12_data = nv12.data.as_slice();

            // NV12 format: Y plane followed by interleaved UV plane
            let y_size = width * height;
            let uv_size = y_size / 2; // UV plane is half the size (2x2 subsampling)

            if nv12_data.len() < y_size + uv_size {
                return Err(format!("NV12 data too small: expected {}, got {}", y_size + uv_size, nv12_data.len()).into());
            }

            // Create planar YUV420 data
            let mut yuv420_data = Vec::with_capacity(y_size + uv_size);

            // Copy Y plane as-is
            yuv420_data.extend_from_slice(&nv12_data[0..y_size]);

            // Convert interleaved UV to separate U and V planes
            let uv_plane = &nv12_data[y_size..y_size + uv_size];

            // Extract U components (even indices in UV plane)
            for i in (0..uv_size).step_by(2) {
                yuv420_data.push(uv_plane[i]);
            }

            // Extract V components (odd indices in UV plane)
            for i in (1..uv_size).step_by(2) {
                yuv420_data.push(uv_plane[i]);
            }

            let yuv_image = YuvImage {
                pixels: yuv420_data.as_slice(),
                width,
                align: 1,
                height,
                subsamp: Subsamp::Sub2x2, // YUV420 (converted from NV12)
            };
            compressor.compress_yuv_to_vec(yuv_image)?
        }
        None => return Err("No image data in ImageRawAny".into()),
    };

    Ok(jpeg_data)
}
