use make87::encodings::{Encoder, ProtobufEncoder};
use make87_messages::core::Header;
use make87_messages::google::protobuf::Timestamp;
use make87_messages::image::compressed::ImageJpeg;
use make87_messages::image::uncompressed::{image_raw_any, ImageRawAny, ImageRgb888, ImageRgba8888, ImageYuv420};
use make87_messages::text::PlainText;
use regex::Regex;
use std::collections::HashMap;
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};
use ndarray::ShapeBuilder as _;

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
        let width = self.data.width as usize;
        let height = self.data.height as usize;

        // Convert YUV420 to RGB using yuvutils-rs
        let rgb_data = yuv420_to_rgb_with_yuvutils(&self.data.data, width, height)?;

        log::debug!("Creating ndarray for {}x{} image with {} RGB bytes", width, height, rgb_data.len());

        // Create ndarray from converted RGB bytes with explicit row-major (C-order) layout
        // The RGB data is in row-major format: RGBRGBRGB... for each row
        let image_array = ndarray::Array::from_shape_vec(
            (height, width, 3).strides((width * 3, 3, 1)), // Explicit strides for HWC layout
            rgb_data
        ).map_err(|e| format!("Failed to create ndarray: {:?}", e))?;

        let image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGB, image_array)?;
        log::debug!("Successfully created rerun::Image, logging to entity: {}", entity_path);
        rec.log(entity_path, &image)?;
        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "YUV420"
    }
}

// YUV420 to RGB conversion using yuvutils-rs v0.8
fn yuv420_to_rgb_with_yuvutils(yuv_data: &[u8], width: usize, height: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let y_size = width * height;
    let uv_size = y_size / 4;  // Each U and V plane is width/2 * height/2
    let expected_total_size = y_size + 2 * uv_size;

    log::debug!("YUV420 conversion: {}x{}, Y_size={}, UV_size={}, expected_total={}, actual_data_len={}",
                width, height, y_size, uv_size, expected_total_size, yuv_data.len());

    if yuv_data.len() < expected_total_size {
        return Err(format!("Insufficient YUV420 data: expected {}, got {}", expected_total_size, yuv_data.len()).into());
    }

    let y_plane = &yuv_data[0..y_size];
    let u_plane = &yuv_data[y_size..y_size + uv_size];
    let v_plane = &yuv_data[y_size + uv_size..y_size + 2 * uv_size];

    // Log some sample values for debugging
    if !y_plane.is_empty() && !u_plane.is_empty() && !v_plane.is_empty() {
        log::debug!("Sample YUV values: Y[0]={}, U[0]={}, V[0]={}", y_plane[0], u_plane[0], v_plane[0]);
        // Log a few more samples for better debugging
        let mid_y = y_size / 2;
        let mid_uv = uv_size / 2;
        log::debug!("Mid YUV values: Y[{}]={}, U[{}]={}, V[{}]={}",
                   mid_y, y_plane[mid_y], mid_uv, u_plane[mid_uv], mid_uv, v_plane[mid_uv]);
    }

    // Create YuvPlanarImage structure with correct strides for packed data
    // Since your encoder packs the data without padding, stride equals actual width
    let yuv_planar = yuvutils_rs::YuvPlanarImage {
        y_plane,
        y_stride: width as u32,  // Y plane stride = width (no padding)
        u_plane,
        u_stride: (width / 2) as u32,  // U plane stride = width/2 (no padding)
        v_plane,
        v_stride: (width / 2) as u32,  // V plane stride = width/2 (no padding)
        width: width as u32,
        height: height as u32,
    };

    let mut rgb_data = vec![0u8; width * height * 3];

    // Try multiple combinations of color space parameters
    let conversion_attempts = [
        // Most common modern settings
        (yuvutils_rs::YuvRange::Limited, yuvutils_rs::YuvStandardMatrix::Bt709),
        // Traditional broadcast settings
        (yuvutils_rs::YuvRange::Limited, yuvutils_rs::YuvStandardMatrix::Bt601),
        // Full range modern
        (yuvutils_rs::YuvRange::Full, yuvutils_rs::YuvStandardMatrix::Bt709),
        // Full range traditional
        (yuvutils_rs::YuvRange::Full, yuvutils_rs::YuvStandardMatrix::Bt601),
    ];

    for (i, (range, matrix)) in conversion_attempts.iter().enumerate() {
        let result = yuvutils_rs::yuv420_to_rgb(
            &yuv_planar,
            &mut rgb_data,
            width as u32 * 3, // RGB stride
            *range,
            *matrix,
        );

        match result {
            Ok(_) => {
                log::debug!("YUV420 to RGB conversion successful with {:?} range, {:?} matrix (attempt {})",
                           range, matrix, i + 1);

                // Log some sample RGB values for debugging
                if rgb_data.len() >= 6 {
                    log::debug!("Sample RGB values: R[0]={}, G[0]={}, B[0]={}",
                               rgb_data[0], rgb_data[1], rgb_data[2]);
                    let mid = (rgb_data.len() / 2) & !2; // Ensure even index for RGB alignment
                    log::debug!("Mid RGB values: R[{}]={}, G[{}]={}, B[{}]={}",
                               mid/3, rgb_data[mid], mid/3, rgb_data[mid+1], mid/3, rgb_data[mid+2]);
                }

                return Ok(rgb_data);
            }
            Err(e) => {
                log::debug!("YUV420 conversion attempt {} failed with {:?} range, {:?} matrix: {:?}",
                           i + 1, range, matrix, e);
            }
        }
    }

    Err("YUV conversion failed with all parameter combinations".into())
}

struct Rgb888Handler<'a> {
    data: &'a ImageRgb888,
}

impl<'a> ImageFormatHandler for Rgb888Handler<'a> {
    fn log_to_rerun(&self, entity_path: String, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        let width = self.data.width as usize;
        let height = self.data.height as usize;

        // Create ndarray from raw RGB bytes
        let image_array = ndarray::Array::from_shape_vec(
            (height, width, 3).f(),
            self.data.data.clone()
        )?;

        let image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGB, image_array)?;
        rec.log(entity_path, &image)?;
        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "RGB888"
    }
}

struct Rgba8888Handler<'a> {
    data: &'a ImageRgba8888,
}

impl<'a> ImageFormatHandler for Rgba8888Handler<'a> {
    fn log_to_rerun(&self, entity_path: String, rec: &rerun::RecordingStream) -> Result<(), Box<dyn Error>> {
        let width = self.data.width as usize;
        let height = self.data.height as usize;

        // Create ndarray from raw RGB bytes
        let image_array = ndarray::Array::from_shape_vec(
            (height, width, 4).f(),
            self.data.data.clone()
        )?;

        let image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGBA, image_array)?;
        rec.log(entity_path, &image)?;
        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "RGBA8888"
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
