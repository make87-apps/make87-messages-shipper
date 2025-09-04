use make87::encodings::{Encoder, ProtobufEncoder};
use make87_messages::core::Header;
use make87_messages::detection::r#box::Boxes2DAxisAligned;
use make87_messages::google::protobuf::Timestamp;
use make87_messages::image::compressed::ImageJpeg;
use make87_messages::image::uncompressed::{
    image_raw_any, ImageRawAny, ImageRgb888, ImageRgba8888, ImageYuv420,
};
use make87_messages::text::PlainText;
use ndarray::ShapeBuilder as _;
use regex::Regex;
use std::collections::HashMap;
use std::error::Error;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

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

fn process_header_and_set_time(
    header: &Option<Header>,
    rec: &rerun::RecordingStream,
) -> (String, f64) {
    let (entity_path, header_time) = match header {
        Some(header) => {
            let time = header
                .timestamp
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
    fn handle_message(
        &self,
        sample: &zenoh::sample::Sample,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>>;
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
    fn handle_message(
        &self,
        sample: &zenoh::sample::Sample,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        rec.log(entity_path, &rerun::TextDocument::new(message_decoded.body))?;
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
    fn handle_message(
        &self,
        sample: &zenoh::sample::Sample,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);
        rec.log(
            entity_path,
            &rerun::EncodedImage::new(message_decoded.data)
                .with_media_type(rerun::MediaType::from("image/jpeg")),
        )?;
        Ok(())
    }
}

// Trait for handling different image formats
trait ImageFormatHandler {
    fn log_to_rerun(
        &self,
        entity_path: String,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>>;
    fn get_format_name(&self) -> &'static str;
}

// Individual format handlers
struct Yuv420Handler<'a> {
    data: &'a ImageYuv420,
}

impl<'a> ImageFormatHandler for Yuv420Handler<'a> {
    fn log_to_rerun(
        &self,
        entity_path: String,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let width = self.data.width as usize;
        let height = self.data.height as usize;

        // Convert YUV420 to RGB
        let rgb_data = yuv420_to_rgb_with_yuvutils(&self.data.data, width, height)?;

        // Create ndarray with explicit strides for proper HWC layout
        let image_array =
            ndarray::Array::from_shape_vec((height, width, 3).strides((width * 3, 3, 1)), rgb_data)
                .map_err(|e| format!("Failed to create ndarray: {:?}", e))?;

        let image = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGB, image_array)?;
        rec.log(entity_path, &image)?;

        // Force immediate flush to prevent buffering buildup
        rec.flush_async();

        // Debug: Monitor RecordingStream state every 20 frames
        static RGB_FRAME_COUNT: AtomicU32 = AtomicU32::new(0);
        let frame_num = RGB_FRAME_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        println!("🔍 RGB888 Frame {} processed", frame_num);
        if frame_num % 20 == 0 {
            println!(
                "🔍 RGB888 - Frame {}: RecordingStream ref_count={}, enabled={}",
                frame_num,
                rec.ref_count(),
                rec.is_enabled()
            );
            println!("📊 RGB888 - Store info: {:?}", rec.store_info());
        }

        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "YUV420"
    }
}

// YUV420 to RGB conversion using yuvutils-rs
fn yuv420_to_rgb_with_yuvutils(
    yuv_data: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let y_size = width * height;
    let uv_size = y_size / 4;
    let expected_total_size = y_size + 2 * uv_size;

    if yuv_data.len() < expected_total_size {
        return Err(format!(
            "Insufficient YUV420 data: expected {}, got {}",
            expected_total_size,
            yuv_data.len()
        )
        .into());
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
    )
    .map_err(|e| format!("YUV420 to RGB conversion failed: {:?}", e))?;

    Ok(rgb_data)
}

struct Rgb888Handler<'a> {
    data: &'a ImageRgb888,
}

impl<'a> ImageFormatHandler for Rgb888Handler<'a> {
    fn log_to_rerun(
        &self,
        entity_path: String,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let width = self.data.width as usize;
        let height = self.data.height as usize;

        // Create ndarray with explicit strides for proper HWC layout - zero copy
        let image_view = ndarray::ArrayView3::from_shape(
            (height, width, 3).strides((width * 3, 3, 1)),
            &self.data.data,
        )
        .map_err(|e| format!("Failed to create ndarray: {:?}", e))?;

        let image = rerun::Image::from_color_model_and_tensor(
            rerun::ColorModel::RGB,
            image_view.to_owned(),
        )?;
        rec.log(entity_path, &image)?;

        // Force immediate flush to prevent buffering buildup
        rec.flush_async();

        // Debug: Monitor RecordingStream state every 20 frames
        static YUV420_FRAME_COUNT: AtomicU32 = AtomicU32::new(0);
        let frame_num = YUV420_FRAME_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if frame_num % 20 == 0 {
            println!(
                "🔍 YUV420 - Frame {}: RecordingStream ref_count={}, enabled={}",
                frame_num,
                rec.ref_count(),
                rec.is_enabled()
            );
            println!("📊 YUV420 - Store info: {:?}", rec.store_info());
        }

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
    fn log_to_rerun(
        &self,
        entity_path: String,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let width = self.data.width as usize;
        let height = self.data.height as usize;

        // Create ndarray with explicit strides for proper HWCA layout - zero copy
        let image_view = ndarray::ArrayView3::from_shape(
            (height, width, 4).strides((width * 4, 4, 1)),
            &self.data.data,
        )
        .map_err(|e| format!("Failed to create ndarray: {:?}", e))?;

        let image = rerun::Image::from_color_model_and_tensor(
            rerun::ColorModel::RGBA,
            image_view.to_owned(),
        )?;
        rec.log(entity_path, &image)?;

        // Force immediate flush to prevent buffering buildup
        rec.flush_async();

        // Debug: Monitor RecordingStream state every 20 frames
        static RGBA_FRAME_COUNT: AtomicU32 = AtomicU32::new(0);
        let frame_num = RGBA_FRAME_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if frame_num % 20 == 0 {
            println!(
                "🔍 RGBA8888 - Frame {}: RecordingStream ref_count={}, enabled={}",
                frame_num,
                rec.ref_count(),
                rec.is_enabled()
            );
            println!("📊 RGBA8888 - Store info: {:?}", rec.store_info());
        }

        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "RGBA8888"
    }
}

// Helper function to handle any image format
fn handle_image_format(
    handler: &dyn ImageFormatHandler,
    entity_path: String,
    rec: &rerun::RecordingStream,
) -> Result<(), Box<dyn Error>> {
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
    fn handle_message(
        &self,
        sample: &zenoh::sample::Sample,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
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
    fn handle_message(
        &self,
        sample: &zenoh::sample::Sample,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        let handler = Yuv420Handler {
            data: &message_decoded,
        };
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
    fn handle_message(
        &self,
        sample: &zenoh::sample::Sample,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        let handler = Rgb888Handler {
            data: &message_decoded,
        };
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
    fn handle_message(
        &self,
        sample: &zenoh::sample::Sample,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        let handler = Rgba8888Handler {
            data: &message_decoded,
        };
        handle_image_format(&handler, entity_path, rec)
    }
}

pub struct Boxes2DAxisAlignedHandler {
    encoder: ProtobufEncoder<Boxes2DAxisAligned>,
}

impl Boxes2DAxisAlignedHandler {
    pub fn new() -> Self {
        Self {
            encoder: ProtobufEncoder::<Boxes2DAxisAligned>::new(),
        }
    }
}

impl MessageHandler for Boxes2DAxisAlignedHandler {
    fn handle_message(
        &self,
        sample: &zenoh::sample::Sample,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        if message_decoded.boxes.is_empty() {
            log::info!("No boxes to log in Boxes2DAxisAligned message");
            return Ok(());
        }

        // Collect all box geometries for batch logging
        let mut box_centers = Vec::new();
        let mut box_half_sizes = Vec::new();

        for (i, box_item) in message_decoded.boxes.iter().enumerate() {
            if let Some(geometry) = &box_item.geometry {
                // Convert box geometry to rerun format
                // Assuming geometry has fields like x, y, width, height
                let center_x = geometry.x + geometry.width / 2.0;
                let center_y = geometry.y + geometry.height / 2.0;

                box_centers.push([center_x, center_y]);
                box_half_sizes.push([geometry.width / 2.0, geometry.height / 2.0]);

                log::info!(
                    "Box {}: center=({}, {}), size=({}, {})",
                    i,
                    center_x,
                    center_y,
                    geometry.width,
                    geometry.height
                );
            } else {
                log::warn!("Box {} has no geometry", i);
            }
        }

        // Log all boxes in one batch call using Boxes2D
        if !box_centers.is_empty() {
            let box_count = box_centers.len();
            rec.log(
                entity_path,
                &rerun::Boxes2D::from_centers_and_half_sizes(box_centers, box_half_sizes),
            )?;
            log::info!("Logged {} boxes to rerun", box_count);
        }

        Ok(())
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
        registry.register("image-compressed-ImageJPEG", || {
            Box::new(ImageCompressedJpegHandler::new())
        });

        // Register composite and individual image format handlers
        registry.register("image-uncompressed-ImageRawAny", || {
            Box::new(ImageRawAnyHandler::new())
        });
        registry.register("image-uncompressed-ImageYUV420", || {
            Box::new(ImageYuv420Handler::new())
        });
        registry.register("image-uncompressed-ImageRGB888", || {
            Box::new(ImageRgb888Handler::new())
        });
        registry.register("image-uncompressed-ImageRGBA8888", || {
            Box::new(ImageRgba8888Handler::new())
        });

        // Register detection message handlers
        registry.register("detection-box-Boxes2DAxisAligned", || {
            Box::new(Boxes2DAxisAlignedHandler::new())
        });

        registry
    }

    fn register(&mut self, message_type: &'static str, factory: HandlerFactory) {
        self.handlers.insert(message_type, factory);
    }

    pub fn create_handler_from_topic_key(
        &self,
        topic_key: &str,
    ) -> Option<Box<dyn MessageHandler>> {
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
