use make87::encodings::{Encoder, ProtobufEncoder};
use make87_messages::core::Header;
use make87_messages::detection::r#box::Boxes2DAxisAligned;
use make87_messages::google::protobuf::Timestamp;
use make87_messages::image::compressed::ImageJpeg;
use make87_messages::image::uncompressed::{
    image_raw_any, ImageRawAny, ImageNv12, ImageRgb888, ImageRgba8888, ImageYuv420,
};
use make87_messages::text::PlainText;
// Removed ndarray import - no longer needed with rerun's native pixel formats
use regex::Regex;
use std::collections::HashMap;
use std::error::Error;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Global state for frame drop detection
static LAST_CAMERA_TIMESTAMP: Mutex<Option<f64>> = Mutex::new(None);
static TOTAL_FRAMES_RECEIVED: AtomicU32 = AtomicU32::new(0);
static TOTAL_FRAMES_DROPPED: AtomicU32 = AtomicU32::new(0);

// Memory tracking
static TOTAL_BYTES_PROCESSED: AtomicUsize = AtomicUsize::new(0);

fn get_process_memory_mb() -> Result<f64, Box<dyn Error>> {
    use std::fs;
    let status = fs::read_to_string("/proc/self/status")?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<f64>() {
                    return Ok(kb / 1024.0); // Convert KB to MB
                }
            }
        }
    }
    Err("Could not parse memory usage".into())
}

fn log_memory_usage(frame_count: u32, data_size: usize) {
    let total_bytes = TOTAL_BYTES_PROCESSED.fetch_add(data_size, Ordering::Relaxed) + data_size;
    if frame_count % 20 == 0 {
        match get_process_memory_mb() {
            Ok(memory_mb) => {
                let total_gb = total_bytes as f64 / 1_073_741_824.0; // Convert to GB
                println!("🧠 Memory: {:.1}MB RSS | {:.2}GB processed | Avg {:.1}MB/frame", 
                    memory_mb, total_gb, total_gb * 1024.0 / frame_count as f64);
            }
            Err(_) => {
                println!("🧠 Memory tracking unavailable (non-Linux system)");
            }
        }
    }
}

fn timestamp_to_secs_f64(ts: &Timestamp) -> f64 {
    ts.seconds as f64 + (ts.nanos as f64 / 1_000_000_000.0)
}

fn format_timestamp_human_readable(ts: &Timestamp) -> String {
    use std::time::{Duration, UNIX_EPOCH};

    let duration = Duration::from_secs(ts.seconds as u64) + Duration::from_nanos(ts.nanos as u64);
    let system_time = UNIX_EPOCH + duration;

    match system_time.duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let millis = d.as_millis();
            let seconds = d.as_secs();
            let ms_part = millis % 1000;

            // Format as HH:MM:SS.mmm
            let hours = (seconds / 3600) % 24;
            let minutes = (seconds / 60) % 60;
            let secs = seconds % 60;

            format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, ms_part)
        }
        Err(_) => "Invalid timestamp".to_string(),
    }
}

fn format_wallclock_time() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let millis = d.as_millis();
            let seconds = d.as_secs();
            let ms_part = millis % 1000;

            // Format as HH:MM:SS.mmm
            let hours = (seconds / 3600) % 24;
            let minutes = (seconds / 60) % 60;
            let secs = seconds % 60;

            format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, ms_part)
        }
        Err(_) => "Invalid wallclock time".to_string(),
    }
}

fn get_current_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn detect_frame_drops(current_timestamp: f64) {
    let frames_received = TOTAL_FRAMES_RECEIVED.fetch_add(1, Ordering::Relaxed) + 1;
    
    if let Ok(mut last_ts) = LAST_CAMERA_TIMESTAMP.lock() {
        if let Some(last) = *last_ts {
            let time_gap = current_timestamp - last;
            // Assuming 20fps = 50ms intervals, detect drops if gap > 75ms
            if time_gap > 0.075 {
                let expected_frames = (time_gap / 0.05).round() as u32;
                let dropped_frames = expected_frames - 1; // -1 because we got the current frame
                if dropped_frames > 0 {
                    let total_dropped = TOTAL_FRAMES_DROPPED.fetch_add(dropped_frames, Ordering::Relaxed) + dropped_frames;
                    println!("🚨 Frame drop detected! Gap: {:.3}s, Estimated drops: {}, Total dropped: {}, Total received: {}", 
                        time_gap, dropped_frames, total_dropped, frames_received);
                }
            }
        }
        *last_ts = Some(current_timestamp);
    }
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

        // Print timestamp from header to check camera timing
        if let Some(header) = &message_decoded.header {
            if let Some(timestamp) = &header.timestamp {
                println!(
                    "🕐 Camera timestamp: {} | Wallclock: {}",
                    format_timestamp_human_readable(timestamp),
                    format_wallclock_time()
                );
            } else {
                println!("🕐 No timestamp in header | Wallclock: {}", format_wallclock_time());
            }
        } else {
            println!("🕐 No header in message | Wallclock: {}", format_wallclock_time());
        }

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

        // Print timestamp from header to check camera timing
        if let Some(header) = &message_decoded.header {
            if let Some(timestamp) = &header.timestamp {
                println!(
                    "🕐 Camera timestamp: {} | Wallclock: {}",
                    format_timestamp_human_readable(timestamp),
                    format_wallclock_time()
                );
            } else {
                println!("🕐 No timestamp in header | Wallclock: {}", format_wallclock_time());
            }
        } else {
            println!("🕐 No header in message | Wallclock: {}", format_wallclock_time());
        }

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
        let total_start = Instant::now();
        println!("  🚀 Processing YUV420 frame with native rerun format ({}x{})", self.data.width, self.data.height);
        
        let width = self.data.width;
        let height = self.data.height;

        // Use rerun's native YUV420 pixel format - avoid cloning data!
        let data_size = self.data.data.len();
        let image_start = Instant::now();
        let image = rerun::Image::from_pixel_format(
            [width, height],
            rerun::PixelFormat::Y_U_V12_LimitedRange,
            &self.data.data[..], // Use slice instead of clone to avoid memory copy
        );
        let image_duration = image_start.elapsed();
        println!("  🖼️  Native YUV420 rerun::Image creation: {:.3}ms", image_duration.as_secs_f64() * 1000.0);

        let log_start = Instant::now();
        rec.log(entity_path, &image)?;
        let log_duration = log_start.elapsed();
        println!("  📤 YUV420 rerun log call: {:.3}ms", log_duration.as_secs_f64() * 1000.0);
        
        // Periodically flush rerun buffers (async, non-blocking) to prevent memory accumulation
        static FLUSH_COUNTER: AtomicU32 = AtomicU32::new(0);
        let flush_count = FLUSH_COUNTER.fetch_add(1, Ordering::Relaxed);
        if flush_count % 10 == 0 {  // Flush every 10 frames instead of every frame
            rec.flush_async();
            println!("  🚽 Async flush triggered (non-blocking)");
        }
        
        let total_duration = total_start.elapsed();
        println!("  ⏱️  Total native YUV420 processing: {:.3}ms", total_duration.as_secs_f64() * 1000.0);

        // Debug: Monitor RecordingStream state every 20 frames
        static RGB_FRAME_COUNT: AtomicU32 = AtomicU32::new(0);
        let frame_num = RGB_FRAME_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        println!("🔍 YUV420 Frame {} processed", frame_num);
        
        // Log memory usage
        log_memory_usage(frame_num, data_size);
        
        if frame_num % 20 == 0 {
            let frames_received = TOTAL_FRAMES_RECEIVED.load(Ordering::Relaxed);
            let frames_dropped = TOTAL_FRAMES_DROPPED.load(Ordering::Relaxed);
            let drop_rate = if frames_received > 0 { (frames_dropped as f32 / (frames_received + frames_dropped) as f32) * 100.0 } else { 0.0 };
            println!(
                "🔍 YUV420 - Frame {}: RecordingStream ref_count={}, enabled={}, Drop rate: {:.1}%",
                frame_num,
                rec.ref_count(),
                rec.is_enabled(),
                drop_rate
            );
            println!("📊 YUV420 - Store info: {:?}", rec.store_info());
        }

        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "YUV420"
    }
}

// Note: Removed expensive YUV420 to RGB conversion function
// Now using rerun's native pixel format support for zero-copy performance!

struct Rgb888Handler<'a> {
    data: &'a ImageRgb888,
}

impl<'a> ImageFormatHandler for Rgb888Handler<'a> {
    fn log_to_rerun(
        &self,
        entity_path: String,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let total_start = Instant::now();
        println!("  🚀 Processing RGB888 frame with native rerun format ({}x{})", self.data.width, self.data.height);
        
        let width = self.data.width;
        let height = self.data.height;

        // Use rerun's native RGB888 format - avoid cloning!
        let image_start = Instant::now();
        let image = rerun::Image::new(
            &self.data.data[..], // Use slice instead of clone
            rerun::ImageFormat::rgb8([width, height])
        );
        let image_duration = image_start.elapsed();
        println!("  🖼️  Native RGB888 rerun::Image creation: {:.3}ms", image_duration.as_secs_f64() * 1000.0);

        let log_start = Instant::now();
        rec.log(entity_path, &image)?;
        let log_duration = log_start.elapsed();
        println!("  📤 RGB888 rerun log call: {:.3}ms", log_duration.as_secs_f64() * 1000.0);
        
        let total_duration = total_start.elapsed();
        println!("  ⏱️  Total native RGB888 processing: {:.3}ms", total_duration.as_secs_f64() * 1000.0);

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
        let total_start = Instant::now();
        println!("  🚀 Processing RGBA8888 frame with native rerun format ({}x{})", self.data.width, self.data.height);
        
        let width = self.data.width;
        let height = self.data.height;

        // Use rerun's native RGBA8888 format - avoid cloning!
        let image_start = Instant::now();
        let image = rerun::Image::new(
            &self.data.data[..], // Use slice instead of clone
            rerun::ImageFormat::rgba8([width, height])
        );
        let image_duration = image_start.elapsed();
        println!("  🖼️  Native RGBA8888 rerun::Image creation: {:.3}ms", image_duration.as_secs_f64() * 1000.0);

        let log_start = Instant::now();
        rec.log(entity_path, &image)?;
        let log_duration = log_start.elapsed();
        println!("  📤 RGBA8888 rerun log call: {:.3}ms", log_duration.as_secs_f64() * 1000.0);
        
        let total_duration = total_start.elapsed();
        println!("  ⏱️  Total native RGBA8888 processing: {:.3}ms", total_duration.as_secs_f64() * 1000.0);

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

struct Nv12Handler<'a> {
    data: &'a ImageNv12,
}

impl<'a> ImageFormatHandler for Nv12Handler<'a> {
    fn log_to_rerun(
        &self,
        entity_path: String,
        rec: &rerun::RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        let total_start = Instant::now();
        println!("  🚀 Processing NV12 frame with native rerun format ({}x{})", self.data.width, self.data.height);
        
        let width = self.data.width;
        let height = self.data.height;

        // Use rerun's native NV12 pixel format - avoid cloning!
        let image_start = Instant::now();
        let image = rerun::Image::from_pixel_format(
            [width, height],
            rerun::PixelFormat::NV12,
            &self.data.data[..], // Use slice instead of clone
        );
        let image_duration = image_start.elapsed();
        println!("  🖼️  Native NV12 rerun::Image creation: {:.3}ms", image_duration.as_secs_f64() * 1000.0);

        let log_start = Instant::now();
        rec.log(entity_path, &image)?;
        let log_duration = log_start.elapsed();
        println!("  📤 NV12 rerun log call: {:.3}ms", log_duration.as_secs_f64() * 1000.0);
        
        let total_duration = total_start.elapsed();
        println!("  ⏱️  Total native NV12 processing: {:.3}ms", total_duration.as_secs_f64() * 1000.0);

        // Debug: Monitor RecordingStream state every 20 frames
        static NV12_FRAME_COUNT: AtomicU32 = AtomicU32::new(0);
        let frame_num = NV12_FRAME_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if frame_num % 20 == 0 {
            let frames_received = TOTAL_FRAMES_RECEIVED.load(Ordering::Relaxed);
            let frames_dropped = TOTAL_FRAMES_DROPPED.load(Ordering::Relaxed);
            let drop_rate = if frames_received > 0 { (frames_dropped as f32 / (frames_received + frames_dropped) as f32) * 100.0 } else { 0.0 };
            println!(
                "🔍 NV12 - Frame {}: RecordingStream ref_count={}, enabled={}, Drop rate: {:.1}%",
                frame_num,
                rec.ref_count(),
                rec.is_enabled(),
                drop_rate
            );
            println!("📊 NV12 - Store info: {:?}", rec.store_info());
        }

        Ok(())
    }

    fn get_format_name(&self) -> &'static str {
        "NV12"
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
        let decode_start = Instant::now();
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let decode_duration = decode_start.elapsed();
        println!("  📦 Protobuf decode: {:.3}ms", decode_duration.as_secs_f64() * 1000.0);

        // Print timestamp from header to check camera timing and detect frame drops
        if let Some(header) = &message_decoded.header {
            if let Some(timestamp) = &header.timestamp {
                let timestamp_secs = timestamp_to_secs_f64(timestamp);
                detect_frame_drops(timestamp_secs);
                println!(
                    "🕐 Camera timestamp: {} | Wallclock: {}",
                    format_timestamp_human_readable(timestamp),
                    format_wallclock_time()
                );
            } else {
                println!("🕐 No timestamp in header | Wallclock: {}", format_wallclock_time());
            }
        } else {
            println!("🕐 No header in message | Wallclock: {}", format_wallclock_time());
        }

        let (entity_path, _header_time) = process_header_and_set_time(&message_decoded.header, rec);

        // Handle the one-of field properly
        let dispatch_start = Instant::now();
        match &message_decoded.image {
            Some(image_raw_any::Image::Rgb888(rgb888)) => {
                println!("  📋 Dispatching RGB888 format ({}x{})", rgb888.width, rgb888.height);
                let handler = Rgb888Handler { data: rgb888 };
                handle_image_format(&handler, entity_path, rec)?;
            }
            Some(image_raw_any::Image::Rgba8888(rgba8888)) => {
                println!("  📋 Dispatching RGBA8888 format ({}x{})", rgba8888.width, rgba8888.height);
                let handler = Rgba8888Handler { data: rgba8888 };
                handle_image_format(&handler, entity_path, rec)?;
            }
            Some(image_raw_any::Image::Yuv420(yuv420)) => {
                println!("  📋 Dispatching YUV420 format ({}x{})", yuv420.width, yuv420.height);
                let handler = Yuv420Handler { data: yuv420 };
                handle_image_format(&handler, entity_path, rec)?;
            }
            Some(image_raw_any::Image::Yuv422(_yuv422)) => {
                println!("  📋 YUV422 format not implemented");
                log::warn!("YUV422 format not yet implemented");
            }
            Some(image_raw_any::Image::Yuv444(_yuv444)) => {
                println!("  📋 YUV444 format not implemented");
                log::warn!("YUV444 format not yet implemented");
            }
            Some(image_raw_any::Image::Nv12(nv12)) => {
                println!("  📋 Dispatching NV12 format ({}x{})", nv12.width, nv12.height);
                let handler = Nv12Handler { data: nv12 };
                handle_image_format(&handler, entity_path, rec)?;
            }
            None => {
                return Err("No image format found in ImageRawAny message".into());
            }
        }
        let dispatch_duration = dispatch_start.elapsed();
        println!("  ⚡ Image dispatch + processing: {:.3}ms", dispatch_duration.as_secs_f64() * 1000.0);

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
