use make87::encodings::{Encoder, ProtobufEncoder};
use make87_messages::text::PlainText;
use make87_messages::core::Header;
use std::error::Error;
use make87_messages::google::protobuf::Timestamp;
use rerun::{EncodedImage, MediaType, RecordingStream, TextDocument};
use zenoh::sample::Sample;
use std::collections::HashMap;
use make87_messages::image::compressed::ImageJpeg;
use regex::Regex;

fn timestamp_to_ns(ts: &Timestamp) -> i64 {
    ts.seconds
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.nanos as i64)
}

fn process_header_and_set_time(header: Option<Header>, rec: &RecordingStream) -> (String, i64) {
    let (entity_path, header_time) = match header {
        Some(header) => {
            let time = header.timestamp
                .map(|ts| timestamp_to_ns(&ts))
                .unwrap_or(0);
            (header.entity_path, time)
        }
        None => ("/".to_string(), 0),
    };

    rec.set_time_sequence("header_time", header_time);
    (entity_path, header_time)
}

pub trait MessageHandler {
    fn handle_message(&self, sample: &Sample, rec: &RecordingStream) -> Result<(), Box<dyn Error>>;
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
    fn handle_message(&self, sample: &Sample, rec: &RecordingStream) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(message_decoded.header, rec);

        rec.log(
            entity_path,
            &TextDocument::new(message_decoded.body)
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
    fn handle_message(&self, sample: &Sample, rec: &RecordingStream) -> Result<(), Box<dyn Error>> {
        let message_decoded = self.encoder.decode(&sample.payload().to_bytes())?;
        let (entity_path, _header_time) = process_header_and_set_time(message_decoded.header, rec);
        rec.log(
            entity_path,
            &EncodedImage::new(message_decoded.data).with_media_type(MediaType::from("image/jpeg"))
        )?;
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
        registry.register("image-compressed-ImageJPEG", || Box::new(ImageCompressedJpegHandler::new()));

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
