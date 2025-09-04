use make87::interfaces::rerun::RerunGRpcInterface;
use make87::interfaces::zenoh::{ConfiguredSubscriber, ZenohInterface};
use std::error::Error;
use std::time::Instant;

mod message_handlers;
use message_handlers::MessageTypeRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let application_config = make87::config::load_config_from_default_env()?;

    let zenoh_interface = ZenohInterface::new(application_config.clone(), "zenoh");
    let session = zenoh_interface.get_session().await?;

    let rerun_grpc_interface = RerunGRpcInterface::new(application_config.clone(), "rerun-grpc");
    let rec = rerun_grpc_interface.get_client_recording_stream("rerun-grpc-client")?;

    let configured_subscriber = zenoh_interface.get_subscriber(&session, "any_message").await?;

    match configured_subscriber {
        ConfiguredSubscriber::Fifo(sub) => {
            // Create registry and determine handler from topic_key
            let registry = MessageTypeRegistry::new();
            let handler = registry
                .create_handler_from_topic_key(sub.key_expr())
                .ok_or_else(|| format!("Unknown message type for topic: {}", sub.key_expr()))?;

            while let Ok(sample) = sub.recv_async().await {
                println!("Received sample. Topic: {}", sample.key_expr());
                
                // Time the entire message handling process
                let process_start = Instant::now();
                if let Err(e) = handler.handle_message(&sample, &rec) {
                    log::error!("Error handling message: {}", e);
                }
                let process_duration = process_start.elapsed();
                println!("⏱️  Processing took: {:.3}ms", process_duration.as_secs_f64() * 1000.0);
            }
        }
        ConfiguredSubscriber::Ring(sub) => {
            // Create registry and determine handler from topic_key
            let registry = MessageTypeRegistry::new();
            let handler = registry
                .create_handler_from_topic_key(sub.key_expr())
                .ok_or_else(|| format!("Unknown message type for topic: {}", sub.key_expr()))?;

            while let Ok(sample) = sub.recv_async().await {
                println!("Received sample. Topic: {}", sample.key_expr());
                
                // Time the entire message handling process
                let process_start = Instant::now();
                if let Err(e) = handler.handle_message(&sample, &rec) {
                    log::error!("Error handling message: {}", e);
                }
                let process_duration = process_start.elapsed();
                println!("⏱️  Processing took: {:.3}ms", process_duration.as_secs_f64() * 1000.0);
            }
        }
    }

    Ok(())
}
