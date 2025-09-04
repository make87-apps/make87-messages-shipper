use make87::interfaces::rerun::RerunGRpcInterface;
use make87::interfaces::zenoh::ZenohInterface;
use make87::interfaces::zenoh::ZenohInterfaceError::SubTopicNotFound;
use std::error::Error;

mod message_handlers;
use message_handlers::MessageTypeRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let application_config = make87::config::load_config_from_default_env()?;

    let zenoh_interface = ZenohInterface::new(application_config.clone(), "zenoh");
    let session = zenoh_interface.get_session().await?;
    let subscriber_config = zenoh_interface
        .get_subscriber_config("any_message")
        .ok_or_else(|| SubTopicNotFound("any_message".to_string()))?;

    let rerun_grpc_interface = RerunGRpcInterface::new(application_config.clone(), "rerun-grpc");
    let rec = rerun_grpc_interface.get_client_recording_stream("rerun-grpc-client")?;

    // Create registry and determine handler from topic_key
    let registry = MessageTypeRegistry::new();
    let handler = registry
        .create_handler_from_topic_key(&subscriber_config.config.topic_key)
        .ok_or_else(|| {
            format!(
                "Unknown message type for topic: {}",
                subscriber_config.config.topic_key
            )
        })?;

    let subscriber = session
        .declare_subscriber(&subscriber_config.config.topic_key)
        .await?;
    while let Ok(sample) = subscriber.recv_async().await {
        println!("Received sample. Topic: {}", sample.key_expr());

        if let Err(e) = handler.handle_message(&sample, &rec) {
            log::error!("Error handling message: {}", e);
        }
    }

    Ok(())
}
