use make87::encodings::{Encoder, ProtobufEncoder};
use make87::interfaces::zenoh::ZenohInterface;
use make87::interfaces::zenoh::ZenohInterfaceError::SubTopicNotFound;
use make87_messages::text::PlainText;
use std::error::Error;
use make87::interfaces::rerun::RerunGRpcInterface;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let application_config = make87::config::load_config_from_default_env()?;


    let message_encoder = ProtobufEncoder::<PlainText>::new();
    let zenoh_interface = ZenohInterface::new(application_config.clone(), "zenoh");
    let session = zenoh_interface.get_session().await?;
    let subscriber_config = zenoh_interface
        .get_subscriber_config("any_message")
        .ok_or_else(|| SubTopicNotFound("any_message".to_string()))?;

    let rerun_grpc_interface = RerunGRpcInterface::new(application_config.clone(),"rerun-grpc");
    let rec = rerun_grpc_interface.get_client_recording_stream("rerun-grpc-client")?;


    let subscriber = session.declare_subscriber(&subscriber_config.config.topic_key).await?;
    while let Ok(sample) = subscriber.recv_async().await {
        log::info!("Received sample. Topic: {}", sample.key_expr());
        let message_decoded = message_encoder.decode(&sample.payload().to_bytes());
        match message_decoded {
            Ok(msg) => {
                log::info!("Received message: {:?}", msg);
                let header = msg.header.unwrap();
                rec.log(header.entity_path, &rerun::TextDocument::new(format!("{:?}: {}",header.timestamp.unwrap(),msg.body)))
                    .unwrap();
            }
            Err(e) => log::error!("Decode error: {}", e),
        }
    }

    Ok(())
}
