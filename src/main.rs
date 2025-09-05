use make87::interfaces::rerun::RerunGRpcInterface;
use make87::interfaces::zenoh::{ConfiguredSubscriber, ZenohInterface};
use std::error::Error;
use std::sync::mpsc;
use std::time::{Duration, Instant};

mod message_handlers;
use message_handlers::MessageTypeRegistry;

/// Check if the gRPC connection is still active
fn check_grpc_connection(rec: &rerun::RecordingStream) -> bool {
    let (tx, rx) = mpsc::channel();

    rec.inspect_sink(move |sink| {
        if let Some(grpc_sink) = sink.as_any().downcast_ref::<rerun::sink::GrpcSink>() {
            let _ = tx.send(grpc_sink.status());
        }
    });

    // Check with a short timeout
    if let Ok(status) = rx.recv_timeout(Duration::from_millis(100)) {
        match status {
            rerun::sink::GrpcSinkConnectionState::Connected => true,
            rerun::sink::GrpcSinkConnectionState::Connecting => true, // Still trying
            rerun::sink::GrpcSinkConnectionState::Disconnected(_) => false,
        }
    } else {
        // Timeout or no status - assume disconnected
        false
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let application_config = make87::config::load_config_from_default_env()?;

    let zenoh_interface = ZenohInterface::new(application_config.clone(), "zenoh");
    let session = zenoh_interface.get_session().await?;

    let rerun_grpc_interface = RerunGRpcInterface::new(application_config.clone(), "rerun-grpc");
    let mut rec = rerun_grpc_interface.get_client_recording_stream("rerun-grpc-client")?;
    let mut last_connection_check = Instant::now();
    const CONNECTION_CHECK_INTERVAL: Duration = Duration::from_secs(2);

    let configured_subscriber = zenoh_interface
        .get_subscriber(&session, "any_message")
        .await?;

    match configured_subscriber {
        ConfiguredSubscriber::Fifo(sub) => {
            // Create registry and determine handler from topic_key
            let registry = MessageTypeRegistry::new();
            let handler = registry
                .create_handler_from_topic_key(sub.key_expr())
                .ok_or_else(|| format!("Unknown message type for topic: {}", sub.key_expr()))?;

            while let Ok(sample) = sub.recv_async().await {
                // Periodically check connection status
                if last_connection_check.elapsed() >= CONNECTION_CHECK_INTERVAL {
                    if !check_grpc_connection(&rec) {
                        log::warn!("gRPC connection lost, attempting to reconnect...");
                        match rerun_grpc_interface.get_client_recording_stream("rerun-grpc-client")
                        {
                            Ok(new_rec) => {
                                rec = new_rec;
                                log::info!("Successfully reconnected to gRPC server");
                            }
                            Err(e) => {
                                log::error!("Failed to reconnect to gRPC server: {}", e);
                                // Continue with old connection, might recover
                            }
                        }
                    }
                    last_connection_check = Instant::now();
                }

                if let Err(e) = handler.handle_message(&sample, &rec) {
                    log::error!("Error handling message: {}", e);
                }
            }
        }
        ConfiguredSubscriber::Ring(sub) => {
            // Create registry and determine handler from topic_key
            let registry = MessageTypeRegistry::new();
            let handler = registry
                .create_handler_from_topic_key(sub.key_expr())
                .ok_or_else(|| format!("Unknown message type for topic: {}", sub.key_expr()))?;

            while let Ok(sample) = sub.recv_async().await {
                // Periodically check connection status
                if last_connection_check.elapsed() >= CONNECTION_CHECK_INTERVAL {
                    if !check_grpc_connection(&rec) {
                        log::warn!("gRPC connection lost, attempting to reconnect...");
                        match rerun_grpc_interface.get_client_recording_stream("rerun-grpc-client")
                        {
                            Ok(new_rec) => {
                                rec = new_rec;
                                log::info!("Successfully reconnected to gRPC server");
                            }
                            Err(e) => {
                                log::error!("Failed to reconnect to gRPC server: {}", e);
                                // Continue with old connection, might recover
                            }
                        }
                    }
                    last_connection_check = Instant::now();
                }

                if let Err(e) = handler.handle_message(&sample, &rec) {
                    log::error!("Error handling message: {}", e);
                }
            }
        }
    }

    Ok(())
}
