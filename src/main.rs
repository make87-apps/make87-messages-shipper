use make87::encodings::ProtobufEncoder;
use make87::interfaces::zenoh::{ZenohInterface, ZenohInterfaceError};
use make87_messages::text::PlainText;
use std::error::Error;
use ZenohInterfaceError::SubTopicNotFound;
use rerun;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let application_config = make87::config::load_config_from_default_env()?;

    // let message_encoder = ProtobufEncoder::<PlainText>::new();
    let zenoh_interface = ZenohInterface::new(application_config.clone(), "zenoh");
    let session = zenoh_interface.get_session().await?;
    // let subscriber_config = zenoh_interface.get_subscriber_config("any_message").ok_or_else(|| SubTopicNotFound("any_message".to_string()))?;


    let rerun_grpc_client = application_config.interfaces.get("rerun-grpc").ok_or_else(|| "Interface 'rerun-grpc' not found.")?.clients.get("rerun-grpc-client").ok_or_else(|| "Client 'rerun-grpc-client' not found.")?;

    let rec = rerun::RecordingStreamBuilder::new("make87_messages_shipper")
        .connect_grpc_opts(format!("rerun+http://{}:{}/proxy", rerun_grpc_client.access_point.vpn_ip, rerun_grpc_client.access_point.vpn_port), rerun::default_flush_timeout())?;


    let subscriber = session
        .declare_subscriber("*/*/make87_messages-$*/**")
        .callback(move |sample| {
            log::info!("Received sample. Topic: {}", sample.key_expr());
            // let message_decoded = message_encoder.decode(&sample.payload().to_bytes());
            // match message_decoded {
            //     Ok(msg) => {
            //         log::info!("Received message: {:?}", msg);
            //         rec.log("any_message", &msg).unwrap();
            //     },
            //     Err(e) => log::error!("Decode error: {}", e),
            // }
        })
        .await?;

    make87::run_forever();

    Ok(())
}
