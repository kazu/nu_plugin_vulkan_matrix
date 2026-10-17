fn main() {
    match nu_plugin_vulkan_matrix::plugin::MatrixPlugin::new() {
        Ok(plugin) => nu_plugin::serve_plugin(&plugin, nu_plugin::MsgPackSerializer),
        Err(error) => {
            eprintln!("{error:?}");
            std::process::exit(1);
        }
    }
}
