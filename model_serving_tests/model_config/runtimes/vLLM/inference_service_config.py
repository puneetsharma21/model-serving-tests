from model_serving_tests.utils.inference_service_builder import InferenceServiceBuilder

def create_vllm_raw_inference_service(
    model_uri: str,
    namespace: str,
    runtime: str = "vllm-runtime",
    protocol: str = "https",
    runtime_image: str = None,
    annotations: dict = None,
):
    """Creates a raw deployment ISVC using vLLM runtime"""
    builder = (
        InferenceServiceBuilder()
        .with_name("vllm-raw-auth-test")
        .with_namespace(namespace)
        .with_predictor_host(runtime)
        .with_protocol(protocol)
        .with_storage_uri(model_uri)
        .with_runtime(runtime)
        .with_runtime_image(runtime_image)
        .with_raw_deployment()
    )

    if annotations:
        builder = builder.with_annotations(annotations)

    return builder.build()
