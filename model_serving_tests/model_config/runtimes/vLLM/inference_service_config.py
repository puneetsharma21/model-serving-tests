from model_serving_tests.utils.inference_service_builder import InferenceServiceBuilder
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import get_client

def create_vllm_raw_inference_service(
    model_uri: str,
    namespace: str,
    runtime: str = "vllm-runtime",
    protocol: str = "https",
    runtime_image: str = None,
    annotations: dict = None,
):
    """Creates a raw deployment ISVC using vLLM runtime and applies it to the cluster."""

    # Build the ISVC spec using the builder
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

    isvc_body = builder.build()

    # Ensure predictor field is present (some builders don't include it properly for raw deployments)
    if "spec" in isvc_body and "predictor" not in isvc_body["spec"]:
        isvc_body["spec"]["predictor"] = {
            "replicas": 1,
            "raw": {
                "model": {
                    "storageUri": model_uri,
                    "runtime": runtime,
                    "runtimeImage": runtime_image,
                }
            }
        }

    assert "predictor" in isvc_body["spec"], "Predictor field is missing in ISVC spec!"

    # Create the ISVC resource
    isvc = InferenceService(
        client=get_client(),
        name=isvc_body["metadata"]["name"],
        namespace=namespace,
    )
    isvc._body = isvc_body

    # Create it in the cluster and wait for readiness
    isvc.create(wait=True)

    return isvc
