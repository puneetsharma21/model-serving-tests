import pytest
import time
import logging
from typing import Callable, Any
from kubernetes.dynamic.client import DynamicClient
from ocp_resources.resource import Resource
from model_serving_tests.endpoint_utility.openai_utility import OpenAIClient
from model_serving_tests.tests.utils import (
    create_runtime_manifest_from_template,
    create_isvc_manifest_from_template,
    create_s3_secret_manifest,
    get_predictor_pod,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "vllm-cpu-model"
DEPLOYMENT_TYPE = "RawDeployment"

CHAT_QUERY = [
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]

@pytest.mark.parametrize(
    "s3_models_storage_uri",
    [{"model-dir": "vllm-cpu-model"}],
    indirect=True
)
def test_vllm_cpu_raw_isvc_auth(
    client: DynamicClient,
    create_namespace: Callable[[str], Resource],
    create_secret_from_file: Callable[[str], Resource],
    create_service_account: Callable[[str], Resource],
    create_serving_runtime_from_file: Callable[[str, str], Resource],
    create_isvc_from_file: Callable[[str, str], Resource],
    runtime: str,
    runtime_image: str,
    accelerator_type: str,
    runtime_name: str,
    http_raw_inference_token: str,
    s3_models_storage_uri: str
) -> None:
    """
    Test vLLM CPU RawDeployment with HTTPS and auth enabled
    """
    namespace_name = MODEL_NAME.lower()

    # Create manifests
    create_runtime_manifest_from_template(DEPLOYMENT_TYPE, runtime_image, runtime_name)
    create_isvc_manifest_from_template(
        DEPLOYMENT_TYPE,
        model_name=MODEL_NAME,
        accelerator_type=accelerator_type,
        gpu_count=0,  # CPU-based
        enable_auth=True,
        protocol="https"
    )
    create_s3_secret_manifest()

    # Create resources
    namespace = create_namespace(namespace_name)
    create_secret_from_file(namespace=namespace.name)
    create_service_account(namespace=namespace.name)
    create_serving_runtime_from_file(namespace=namespace.name, path=runtime)
    isvc = create_isvc_from_file(namespace=namespace.name, model_name=MODEL_NAME)

    # Wait for predictor pod
    time.sleep(10)
    predictor_pod = get_predictor_pod(client, namespace=namespace.name, is_name=isvc.name)
    predictor_pod.wait_for_status("Running", timeout=600)
    predictor_pod.wait_for_condition("Ready", "True", timeout=600)

    # Query the endpoint
    time.sleep(10)
    endpoint = isvc.status.url
    LOGGER.info(f"Sending request to endpoint: {endpoint}")
    response = OpenAIClient(endpoint=endpoint, token=http_raw_inference_token).chat(CHAT_QUERY)

    assert "choices" in response
