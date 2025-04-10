import pytest
from ocp_resources.resource import ResourceEditor
from model_serving_tests.endpoint_utility.openai_utility import OpenAIClient
from utilities.constants import Annotations
from utilities.infra import check_pod_status_in_time, get_pods_by_isvc_label

pytestmark = pytest.mark.usefixtures("valid_aws_config")


COMPLETION_QUERY = {
    "model": "vllm",
    "messages": [
        {"role": "user", "content": "What is AI?"}
    ],
    "temperature": 0.7
}


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "kserve-vllm-token-auth"},
            {"model-dir": "s3://your-bucket/path-to-vllm-model"},  # Replace with real path
        )
    ],
    indirect=True,
)
class TestKserveTokenAuthenticationVLLMRaw:
    """Tests for vLLM raw ISVC with token-based authentication enabled and disabled"""
    def test_model_authentication_using_rest_raw(self, http_s3_vllm_raw_inference_service, http_raw_inference_token):
        """Query secured vLLM raw ISVC with token"""
        response = OpenAIClient(
            endpoint=http_s3_vllm_raw_inference_service.status.url,
            token=http_raw_inference_token
        ).chat(COMPLETION_QUERY)
        assert "choices" in response

    def test_disabled_raw_model_authentication(self, patched_remove_vllm_authentication_isvc):
        """Query after auth is disabled - should work without token"""
        response = OpenAIClient(
            endpoint=patched_remove_vllm_authentication_isvc.status.url
        ).chat(COMPLETION_QUERY)
        assert "choices" in response

    def test_re_enabled_raw_model_authentication(self, http_s3_vllm_raw_inference_service, http_raw_inference_token):
        """Query after re-enabling auth"""
        response = OpenAIClient(
            endpoint=http_s3_vllm_raw_inference_service.status.url,
            token=http_raw_inference_token
        ).chat(COMPLETION_QUERY)
        assert "choices" in response

    def test_disable_enable_auth_no_pod_rollout(self, http_s3_vllm_raw_inference_service):
        """Ensure auth toggle doesn't trigger pod restart"""
        pod = get_pods_by_isvc_label(
            client=http_s3_vllm_raw_inference_service.client,
            isvc=http_s3_vllm_raw_inference_service,
        )[0]

        # Disable auth
        ResourceEditor(
            patches={
                http_s3_vllm_raw_inference_service: {
                    "metadata": {"annotations": {Annotations.KserveAuth.SECURITY: "false"}}
                }
            }
        ).update()
        check_pod_status_in_time(pod=pod, status={pod.Status.RUNNING})

        # Enable auth
        ResourceEditor(
            patches={
                http_s3_vllm_raw_inference_service: {
                    "metadata": {"annotations": {Annotations.KserveAuth.SECURITY: "true"}}
                }
            }
        ).update()
        check_pod_status_in_time(pod=pod, status={pod.Status.RUNNING})
