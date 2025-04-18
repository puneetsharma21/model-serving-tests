import shlex

from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger
from contextlib import contextmanager
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client
from utilities.constants import Annotations
from utilities.constants import KServeDeploymentType

from utilities.constants import Timeout
from timeout_sampler import TimeoutExpiredError, TimeoutSampler
import utilities.general
from typing import Any, Generator, Optional, Set
LOGGER = get_logger(name=__name__)


def get_pods_by_isvc_label(client: DynamicClient, isvc: InferenceService, runtime_name: str | None = None) -> list[Pod]:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService):InferenceService object.
        runtime_name (str): ServingRuntime name

    Returns:
        list[Pod]: A list of all matching pods

    Raises:
        ResourceNotFoundError: if no pods are found.
    """
    label_selector = utilities.general.create_isvc_label_selector_str(
        isvc=isvc, resource_type="pod", runtime_name=runtime_name
    )

    if pods := [
        pod
        for pod in Pod.get(
            dyn_client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
        )
    ]:
        return pods

    raise ResourceNotFoundError(f"{isvc.name} has no pods")


def get_openshift_token() -> str:
    """
    Get the OpenShift token.

    Returns:
        str: The OpenShift token.

    """
    return run_command(command=shlex.split("oc whoami -t"))[1].strip()



def get_inference_serving_runtime(isvc: InferenceService) -> ServingRuntime:
    """
    Get the serving runtime for the inference service.

    Args:
        isvc (InferenceService):InferenceService object.

    Returns:
        ServingRuntime: ServingRuntime object.

    Raises:
        ResourceNotFoundError: if the serving runtime does not exist.

    """
    runtime = ServingRuntime(
        client=isvc.client,
        namespace=isvc.namespace,
        name=isvc.instance.spec.predictor.model.runtime,
    )

    if runtime.exists:
        return runtime

    raise ResourceNotFoundError(f"{isvc.name} runtime {runtime.name} does not exist")



def create_inference_token(model_service_account: ServiceAccount) -> str:
    """
    Generates an inference token for the given model service account.

    Args:
        model_service_account (ServiceAccount): An object containing the namespace and name
                               of the service account.

    Returns:
        str: The generated inference token.
    """
    return run_command(
        shlex.split(f"oc create token -n {model_service_account.namespace} {model_service_account.name}")
    )[1].strip()



def check_pod_status_in_time(pod: Pod, status: Set[str], duration: int = Timeout.TIMEOUT_2MIN, wait: int = 1) -> None:
    """
    Checks if a pod status is maintained for a given duration. If not, an AssertionError is raised.

    Args:
        pod (Pod): The pod to check
        status (Set[Pod.Status]): Expected pod status(es)
        duration (int): Maximum time to check for in seconds
        wait (int): Time to wait between checks in seconds

    Raises:
        AssertionError: If pod status is not in the expected set
    """
    LOGGER.info(f"Checking pod status for {pod.name} to be {status} for {duration} seconds")

    sampler = TimeoutSampler(
        wait_timeout=duration,
        sleep=wait,
        func=lambda: pod.instance,
    )

    try:
        for sample in sampler:
            if sample:
                if sample.status.phase not in status:
                    raise AssertionError(f"Pod status is not the expected: {pod.status}")

    except TimeoutExpiredError:
        LOGGER.info(f"Pod status is {pod.status} as expected")

from openshift.dynamic import DynamicClient
from kubernetes import config

def get_route_token(route_url: str, verify_tls: bool = False) -> str:
    """
    Example: returns the current OpenShift user token for authentication.
    """
    k8s_client = config.new_client_from_config()
    dyn_client = DynamicClient(k8s_client)
    return k8s_client.configuration.api_key.get('authorization', '').replace('Bearer ', '')
