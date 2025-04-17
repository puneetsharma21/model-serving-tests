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
try:
    from ocp_resources.project_project_openshift_io import Project
except ImportError:
    Project = None

from ocp_resources.project_request import ProjectRequest
from ocp_resources.resource import get_client
from utilities.constants import Annotations
from utilities.constants import KServeDeploymentType

from utilities.constants import Timeout
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, retry
import utilities.general
from typing import Any, Generator, Optional, Set
LOGGER = get_logger(name=__name__)


@contextmanager
def create_ns(
    name: str | None = None,
    admin_client: DynamicClient | None = None,
    unprivileged_client: DynamicClient | None = None,
    teardown: bool = True,
    delete_timeout: int = Timeout.TIMEOUT_4MIN,
    labels: dict[str, str] | None = None,
    ns_annotations: dict[str, str] | None = None,
    model_mesh_enabled: bool = False,
    add_dashboard_label: bool = False,
    pytest_request: FixtureRequest | None = None,
) -> Generator[Namespace | Project, Any, Any]:
    """
    Create namespace with admin or unprivileged client.

    For a namespace / project which contains Serverless ISVC,  there is a workaround for RHOAIENG-19969.
    Currently, when Serverless ISVC is deleted and the namespace is deleted, namespace "SomeResourcesRemain" is True.
    This is because the serverless pods are not immediately deleted resulting in prolonged namespace deletion.
    Waiting for the pod(s) to be deleted before cleanup, eliminates the issue.

    Args:
        name (str): namespace name.
            Can be overwritten by `request.param["name"]`
        admin_client (DynamicClient): admin client.
        unprivileged_client (UnprivilegedClient): unprivileged client.
        teardown (bool): should run resource teardown
        delete_timeout (int): delete timeout.
        labels (dict[str, str]): labels dict to set for namespace
        ns_annotations (dict[str, str]): annotations dict to set for namespace
            Can be overwritten by `request.param["annotations"]`
        model_mesh_enabled (bool): if True, model mesh will be enabled in namespace.
            Can be overwritten by `request.param["modelmesh-enabled"]`
        add_dashboard_label (bool): if True, dashboard label will be added to namespace
            Can be overwritten by `request.param["add-dashboard-label"]`
        pytest_request (FixtureRequest): pytest request

    Yields:
        Namespace | Project: namespace or project

    """
    if pytest_request:
        name = pytest_request.param.get("name", name)
        ns_annotations = pytest_request.param.get("annotations", ns_annotations)
        model_mesh_enabled = pytest_request.param.get("modelmesh-enabled", model_mesh_enabled)
        add_dashboard_label = pytest_request.param.get("add-dashboard-label", add_dashboard_label)

    namespace_kwargs = {
        "name": name,
        "client": admin_client,
        "teardown": teardown,
        "delete_timeout": delete_timeout,
        "label": labels or {},
    }

    if ns_annotations:
        namespace_kwargs["annotations"] = ns_annotations

    if model_mesh_enabled:
        namespace_kwargs["label"]["modelmesh-enabled"] = "true"  # type: ignore

    if add_dashboard_label:
        namespace_kwargs["label"][Labels.OpenDataHub.DASHBOARD] = "true"  # type: ignore

    if unprivileged_client:
        with ProjectRequest(name=name, client=unprivileged_client, teardown=teardown):
            if Project is not None:
                # Use Project normally
                project = Project(**namespace_kwargs)
                ...
            else:
                LOGGER.warning("Project import failed. Skipping related functionality.")

            project.wait_for_status(status=project.Status.ACTIVE, timeout=Timeout.TIMEOUT_2MIN)
            yield project

            if teardown:
                wait_for_serverless_pods_deletion(resource=project, admin_client=admin_client)

    else:
        with Namespace(**namespace_kwargs) as ns:
            ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=Timeout.TIMEOUT_2MIN)
            yield ns

            if teardown:
                wait_for_serverless_pods_deletion(resource=ns, admin_client=admin_client)


def wait_for_serverless_pods_deletion(resource: Project | Namespace, admin_client: DynamicClient | None) -> None:
    """
    Wait for serverless pods deletion.

    Args:
        resource (Project | Namespace): project or namespace
        admin_client (DynamicClient): admin client.

    Returns:
        bool: True if we should wait for namespace deletion else False

    """
    client = admin_client or get_client()
    for pod in Pod.get(dyn_client=client, namespace=resource.name):
        if (
            pod.exists
            and pod.instance.metadata.annotations.get(Annotations.KserveIo.DEPLOYMENT_MODE)
            == KServeDeploymentType.SERVERLESS
        ):
            LOGGER.info(f"Waiting for {KServeDeploymentType.SERVERLESS} pod {pod.name} to be deleted")
            pod.wait_deleted(timeout=Timeout.TIMEOUT_1MIN)


@retry(
    wait_timeout=Timeout.TIMEOUT_30SEC,
    sleep=1,
    exceptions_dict={ResourceNotFoundError: []},
)

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
