from ocp_resources.inference_service import InferenceService

class InferenceServiceBuilder:
    def __init__(self):
        self.params = {
            "metadata": {
                "name": None,
                "namespace": None,
                "annotations": {},
            },
            "spec": {
                "predictor": {
                    "model": {
                        "storageUri": None,
                    },
                    "runtime": None,
                    "protocolVersion": None,
                }
            },
        }

    def with_name(self, name):
        self.params["metadata"]["name"] = name
        return self

    def with_namespace(self, namespace):
        self.params["metadata"]["namespace"] = namespace
        return self

    def with_annotations(self, annotations):
        self.params["metadata"]["annotations"].update(annotations)
        return self

    def with_predictor_host(self, host):
        # Optional: Add logic if needed
        return self

    def with_protocol(self, protocol):
        self.params["spec"]["predictor"]["protocolVersion"] = protocol
        return self

    def with_storage_uri(self, uri):
        self.params["spec"]["predictor"]["model"]["storageUri"] = uri
        return self

    def with_runtime(self, runtime):
        self.params["spec"]["predictor"]["runtime"] = runtime
        return self

    def with_runtime_image(self, image):
        self.params["spec"]["predictor"]["model"]["image"] = image
        return self

    def with_raw_deployment(self):
        # Add config for raw deployment if needed
        return self

    def build(self):
        return InferenceService(**self.params)

