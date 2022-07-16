import io
import numpy as np
import json

from typing import List
from PIL import Image

from triton_python_backend_utils import Tensor, InferenceResponse, \
    get_input_tensor_by_name, InferenceRequest, get_output_config_by_name, \
    triton_string_to_numpy


def image_preprocess(image, target_size, stride=32):

    ih, iw = target_size
    h, w, _ = np.array(image).shape

    scale = min(iw/w, ih/h)
    nw, nh = int(round(scale * w)), int(round(scale * h))
    dw, dh = iw - nw, ih - nh  # wh padding
    # wh padding to the closest value dividable by stride
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    # scaled with padding that is dividable by stride
    scaled_w, scaled_h = nw + dw, nh + dh
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))

    image_resized = image.resize((nw, nh))
    image_resized = np.array(image_resized)  # h x w x c
    image_resized = np.transpose(
        image_resized, axes=(2, 0, 1))  # convert to c x h x w

    image_padded = np.full(shape=[3, scaled_h, scaled_w], fill_value=114.0)
    image_padded[:, top:nh+top, left:nw+left] = image_resized
    image_padded = np.ascontiguousarray(image_padded)

    image_padded = image_padded / 255.

    return image_padded, scaled_h, scaled_w, h, w


class TritonPythonModel(object):
    def __init__(self):
        self.tf = None
        self.output_names = {
            'img': 'output',
            'orig_img_hw': 'output_orig_img_hw',
            'scaled_img_hw': 'output_scaled_img_hw',
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        output_configs = {k: get_output_config_by_name(
            model_config, name) for k, name in self.output_names.items()}
        for k, cfg in output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

        self.output_dtypes = {k: triton_string_to_numpy(
            cfg['data_type']) for k, cfg in output_configs.items()}

    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        input_name = 'input'
        output_name = 'output'

        responses = []
        for request in inference_requests:
            # This model only process one input per request. We use
            # get_input_tensor_by_name instead of checking
            # len(request.inputs()) to allow for multiple inputs but
            # only process the one we want. Same rationale for the outputs
            batch_in_tensor: Tensor = get_input_tensor_by_name(
                request, input_name)
            if batch_in_tensor is None:
                raise ValueError(
                    f'Input tensor {input_name} not found '
                    f'in request {request.request_id()}')

            if output_name not in request.requested_output_names():
                raise ValueError(
                    f'The output with name {output_name} is '
                    f'not in the requested outputs '
                    f'{request.requested_output_names()}')

            batch_in = batch_in_tensor.as_numpy()  # shape (batch_size, 1)

            if batch_in.dtype.type is not np.object_:
                raise ValueError(
                    f'Input datatype must be np.object_, '
                    f'got {batch_in.dtype.type}')

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}
            for img in batch_in:  # img is shape (1,)
                pil_img = Image.open(io.BytesIO(img.astype(bytes)))  # RGB
                image_data, scaled_h, scaled_w, orig_h, orig_w = image_preprocess(
                    pil_img, [640, 640])
                image_data = image_data.astype(np.float32)
                batch_out['img'].append(image_data)
                batch_out['orig_img_hw'].append([orig_h, orig_w])
                batch_out['scaled_img_hw'].append([scaled_h, scaled_w])

            # Format outputs to build an InferenceResponse
            # Assumes there is only one output
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor
            # to handle errors
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses
