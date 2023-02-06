from labels import COCOLabels
import numpy as np
import json
import sys
import time
import torch
import torchvision

from pathlib import Path
from typing import List

from triton_python_backend_utils import get_output_config_by_name, triton_string_to_numpy, get_input_config_by_name
from c_python_backend_utils import Tensor, InferenceResponse, InferenceRequest

sys.path.append(Path(__file__).parent.absolute().as_posix())


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def scale_coords(img1_hw, coords, img0_hw, ratio_pad=None):
    # Rescale coords (xyxy) from img1_hw to img0_hw shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_hw[0] / img0_hw[0],
                   img1_hw[1] / img0_hw[1])  # gain  = old / new
        pad = (img1_hw[1] - img0_hw[1] * gain) / \
            2, (img1_hw[0] - img0_hw[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_hw)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


class TritonPythonModel(object):
    def __init__(self):
        self.input_names = {
            'output_1': 'input_1',
            'output_2': 'input_2',
            'output_4': 'input_4',
            'output_4': 'input_4',
            'orig_img_hw': 'input_orig_img_hw',
            'scaled_img_hw': 'input_scaled_img_hw',
        }
        self.output_names = {
            'bboxes': 'output_bboxes',
            'labels': 'output_labels'
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        if 'input' not in model_config:
            raise ValueError('Input is not defined in the model config')

        input_configs = {k: get_input_config_by_name(
            model_config, name) for k, name in self.input_names.items()}
        for k, cfg in input_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Input {self.input_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for input {self.input_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for input {self.input_names[k]} is not defined in the model config')

        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')
        if len(model_config['output']) != 2:
            raise ValueError(
                f'Expected 2 outputs, got {len(model_config["output"])}')

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
        responses = []

        for request in inference_requests:
            batch_in = {}
            for k, name in self.input_names.items():
                tensor = get_input_tensor_by_name(request, name)
                if tensor is None:
                    raise ValueError(f'Input tensor {name} not found '
                                     f'in request {request.request_id()}')
                # shape (batch_size, ...) tensor.as_numpy()
                batch_in[k] = torch.from_numpy(tensor.as_numpy())

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}

            # list of detections on (n,6) tensor per image [xyxy, conf, cls]
            pred_list = non_max_suppression(batch_in['output_1'])

            max_num_bboxes_in_single_img = 0
            for pred, orig_img_hw, scaled_img_hw in zip(pred_list, batch_in['orig_img_hw'], batch_in['scaled_img_hw']):
                max_num_bboxes_in_single_img = max(
                    max_num_bboxes_in_single_img, len(pred))

                # Rescale bounding boxes in pred (n, 6) back to original image size
                pred[:, :4] = scale_coords(
                    scaled_img_hw, pred[:, :4], orig_img_hw).round()

                # Change from pytorch tensor to numpy array
                pred = pred.numpy()

                if self.output_names['bboxes'] in request.requested_output_names():
                    if len(pred) > 0:
                        batch_out['bboxes'].append(pred[:, :5])
                    else:
                        batch_out['bboxes'].append(np.array([]))
                if self.output_names['labels'] in request.requested_output_names():
                    if len(pred) > 0:
                        batch_out['labels'].append(
                            [COCOLabels(idx).name.lower() for idx in pred[:, 5]])
                    else:
                        batch_out['labels'].append([])

            if max_num_bboxes_in_single_img == 0:
                # When no detected object at all in all imgs in the batch
                for idx, _ in enumerate(batch_out['bboxes']):
                    batch_out['bboxes'][idx] = [[-1, -1, -1, -1, -1]]
                for idx, _ in enumerate(batch_out['labels']):
                    batch_out['labels'][idx] = ["0"]
            else:
                # The output of all imgs must have the same size for Triton to be able to output a Tensor of type self.output_dtypes
                # Non-meaningful bounding boxes have coords [-1, -1, -1, -1, -1] and label '0'
                # Loop over images in batch
                for idx, out in enumerate(batch_out['bboxes']):
                    if len(out) < max_num_bboxes_in_single_img:
                        num_to_add = max_num_bboxes_in_single_img - len(out)
                        to_add = -np.ones((num_to_add, 5))
                        if len(out) == 0:
                            batch_out['bboxes'][idx] = to_add
                        else:
                            batch_out['bboxes'][idx] = np.vstack((out, to_add))

                # Loop over images in batch
                for idx, out in enumerate(batch_out['labels']):
                    if len(out) < max_num_bboxes_in_single_img:
                        num_to_add = max_num_bboxes_in_single_img - len(out)
                        to_add = ['0'] * num_to_add
                        if len(out) == 0:
                            batch_out['labels'][idx] = to_add
                        else:
                            batch_out['labels'][idx] = out + to_add

            # Format outputs to build an InferenceResponse
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses
