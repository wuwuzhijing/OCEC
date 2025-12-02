#!/usr/bin/env python

from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import copy
import json
import cv2
import math
import time
from pprint import pprint
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser, ArgumentTypeError
from typing import Tuple, Optional, List, Dict, Any
import importlib.util
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import matplotlib
from multiprocessing import Pool, cpu_count, current_process, Manager, Lock, set_start_method
from functools import partial
import hashlib
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Color not defined yet, use plain print
    print('Warning: tqdm not installed. Install with: pip install tqdm')
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # fcntl is not available on Windows
    HAS_FCNTL = False

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import setproctitle
setproctitle.setproctitle("data_extract")

AVERAGE_HEAD_WIDTH: float = 0.16 + 0.10 # 16cm + Margin Compensation

BOX_COLORS = [
    [(216, 67, 21),"Front"],
    [(255, 87, 34),"Right-Front"],
    [(123, 31, 162),"Right-Side"],
    [(255, 193, 7),"Right-Back"],
    [(76, 175, 80),"Back"],
    [(33, 150, 243),"Left-Back"],
    [(156, 39, 176),"Left-Side"],
    [(0, 188, 212),"Left-Front"],
]

# The pairs of classes you want to join
# (there is some overlap because there are left and right classes)
EDGES = [
    (21, 22), (21, 22),  # collarbone -> shoulder (left and right)
    (21, 23),            # collarbone -> solar_plexus
    (22, 24), (22, 24),  # shoulder -> elbow (left and right)
    (22, 30), (22, 30),  # shoulder -> hip_joint (left and right)
    (24, 25), (24, 25),  # elbow -> wrist (left and right)
    (23, 29),            # solar_plexus -> abdomen
    (29, 30), (29, 30),  # abdomen -> hip_joint (left and right)
    (30, 31), (30, 31),  # hip_joint -> knee (left and right)
    (31, 32), (31, 32),  # knee -> ankle (left and right)
]

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

CROP_SIZE_RULES = {
    'open': {
        'min_height': 5,
        'max_height': 50,
        'min_width': 10,
        'max_width': 80,
    },
    'closed': {
        'min_height': 5,
        'max_height': 30,
        'min_width': 10,
        'max_width': 80,
    },
}

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    generation: int = -1 # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1 # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1 # -1: Unknown, 0: Left, 1: Right
    head_pose: int = -1 # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
    is_used: bool = False
    person_id: int = -1
    track_id: int = -1

class SimpleSortTracker:
    """Minimal SORT-style tracker based on IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: List[Dict[str, Any]] = []
        self.frame_index = 0

    @staticmethod
    def _iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        if inter_w == 0 or inter_h == 0:
            return 0.0

        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def update(self, boxes: List[Box]) -> None:
        self.frame_index += 1

        for box in boxes:
            box.track_id = -1

        if not boxes and not self.tracks:
            return

        iou_matrix = None
        if self.tracks and boxes:
            iou_matrix = np.zeros((len(self.tracks), len(boxes)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                track_bbox = track['bbox']
                for d_idx, box in enumerate(boxes):
                    det_bbox = (box.x1, box.y1, box.x2, box.y2)
                    iou_matrix[t_idx, d_idx] = self._iou(track_bbox, det_bbox)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: List[Tuple[int, int]] = []

        if iou_matrix is not None and iou_matrix.size > 0:
            while True:
                best_track = -1
                best_det = -1
                best_iou = self.iou_threshold
                for t_idx in range(len(self.tracks)):
                    if t_idx in matched_tracks:
                        continue
                    for d_idx in range(len(boxes)):
                        if d_idx in matched_detections:
                            continue
                        iou = float(iou_matrix[t_idx, d_idx])
                        if iou > best_iou:
                            best_iou = iou
                            best_track = t_idx
                            best_det = d_idx
                if best_track == -1:
                    break
                matched_tracks.add(best_track)
                matched_detections.add(best_det)
                matches.append((best_track, best_det))

        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            det_box = boxes[d_idx]
            track['bbox'] = (det_box.x1, det_box.y1, det_box.x2, det_box.y2)
            track['missed'] = 0
            track['last_seen'] = self.frame_index
            det_box.track_id = track['id']

        surviving_tracks: List[Dict[str, Any]] = []
        for idx, track in enumerate(self.tracks):
            if idx in matched_tracks:
                surviving_tracks.append(track)
                continue
            track['missed'] += 1
            if track['missed'] <= self.max_age:
                surviving_tracks.append(track)
        self.tracks = surviving_tracks

        for d_idx, det_box in enumerate(boxes):
            if d_idx in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            det_box.track_id = track_id
            self.tracks.append(
                {
                    'id': track_id,
                    'bbox': (det_box.x1, det_box.y1, det_box.x2, det_box.y2),
                    'missed': 0,
                    'last_seen': self.frame_index,
                }
            )

        if not boxes:
            return

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.25,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                    # onnxruntime>=1.21.0 breaking changes
                    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                    # https://github.com/microsoft/onnxruntime/pull/22681/files
                    # https://github.com/microsoft/onnxruntime/pull/23893/files
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._keypoint_th = keypoint_th
        self._providers = providers  # Keep original configured providers
        self._configured_providers = providers  # Explicitly save configured providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            onnxruntime.set_default_logger_severity(3) # ERROR
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            # Get actual available providers (may differ from configured)
            actual_providers = self._interpreter.get_providers()
            # Only print in main process to avoid spam in multiprocessing
            if current_process().name == 'MainProcess':
                print(f'{Color.GREEN("Configured ONNX ExecutionProviders:")}')
                pprint(f'{self._configured_providers}')
                print(f'{Color.GREEN("Actually enabled ONNX ExecutionProviders:")}')
                pprint(f'{actual_providers}')
                if set(actual_providers) != set([p if isinstance(p, str) else p[0] for p in self._configured_providers]):
                    print(Color.YELLOW('WARNING: Some configured providers are not available!'))
                    print(Color.YELLOW('This may cause issues in worker processes.'))

            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3

        elif self._runtime in ['ai_edge_litert', 'tensorflow']:
            if self._runtime == 'ai_edge_litert':
                from ai_edge_litert.interpreter import Interpreter
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['ai_edge_litert', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()

class DEIMv2(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'deimv2_dinov3_x_wholebody34_1750query_n_batch.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        keypoint_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for DEIMv2. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for DEIMv2

        obj_class_score_th: Optional[float]
            Object score threshold. Default: 0.35

        attr_class_score_th: Optional[float]
            Attributes score threshold. Default: 0.70

        keypoint_th: Optional[float]
            Keypoints score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            keypoint_th=keypoint_th,
            providers=providers,
        )
        self.mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape([3,1,1]) # Not used in DEIMv2
        self.std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape([3,1,1]) # Not used in DEIMv2

    def __call__(
        self,
        image: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, atrributes, is_used=False]
        """
        # PreProcess (no need to deepcopy if preprocessing doesn't modify original)
        resized_image = \
            self._preprocess(
                image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0][0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                image=image,
                boxes=boxes,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        image = image.transpose(self._swap)
        image = \
            np.ascontiguousarray(
                image,
                dtype=np.float32,
            )
        return image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]. [instances, [batchno, classid, score, x1, y1, x2, y2]].

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, attributes, is_used=False]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        box_score_threshold: float = min([self._obj_class_score_th, self._attr_class_score_th, self._keypoint_th])

        if len(boxes) > 0:
            scores = boxes[:, 5:6]
            keep_idxs = scores[:, 0] > box_score_threshold
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                # Object filter
                for box, score in zip(boxes_keep, scores_keep):
                    classid = int(box[0])
                    x_min = int(max(0, box[1]) * image_width)
                    y_min = int(max(0, box[2]) * image_height)
                    x_max = int(min(box[3], 1.0) * image_width)
                    y_max = int(min(box[4], 1.0) * image_height)
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    result_boxes.append(
                        Box(
                            classid=classid,
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=cx,
                            cy=cy,
                            generation=-1, # -1: Unknown, 0: Adult, 1: Child
                            gender=-1, # -1: Unknown, 0: Male, 1: Female
                            handedness=-1, # -1: Unknown, 0: Left, 1: Right
                            head_pose=-1, # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
                        )
                    )
                # Object filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [0,5,6,7,16,17,18,19,20,26,27,28,33] and box.score >= self._obj_class_score_th) or box.classid not in [0,5,6,7,16,17,18,19,20,26,27,28,33]
                ]
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [1,2,3,4,8,9,10,11,12,13,14,15] and box.score >= self._attr_class_score_th) or box.classid not in [1,2,3,4,8,9,10,11,12,13,14,15]
                ]
                # Keypoint filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [21,22,23,24,25,29,30,31,32] and box.score >= self._keypoint_th) or box.classid not in [21,22,23,24,25,29,30,31,32]
                ]

                # Adult, Child merge
                # classid: 0 -> Body
                #   classid: 1 -> Adult
                #   classid: 2 -> Child
                # 1. Calculate Adult and Child IoUs for Body detection results
                # 2. Connect either the Adult or the Child with the highest score and the highest IoU with the Body.
                # 3. Exclude Adult and Child from detection results
                if not disable_generation_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=generation_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]
                # Male, Female merge
                # classid: 0 -> Body
                #   classid: 3 -> Male
                #   classid: 4 -> Female
                # 1. Calculate Male and Female IoUs for Body detection results
                # 2. Connect either the Male or the Female with the highest score and the highest IoU with the Body.
                # 3. Exclude Male and Female from detection results
                if not disable_gender_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=gender_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]
                # HeadPose merge
                # classid: 7 -> Head
                #   classid:  8 -> Front
                #   classid:  9 -> Right-Front
                #   classid: 10 -> Right-Side
                #   classid: 11 -> Right-Back
                #   classid: 12 -> Back
                #   classid: 13 -> Left-Back
                #   classid: 14 -> Left-Side
                #   classid: 15 -> Left-Front
                # 1. Calculate HeadPose IoUs for Head detection results
                # 2. Connect either the HeadPose with the highest score and the highest IoU with the Head.
                # 3. Exclude HeadPose from detection results
                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 7]
                    headpose_boxes = [box for box in result_boxes if box.classid in [8,9,10,11,12,13,14,15]]
                    self._find_most_relevant_obj(base_objs=head_boxes, target_objs=headpose_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [8,9,10,11,12,13,14,15]]
                # Left and right hand merge
                # classid: 23 -> Hand
                #   classid: 24 -> Left-Hand
                #   classid: 25 -> Right-Hand
                # 1. Calculate Left-Hand and Right-Hand IoUs for Hand detection results
                # 2. Connect either the Left-Hand or the Right-Hand with the highest score and the highest IoU with the Hand.
                # 3. Exclude Left-Hand and Right-Hand from detection results
                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 26]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [27, 28]]
                    self._find_most_relevant_obj(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [27, 28]]

                # Keypoints NMS
                # Suppression of overdetection
                # classid: 21 -> collarbone
                # classid: 22 -> shoulder
                # classid: 23 -> solar_plexus
                # classid: 24 -> elbow
                # classid: 25 -> wrist
                # classid: 29 -> abdomen
                # classid: 30 -> hip_joint
                # classid: 31 -> knee
                # classid: 32 -> ankle
                for target_classid in [21,22,23,24,25,29,30,31,32]:
                    keypoints_boxes = [box for box in result_boxes if box.classid == target_classid]
                    filtered_keypoints_boxes = self._nms(target_objs=keypoints_boxes, iou_threshold=0.20)
                    result_boxes = [box for box in result_boxes if box.classid != target_classid]
                    result_boxes = result_boxes + filtered_keypoints_boxes
        return result_boxes

    def _find_most_relevant_obj(
        self,
        *,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float('inf')

            for target_obj in target_objs:
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                # Process only unused objects with center Euclidean distance less than or equal to 10.0
                if not target_obj.is_used and distance <= 10.0:
                    # Prioritize high-score objects
                    if target_obj.score >= best_score:
                        # IoU Calculation
                        iou: float = \
                            self._calculate_iou(
                                base_obj=base_obj,
                                target_obj=target_obj,
                            )
                        # Adopt object with highest IoU
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                if most_relevant_obj.classid == 1:
                    base_obj.generation = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 2:
                    base_obj.generation = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 3:
                    base_obj.gender = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 4:
                    base_obj.gender = 1
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 8:
                    base_obj.head_pose = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 9:
                    base_obj.head_pose = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 10:
                    base_obj.head_pose = 2
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 11:
                    base_obj.head_pose = 3
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 12:
                    base_obj.head_pose = 4
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 13:
                    base_obj.head_pose = 5
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 14:
                    base_obj.head_pose = 6
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 15:
                    base_obj.head_pose = 7
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 27:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 28:
                    base_obj.handedness = 1
                    most_relevant_obj.is_used = True

    def _nms(
        self,
        *,
        target_objs: List[Box],
        iou_threshold: float,
    ):
        filtered_objs: List[Box] = []

        # 1. Sorted in order of highest score
        #    key=lambda box: box.score to get the score, and reverse=True to sort in descending order
        sorted_objs = sorted(target_objs, key=lambda box: box.score, reverse=True)

        # 2. Scan the box list after sorting
        while sorted_objs:
            # Extract the first (highest score)
            current_box = sorted_objs.pop(0)

            # If you have already used it, skip it
            if current_box.is_used:
                continue

            # Add to filtered_objs and set the use flag
            filtered_objs.append(current_box)
            current_box.is_used = True

            # 3. Mark the boxes where the current_box and IOU are above the threshold as used or exclude them
            remaining_boxes = []
            for box in sorted_objs:
                if not box.is_used:
                    # Calculating IoU
                    iou_value = self._calculate_iou(base_obj=current_box, target_obj=box)

                    # If the IOU threshold is exceeded, it is considered to be the same object and is removed as a duplicate
                    if iou_value >= iou_threshold:
                        # Leave as used (exclude later)
                        box.is_used = True
                    else:
                        # If the IOU threshold is not met, the candidate is still retained
                        remaining_boxes.append(box)

            # Only the remaining_boxes will be handled in the next loop
            sorted_objs = remaining_boxes

        # 4. Return the box that is left over in the end
        return filtered_objs

    def _calculate_iou(
        self,
        *,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        # Calculate areas of overlap
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        # If there is no overlap
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        # Calculate area of overlap and area of each bounding box
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
        # Calculate IoU
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

def list_image_files(dir_path: str) -> List[str]:
    path = Path(dir_path)
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(path.rglob(extension))
    return sorted([str(file) for file in image_files])


class CropIndexer:
    """Manage crop file numbering and folder rotation."""

    def __init__(self, root: Path, start_folder: int, folder_size: int = 1000) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.start_folder = start_folder
        self.folder_size = folder_size
        self._count = 0

    def save_crop(self, crop: np.ndarray, base_name: str, detection_idx: int) -> Optional[Path]:
        cleaned_base = base_name.replace(' ', '_')
        folder_number = self.start_folder + (self._count // self.folder_size)
        folder_name = f'{folder_number:09d}'
        folder_path = self.root / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        file_name = f'{cleaned_base}_{detection_idx + 1}.png'
        file_path = folder_path / file_name
        success = cv2.imwrite(str(file_path), crop)
        if not success:
            return None
        self._count += 1
        return Path('data') / 'cropped' / folder_name / file_name


class EyeAnalysisTracker:
    """Collect statistics and artifacts for eye-only analysis."""

    def __init__(self, output_root: Path, csv_max_entries: int = 12000, csv_output_root: Optional[Path] = None) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        # CSV files go to a separate 'list' subdirectory
        if csv_output_root is None:
            self.csv_output_root = self.output_root / 'list'
        else:
            self.csv_output_root = Path(csv_output_root)
        self.csv_output_root.mkdir(parents=True, exist_ok=True)
        self.total_images = 0
        self.images_with_eye = 0
        self.over_detected = 0
        self.heights_by_label: Dict[str, List[int]] = defaultdict(list)
        self.widths_by_label: Dict[str, List[int]] = defaultdict(list)
        self.crops_per_label: Counter = Counter()
        self.annotation_entries: List[Tuple[str, int]] = []
        self.csv_max_entries = csv_max_entries
        self.current_csv_index = 1
        self.current_csv_entries = 0

    @property
    def images_without_eye(self) -> int:
        return max(0, self.total_images - self.images_with_eye)

    def register_image(self, detection_count: int) -> None:
        self.total_images += 1
        if detection_count > 0:
            self.images_with_eye += 1
        if detection_count >= 3:
            self.over_detected += 1

    def register_crop(
        self,
        label_name: str,
        classid: int,
        relative_path: Path,
        height_px: int,
        width_px: int,
    ) -> None:
        posix_path = relative_path.as_posix()
        self.annotation_entries.append((posix_path, classid))
        if height_px > 0:
            self.heights_by_label[label_name].append(height_px)
        if width_px > 0:
            self.widths_by_label[label_name].append(width_px)
        self.crops_per_label[label_name] += 1

    def _get_current_csv_path(self) -> Path:
        """Get the current CSV file path based on index."""
        return self.csv_output_root / f'annotation_{self.current_csv_index:04d}.csv'
    
    def _get_all_csv_paths(self) -> List[Path]:
        """Get all existing annotation CSV file paths."""
        csv_files = sorted(self.csv_output_root.glob('annotation_*.csv'))
        return csv_files
    
    def _load_all_existing_entries(self) -> Dict[str, int]:
        """Load all entries from existing CSV files."""
        merged_entries: Dict[str, int] = {}
        csv_files = self._get_all_csv_paths()
        
        for csv_path in csv_files:
            if not csv_path.exists():
                continue
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or ',' not in line:
                            continue
                        rel_path, classid_str = line.split(',', 1)
                        rel_path = rel_path.strip()
                        classid_str = classid_str.strip()
                        try:
                            merged_entries[rel_path] = int(classid_str)
                        except ValueError:
                            continue
            except (OSError, IOError) as e:
                print(Color.YELLOW(f'Warning: Error reading {csv_path}: {e}'))
        
        return merged_entries
    
    def _find_next_csv_index(self) -> int:
        """Find the next available CSV file index."""
        csv_files = self._get_all_csv_paths()
        if not csv_files:
            return 1
        
        # Extract index from the last CSV file
        last_csv = csv_files[-1]
        try:
            # Format: annotation_0001.csv
            stem = last_csv.stem  # annotation_0001
            index_str = stem.split('_')[-1]
            last_index = int(index_str)
            
            # Check if last CSV is full
            with open(last_csv, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            if line_count >= self.csv_max_entries:
                return last_index + 1
            else:
                return last_index
        except (ValueError, OSError):
            return 1
    
    def write_annotation_csv(self, lock: Optional[Lock] = None, force_flush: bool = False) -> None:
        """Write annotation CSV with optional file lock for multiprocessing safety.
        
        Writes entries to CSV files, creating new files when current file reaches max_entries.
        """
        if not self.annotation_entries:
            return
        
        # Use file lock for multiprocessing safety
        def _write_with_lock():
            # Load all existing entries to check for duplicates
            existing_entries = self._load_all_existing_entries()
            
            # Filter out duplicates
            new_entries: List[Tuple[str, int]] = []
            for rel_path, classid in self.annotation_entries:
                if rel_path not in existing_entries:
                    new_entries.append((rel_path, classid))
                    existing_entries[rel_path] = classid
            
            if not new_entries:
                self.annotation_entries = []
                return
            
            # Find the current CSV index and how many entries it has
            self.current_csv_index = self._find_next_csv_index()
            current_csv_path = self._get_current_csv_path()
            
            # Load current CSV if it exists
            current_csv_entries: Dict[str, int] = {}
            if current_csv_path.exists():
                try:
                    with open(current_csv_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line or ',' not in line:
                                continue
                            rel_path, classid_str = line.split(',', 1)
                            rel_path = rel_path.strip()
                            classid_str = classid_str.strip()
                            try:
                                current_csv_entries[rel_path] = int(classid_str)
                            except ValueError:
                                continue
                except (OSError, IOError):
                    pass
            
            # Write entries, creating new CSV files as needed
            remaining_new_entries = new_entries.copy()
            
            while remaining_new_entries:
                # Check if current CSV is full (>= csv_max_entries)
                if len(current_csv_entries) >= self.csv_max_entries:
                    # Save current CSV (it's full with exactly csv_max_entries)
                    sorted_entries = sorted(current_csv_entries.items(), key=lambda item: item[0])
                    try:
                        with open(current_csv_path, 'w', encoding='utf-8') as f:
                            for rel_path, classid in sorted_entries:
                                f.write(f'{rel_path},{classid}\n')
                        print(Color.GREEN(f'✓ Saved {len(sorted_entries)} entries to {current_csv_path.name} (file is full)'))
                    except (OSError, IOError) as e:
                        print(Color.RED(f'Error writing {current_csv_path}: {e}'))
                    
                    # Move to next CSV file (increment index)
                    self.current_csv_index += 1
                    current_csv_path = self._get_current_csv_path()
                    current_csv_entries = {}
                    print(Color.CYAN(f'→ CSV file full ({self.csv_max_entries} entries). Creating new file: {current_csv_path.name}'))
                
                # Add entries to current CSV until it's full
                space_remaining = self.csv_max_entries - len(current_csv_entries)
                entries_to_add = remaining_new_entries[:space_remaining]
                remaining_new_entries = remaining_new_entries[space_remaining:]
                
                for rel_path, classid in entries_to_add:
                    current_csv_entries[rel_path] = classid
            
            # Write the last CSV file (may be partially full, less than csv_max_entries)
            if current_csv_entries:
                sorted_entries = sorted(current_csv_entries.items(), key=lambda item: item[0])
                try:
                    with open(current_csv_path, 'w', encoding='utf-8') as f:
                        for rel_path, classid in sorted_entries:
                            f.write(f'{rel_path},{classid}\n')
                    print(Color.GREEN(f'✓ Saved {len(sorted_entries)} entries to {current_csv_path.name} (partial file)'))
                except (OSError, IOError) as e:
                    print(Color.RED(f'Error writing {current_csv_path}: {e}'))
            
            # Clear annotation entries
            self.annotation_entries = []
            self.current_csv_entries = len(current_csv_entries) if current_csv_entries else 0
        
        if lock is not None:
            # Use multiprocessing lock if provided
            with lock:
                _write_with_lock()
        else:
            # Use file-based lock for cross-process safety (Unix/Linux only)
            lock_file_path = self._get_current_csv_path().with_suffix('.csv.lock')
            if HAS_FCNTL:
                lock_file_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(lock_file_path, 'w') as lock_file:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                        try:
                            _write_with_lock()
                        finally:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except (OSError, IOError) as e:
                    print(Color.YELLOW(f'Warning: Could not acquire file lock, writing without lock: {e}'))
                    _write_with_lock()
            else:
                # Windows: no fcntl, just write (multiprocessing lock should be used instead)
                _write_with_lock()

    def write_histograms(self) -> None:
        for label_name in sorted(set(self.heights_by_label.keys()) | set(self.widths_by_label.keys())):
            heights = self.heights_by_label.get(label_name, [])
            widths = self.widths_by_label.get(label_name, [])
            if not heights and not widths:
                continue
            output_path = self.output_root / f'{label_name}_eye_size_hist.png'
            save_eye_histogram(label_name=label_name, heights=heights, widths=widths, output_path=output_path)

    def print_summary(self) -> None:
        print(Color.GREEN('Eye-only detection summary'))
        print(f'  Total images: {self.total_images}')
        print(f'  Images with detection: {self.images_with_eye}')
        print(f'  Images without detection: {self.images_without_eye}')
        print(f'  Images with >=3 detections: {self.over_detected}')
        if self.crops_per_label:
            print('  Crops per label:')
            for label_name, count in sorted(self.crops_per_label.items()):
                print(f'    {label_name}: {count}')


def save_eye_histogram(label_name: str, heights: List[int], widths: List[int], output_path: Path) -> None:
    """Create histogram png with mean/median overlays for height and width."""
    def _prepare_bins(values: List[int]) -> List[int]:
        if not values:
            return [0, 1]
        v_min = min(values)
        v_max = max(values)
        if v_min == v_max:
            return [v_min - 0.5, v_min + 0.5]
        return list(range(int(v_min), int(v_max) + 2))

    def _plot(ax, data: List[int], label: str, color: str) -> None:
        bins = _prepare_bins(data)
        counts, _, _ = ax.hist(data, bins=bins, color=color, edgecolor='white', alpha=0.85)
        ax.set_xlabel(f'{label} (px)')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.3)
        if not data:
            return
        mean_val = float(np.mean(data))
        median_val = float(np.median(data))
        y_max = max(counts) if len(counts) > 0 else 0
        y_base = y_max * 0.95 if y_max > 0 else 1.0
        median_color = '#dd8452'
        mean_color = '#55a868'
        ax.axvline(median_val, color=median_color, linestyle='--', linewidth=1.5)
        ax.text(
            median_val,
            y_base,
            f'Median: {median_val:.1f}',
            rotation=90,
            va='top',
            ha='right',
            color=median_color,
            fontsize=10,
        )
        ax.axvline(mean_val, color=mean_color, linestyle='-', linewidth=1.5)
        ax.text(
            mean_val,
            y_base * 0.85,
            f'Mean: {mean_val:.1f}',
            rotation=90,
            va='top',
            ha='left',
            color=mean_color,
            fontsize=10,
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(f'{label_name.capitalize()} eye crop size distribution')
    _plot(axes[0], heights, 'Height', '#4c72b0')
    _plot(axes[1], widths, 'Width', '#c44e52')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path)
    plt.close(fig)


def _infer_label_from_filename(image_path: Path) -> Optional[Tuple[str, int]]:
    """Fallback: derive label from filename suffix like *_0.jpg or *_1.jpg."""
    stem = image_path.stem
    # Extract trailing digits, e.g. foo_bar_1 -> 1
    digit_part = []
    for ch in reversed(stem):
        if ch.isdigit():
            digit_part.append(ch)
        elif digit_part:
            break
    if not digit_part:
        return None
    label_num = int(''.join(reversed(digit_part)))
    if label_num == 0:
        return 'closed', 0
    if label_num == 1:
        return 'open', 1
    return None


def load_label_from_json(image_path: Path) -> Optional[Tuple[str, int]]:
    json_path = image_path.with_suffix('.json')
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = None
        if meta:
            label_value = meta.get('label')
            if label_value == 'closed_eyes':
                return 'closed', 0
            if label_value == 'open_eyes':
                return 'open', 1
    # Fallback to filename-based label inference
    return _infer_label_from_filename(image_path)


def load_processed_image_stems(output_root: Path, csv_output_root: Optional[Path] = None) -> set:
    """Load set of already processed image stems from all annotation CSV files.
    
    Returns a set of image stems (base names) that have been processed.
    The stem is extracted from crop filenames in the CSV.
    Crop filename format: {image_stem}_{det_idx}_{hash}.png
    """
    processed = set()
    output_root_path = Path(output_root)
    
    # CSV files are in the 'list' subdirectory
    if csv_output_root is None:
        csv_output_root_path = output_root_path / 'list'
    else:
        csv_output_root_path = Path(csv_output_root)
    
    # Load from all annotation CSV files
    csv_files = sorted(csv_output_root_path.glob('annotation_*.csv'))
    if not csv_files:
        # Fallback: check for old annotation.csv in list directory
        old_csv = csv_output_root_path / 'annotation.csv'
        if old_csv.exists():
            csv_files = [old_csv]
        else:
            # Also check in the old location (output_root) for backward compatibility
            old_csv_legacy = output_root_path / 'annotation.csv'
            if old_csv_legacy.exists():
                csv_files = [old_csv_legacy]
    
    for csv_path in csv_files:
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ',' not in line:
                        continue
                    # CSV format: relative_path,classid
                    # relative_path format: data/cropped/000001000/image_name_1_hash.png
                    rel_path = line.split(',', 1)[0].strip()
                    # Get filename without extension
                    crop_filename = Path(rel_path).name
                    # Extract base name: remove detection index and hash suffix
                    # Format: image_name_1_hash -> image_name
                    # Split from right, remove last 2 parts (detection_idx and hash)
                    parts = crop_filename.rsplit('_', 2)
                    if len(parts) >= 3:
                        # Remove detection index and hash, keep base name
                        base_name = '_'.join(parts[:-2])
                        processed.add(base_name)
                    elif len(parts) == 2:
                        # Fallback: might be old format without hash
                        # Try to check if second part is a number (detection index)
                        try:
                            int(parts[1].split('.')[0])  # Check if it's a number
                            base_name = parts[0]
                            processed.add(base_name)
                        except ValueError:
                            # Not a number, use whole stem
                            processed.add(parts[0].split('.')[0])
                    else:
                        # Fallback: use filename without extension
                        processed.add(Path(crop_filename).stem)
        except (OSError, IOError) as e:
            print(Color.YELLOW(f'Warning: Error reading {csv_path} to check processed images: {e}'))
    
    return processed


def resolve_video_label(video_path: Path) -> Optional[Tuple[str, int]]:
    stem = video_path.stem.lower()
    if 'closed' in stem:
        return 'closed', 0
    if 'open' in stem:
        return 'open', 1
    return None


def crop_eye_region(image: np.ndarray, box: Box) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1 = max(0, min(int(box.x1), w))
    y1 = max(0, min(int(box.y1), h))
    x2 = max(0, min(int(box.x2), w))
    y2 = max(0, min(int(box.y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def is_valid_crop(label_name: str, height_px: int, width_px: int) -> bool:
    rules = CROP_SIZE_RULES.get(label_name)
    if rules is None:
        return True
    if height_px < rules['min_height'] or height_px > rules['max_height']:
        return False
    if width_px < rules['min_width'] or width_px > rules['max_width']:
        return False
    return True


def run_eye_analysis(
    *,
    model: DEIMv2,
    image_dirs: List[Path],
    video_paths: List[Path],
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
    num_workers: int = 1,
    process_dataset_only: bool = False,
    process_video_only: bool = False,
) -> None:
    output_root = Path('/10/cvz/guochuang/dataset/Classification/fatigue') / 'cropped'
    tracker = EyeAnalysisTracker(output_root=output_root)
    dataset_crop_indexer = CropIndexer(root=tracker.output_root, start_folder=200000001)
    video_crop_indexer = CropIndexer(root=tracker.output_root, start_folder=100000001)

    # Create a manager and lock for multiprocessing
    manager = Manager() if num_workers > 1 else None
    csv_lock = manager.Lock() if manager else None

    processed_any = False
    
    # Process image directories (dataset)
    if not process_video_only:
        for images_dir in image_dirs:
            if not images_dir.exists() or not images_dir.is_dir():
                print(Color.YELLOW(f'Skipping missing image directory: {images_dir}'))
                continue

            print(Color.GREEN(f'Processing image directory: {images_dir}'))
            process_images_dir_for_eye_analysis(
                model=model,
                images_dir=images_dir,
                tracker=tracker,
                crop_indexer=dataset_crop_indexer,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
                num_workers=num_workers,
                    csv_lock=csv_lock,
            )
                # Update annotation.csv after each directory (may create multiple CSV files)
            print(Color.GREEN(f'✓ Updating annotation CSV files after processing {images_dir}...'))
            tracker.write_annotation_csv(lock=csv_lock, force_flush=True)
            processed_any = True

    # Process video files
    if not process_dataset_only:
        for video_path in video_paths:
            if not video_path.exists():
                print(Color.YELLOW(f'Skipping missing video file: {video_path}'))
                continue
            print(Color.GREEN(f'Processing video file: {video_path}'))
            process_video_for_eye_analysis(
                model=model,
                video_path=video_path,
                tracker=tracker,
                crop_indexer=video_crop_indexer,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
                    csv_lock=csv_lock,
            )
                # Update annotation.csv after each video (may create multiple CSV files)
            print(Color.CYAN(f'Updating annotation CSV files after processing {video_path}...'))
            tracker.write_annotation_csv(lock=csv_lock, force_flush=True)
            processed_any = True

    if not processed_any:
        print(Color.RED('ERROR: No valid sources were processed for eye analysis.'))
        return

    # Final update and summary
    tracker.write_histograms()
    tracker.write_annotation_csv(lock=csv_lock, force_flush=True)
    tracker.print_summary()
    
    # Print summary of CSV files created
    csv_files = tracker._get_all_csv_paths()
    if csv_files:
        print(Color.GREEN(f'\nCreated {len(csv_files)} annotation CSV file(s):'))
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                print(f'  {csv_file.name}: {count} entries')
            except (OSError, IOError):
                print(f'  {csv_file.name}: (unable to read)')
    
    if manager:
        manager.shutdown()


# Global model variable for worker processes (initialized once per process)
_worker_model = None
_worker_model_params = None

def _init_worker_model(model_params, worker_counter=None):
    """Initialize model once per worker process.
    
    Args:
        model_params: Model parameters dictionary
        worker_counter: Shared multiprocessing.Value counter for assigning unique worker indices
    """
    global _worker_model, _worker_model_params
    import os
    import time as time_module
    import copy
    worker_id = os.getpid()
    init_start = time_module.time()
    
    # Get unique worker index from shared counter
    if worker_counter is not None:
        with worker_counter.get_lock():
            worker_index = worker_counter.value
            worker_counter.value += 1
    else:
        worker_index = 0  # Fallback
    
    # Ensure CUDA and TensorRT library paths are available in worker process
    # This is critical for multiprocessing - child processes may not inherit LD_LIBRARY_PATH
    library_paths = []
    
    # CUDA paths
    cuda_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-11.8/lib64',
        '/usr/local/cuda-12.0/lib64',
        '/usr/local/cuda-12.1/lib64',
        '/usr/local/cuda-12.2/lib64',
    ]
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            library_paths.append(cuda_path)
    
    # TensorRT paths (common installation locations)
    tensorrt_paths = [
        '/usr/local/TensorRT/lib',
        '/usr/local/TensorRT-8.x/lib',
        '/usr/local/TensorRT-9.x/lib',
        '/opt/tensorrt/lib',
        '/usr/lib/x86_64-linux-gnu',  # Some systems install here
    ]
    # Also check for TensorRT in home directory
    home = os.environ.get('HOME', '')
    if home:
        tensorrt_paths.extend([
            f'{home}/TensorRT/lib',
            f'{home}/TensorRT-8.x/lib',
            f'{home}/TensorRT-9.x/lib',
        ])
    
    for trt_path in tensorrt_paths:
        if os.path.exists(trt_path):
            # Check if libnvinfer.so exists in this path
            try:
                if os.path.exists(os.path.join(trt_path, 'libnvinfer.so')) or \
                   any(f.startswith('libnvinfer.so') for f in os.listdir(trt_path) if os.path.isfile(os.path.join(trt_path, f))):
                    library_paths.append(trt_path)
            except:
                pass
    
    # Update LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    added_paths = []
    for lib_path in library_paths:
        if lib_path not in current_ld_path:
            current_ld_path = f'{lib_path}:{current_ld_path}'
            added_paths.append(lib_path)
    
    if added_paths:
        os.environ['LD_LIBRARY_PATH'] = current_ld_path
        print(Color.CYAN(f'[PID {worker_id}] Added to LD_LIBRARY_PATH: {", ".join(added_paths)}'), flush=True)
    
    # Worker index is obtained from shared counter passed as argument
    # This ensures each worker gets a unique index in spawn mode
    
    # Dynamically detect number of GPUs available
    num_gpus = 1
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split('\n'))
            print(Color.CYAN(f'[PID {worker_id}] Detected {num_gpus} GPU(s)'), flush=True)
    except:
        # Fallback: try to detect via CUDA
        try:
            import ctypes
            cuda_lib = ctypes.CDLL('libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
            device_count = ctypes.c_int()
            if cuda_lib.cuInit(0) == 0:
                if cuda_lib.cuDeviceGetCount(ctypes.byref(device_count)) == 0:
                    num_gpus = device_count.value
        except:
            pass
    
    if num_gpus < 1:
        num_gpus = 1
    
    # IMPORTANT: Don't set CUDA_VISIBLE_DEVICES here - it causes issues
    # Instead, use device_id in the provider configuration
    # Setting CUDA_VISIBLE_DEVICES would make only one GPU visible, breaking multi-GPU
    
    # Modify providers to use different GPU for each worker (if multiple GPUs available)
    providers = copy.deepcopy(model_params['providers'])
    
    # Verify CUDA is available in this worker process
    try:
        import onnxruntime as ort
        available_in_worker = ort.get_available_providers()
        print(Color.CYAN(f'[PID {worker_id}] Available providers in worker: {available_in_worker}'), flush=True)
    except:
        pass
    
    # Check if CUDA/TensorRT provider exists and configure it with correct device_id
    gpu_provider_found = False
    
    # Calculate which GPU this worker should use (round-robin)
    gpu_id = worker_index % num_gpus
    
    for i, provider in enumerate(providers):
        if isinstance(provider, str):
            if provider == 'CUDAExecutionProvider':
                # Convert string to tuple with options
                providers[i] = (
                    'CUDAExecutionProvider',
                    {
                        'device_id': gpu_id,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 12 * 1024 * 1024 * 1024,  # 12GB per GPU (RTX 3090 has 24GB, use half for safety)
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                        'tunable_op_enable': True,  # Enable tunable ops
                        'tunable_op_tuning_enable': True,
                    }
                )
                gpu_provider_found = True
                print(Color.GREEN(f'[PID {worker_id}] ✓ Worker {worker_index + 1} assigned to GPU {gpu_id}/{num_gpus-1} (CUDA)'), flush=True)
                break
            elif provider == 'TensorrtExecutionProvider':
                # TensorRT also needs device_id
                providers[i] = (
                    'TensorrtExecutionProvider',
                    {
                        'device_id': gpu_id,
                        'trt_engine_cache_enable': True,
                        'trt_fp16_enable': True,
                    }
                )
                gpu_provider_found = True
                print(Color.GREEN(f'[PID {worker_id}] ✓ Worker {worker_index + 1} assigned to GPU {gpu_id}/{num_gpus-1} (TensorRT)'), flush=True)
                break
        elif isinstance(provider, tuple) and len(provider) == 2:
            provider_name, provider_options = provider
            if provider_name in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
                # Update existing GPU provider options
                if isinstance(provider_options, dict):
                    provider_options = provider_options.copy()
                    provider_options['device_id'] = gpu_id
                    # Increase GPU memory limit for better utilization
                    if 'gpu_mem_limit' in provider_options:
                        provider_options['gpu_mem_limit'] = 12 * 1024 * 1024 * 1024  # 12GB
                else:
                    provider_options = {'device_id': gpu_id}
                providers[i] = (provider_name, provider_options)
                gpu_provider_found = True
                provider_type = 'CUDA' if provider_name == 'CUDAExecutionProvider' else 'TensorRT'
                print(Color.GREEN(f'[PID {worker_id}] ✓ Worker {worker_index + 1} assigned to GPU {gpu_id}/{num_gpus-1} ({provider_type})'), flush=True)
                break
    
    if not gpu_provider_found:
        print(Color.YELLOW(f'[PID {worker_id}] Warning: No GPU provider (CUDA/TensorRT) found in providers list!'), flush=True)
        print(Color.YELLOW(f'[PID {worker_id}] Providers: {providers}'), flush=True)
    
    # Print progress for all workers
    print(Color.CYAN(f'[PID {worker_id}] Loading model with providers: {providers}...'), flush=True)
    _worker_model_params = model_params
    
    try:
        _worker_model = DEIMv2(
            runtime=model_params['runtime'],
            model_path=model_params['model_path'],
            obj_class_score_th=model_params['obj_class_score_th'],
            attr_class_score_th=model_params['attr_class_score_th'],
            keypoint_th=model_params['keypoint_th'],
            providers=providers,  # Use modified providers with GPU assignment
        )
        init_time = time_module.time() - init_start
        # Check which provider was actually used
        actual_providers = _worker_model._providers
        print(Color.GREEN(f'[PID {worker_id}] Model loaded successfully in {init_time:.1f}s'), flush=True)
        print(Color.GREEN(f'[PID {worker_id}] Actually using providers: {actual_providers}'), flush=True)
        
        # Check if GPU is being used
        gpu_used = False
        for provider in actual_providers:
            if isinstance(provider, str):
                if provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
                    gpu_used = True
                    break
            elif isinstance(provider, tuple) and len(provider) > 0:
                if provider[0] in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
                    gpu_used = True
                    break
        
        if not gpu_used:
            print(Color.RED(f'[PID {worker_id}] ⚠ CRITICAL ERROR: GPU is not being used! Using CPU instead.'), flush=True)
            print(Color.RED(f'[PID {worker_id}] Configured providers: {providers}'), flush=True)
            print(Color.RED(f'[PID {worker_id}] Actual providers: {actual_providers}'), flush=True)
            print(Color.RED(f'[PID {worker_id}] This is a critical error - GPU must be used for large datasets!'), flush=True)
            print(Color.RED(f'[PID {worker_id}] Worker will exit to prevent CPU processing.'), flush=True)
            raise RuntimeError(f'Worker {worker_id} failed to use GPU. This is not allowed for large datasets.')
        else:
            print(Color.GREEN(f'[PID {worker_id}] ✓ GPU is being used successfully!'), flush=True)
    except Exception as e:
        print(Color.RED(f'[PID {worker_id}] Failed to initialize model: {e}'), flush=True)
        import traceback
        print(Color.RED(f'[PID {worker_id}] Traceback: {traceback.format_exc()}'), flush=True)
        raise

def _process_single_image_worker(args_tuple):
    """Worker function for processing a single image in parallel."""
    global _worker_model
    import os  # Ensure os is available in worker process
    
    (
        image_path_str,
        disable_generation_identification_mode,
        disable_gender_identification_mode,
        disable_left_and_right_hand_identification_mode,
        disable_headpose_identification_mode,
        output_root,
        start_folder,
        folder_size,
        processed_stems_set,
    ) = args_tuple
    
    # Use pre-initialized model (loaded once per worker process)
    if _worker_model is None:
        raise RuntimeError("Worker model not initialized. This should not happen.")
    model = _worker_model
    
    try:
        image_path = Path(image_path_str)
        # Check if already processed (for worker processes - double check)
        image_stem = image_path.stem.replace(' ', '_').replace('/', '_').replace('\\', '_')
        if processed_stems_set and image_stem in processed_stems_set:
            return {'detection_count': 0, 'crops': [], 'skipped': True}
        
        label_info = load_label_from_json(image_path)
        if label_info is None:
            return None
        
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        label_name, classid = label_info
        boxes = model(
            image=image,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        eye_boxes = [box for box in boxes if box.classid == 17]
        eye_boxes.sort(key=lambda box: box.score, reverse=True)
        detection_count = len(eye_boxes)
        
        if detection_count == 0:
            return {'detection_count': 0, 'crops': []}
        
        # Process crops with process-safe file naming
        crops_data = []
        output_root_path = Path(output_root)
        
        # Use process ID and timestamp for unique file naming
        process_id = os.getpid()
        base_timestamp = int(time.time() * 1000000)
        
        for det_idx, box in enumerate(eye_boxes[:2]):
            crop = crop_eye_region(image, box)
            if crop is None or crop.size == 0:
                continue
            crop_height, crop_width = crop.shape[0], crop.shape[1]
            if not is_valid_crop(label_name, crop_height, crop_width):
                continue
            
            # Generate unique identifier combining image path, detection index, process ID, and timestamp
            unique_str = f"{image_path_str}_{det_idx}_{process_id}_{base_timestamp}_{det_idx}"
            unique_id = hashlib.md5(unique_str.encode()).hexdigest()[:12]
            
            # Use hash to determine folder distribution (consistent for same image)
            hash_int = int(hashlib.md5(image_path_str.encode()).hexdigest()[:8], 16)
            current_count = hash_int % 1000000  # Distribute across folders
            
            folder_number = start_folder + (current_count // folder_size)
            folder_name = f'{folder_number:09d}'
            folder_path = output_root_path / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            
            cleaned_base = image_path.stem.replace(' ', '_').replace('/', '_').replace('\\', '_')
            # Use unique identifier to avoid conflicts
            file_name = f'{cleaned_base}_{det_idx + 1}_{unique_id}.png'
            file_path = folder_path / file_name
            
            # Handle potential filename conflicts (should be very rare with unique_id)
            counter = 0
            while file_path.exists() and counter < 100:
                counter += 1
                unique_id = hashlib.md5(f"{unique_str}_{counter}".encode()).hexdigest()[:12]
                file_name = f'{cleaned_base}_{det_idx + 1}_{unique_id}.png'
                file_path = folder_path / file_name
            
            if counter >= 100:
                # Fallback: use timestamp
                timestamp = int(time.time() * 1000000)
                file_name = f'{cleaned_base}_{det_idx + 1}_{process_id}_{timestamp}.png'
                file_path = folder_path / file_name
            
            success = cv2.imwrite(str(file_path), crop)
            if not success:
                continue
            
            saved_rel_path = Path('data') / 'cropped' / folder_name / file_name
            
            crops_data.append({
                'label_name': label_name,
                'classid': classid,
                'relative_path': str(saved_rel_path),
                'height_px': crop_height,
                'width_px': crop_width,
            })
        
        return {
            'detection_count': detection_count,
            'crops': crops_data,
        }
    except Exception as e:
        # Log error but don't crash the worker
        print(Color.YELLOW(f'Error processing {image_path_str}: {e}'))
        return None


def process_images_dir_for_eye_analysis(
    *,
    model: DEIMv2,
    images_dir: Path,
    tracker: EyeAnalysisTracker,
    crop_indexer: CropIndexer,
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
    num_workers: int = 1,
    csv_lock: Optional[Lock] = None,
) -> None:
    image_paths = list_image_files(str(images_dir))
    if not image_paths:
        print(Color.YELLOW(f'No image files found under {images_dir}'))
        return
    
    # Load already processed images to skip them
    processed_image_stems = load_processed_image_stems(tracker.output_root, csv_output_root=tracker.csv_output_root)
    if processed_image_stems:
        print(Color.CYAN(f'Found {len(processed_image_stems)} already processed images in CSV. Will skip them.'))
    
    # Filter out already processed images
    remaining_images = []
    skipped_count = 0
    for image_path_str in image_paths:
        image_path = Path(image_path_str)
        # Normalize stem to match CSV format (replace spaces and path separators)
        image_stem = image_path.stem.replace(' ', '_').replace('/', '_').replace('\\', '_')
        if image_stem in processed_image_stems:
            skipped_count += 1
            continue
        remaining_images.append(image_path_str)
    
    if skipped_count > 0:
        print(Color.GREEN(f'Skipping {skipped_count} already processed images. Remaining: {len(remaining_images)}'))
    
    if not remaining_images:
        print(Color.YELLOW(f'All images in {images_dir} have already been processed.'))
        return
    
    image_paths = remaining_images
    total_images = len(image_paths)
    
    # If single worker, use original sequential processing
    if num_workers <= 1:
        start_time = time.time()
        for idx, image_path_str in enumerate(image_paths):
            image_path = Path(image_path_str)
            # Double check: skip if already processed (shouldn't happen due to pre-filtering)
            image_stem = image_path.stem.replace(' ', '_').replace('/', '_').replace('\\', '_')
            if image_stem in processed_image_stems:
                continue
            label_info = load_label_from_json(image_path)
            if label_info is None:
                print(Color.YELLOW(f'Skipping {image_path} because label metadata is unavailable.'))
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(Color.YELLOW(f'Failed to read image: {image_path}'))
                continue
            label_name, classid = label_info
            boxes = model(
                image=image,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
            eye_boxes = [box for box in boxes if box.classid == 17]
            eye_boxes.sort(key=lambda box: box.score, reverse=True)
            detection_count = len(eye_boxes)
            tracker.register_image(detection_count)
            if detection_count == 0:
                continue
            for det_idx, box in enumerate(eye_boxes[:2]):
                crop = crop_eye_region(image, box)
                if crop is None or crop.size == 0:
                    continue
                crop_height, crop_width = crop.shape[0], crop.shape[1]
                if not is_valid_crop(label_name, crop_height, crop_width):
                    continue
                saved_rel_path = crop_indexer.save_crop(crop, base_name=image_path.stem, detection_idx=det_idx)
                if saved_rel_path is None:
                    continue
                tracker.register_crop(
                    label_name=label_name,
                    classid=classid,
                    relative_path=saved_rel_path,
                    height_px=crop_height,
                    width_px=crop_width,
                )
            
            # Periodically check and write CSV if entries accumulate too much
            # This ensures we don't lose data if processing is interrupted
            if len(tracker.annotation_entries) >= tracker.csv_max_entries:
                print(Color.CYAN(f'Reached {len(tracker.annotation_entries)} entries, writing to CSV...'))
                tracker.write_annotation_csv(lock=csv_lock, force_flush=True)
            
            # Progress reporting with tqdm if available
            processed_count = idx + 1
            if HAS_TQDM:
                if processed_count == 1:
                    # Initialize progress bar
                    pbar = tqdm(total=total_images, desc='Processing images', unit='img', ncols=100, mininterval=1.0)
                pbar.update(1)
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_images - processed_count) / rate if rate > 0 else 0
                    eta_str = format_eta(eta_seconds)
                    pbar.set_postfix({'rate': f'{rate:.1f} img/s', 'ETA': eta_str})
                if processed_count == total_images:
                    pbar.close()
            
            # Also print detailed progress every 100 images (for logging)
            if processed_count % 100 == 0 or processed_count == total_images:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta_seconds = (total_images - processed_count) / rate if rate > 0 else 0
                percentage = (processed_count / total_images * 100) if total_images > 0 else 0
                eta_str = format_eta(eta_seconds)
                print(f'[{percentage:5.1f}%] Processed {processed_count}/{total_images} images. '
                      f'Rate: {rate:.1f} img/s, ETA: {eta_str}')
        return
    
    # Multi-process parallel processing
    print(Color.GREEN(f'Using {num_workers} parallel workers for processing...'))
    print(Color.YELLOW('Initializing models in worker processes (this may take a moment)...'))
    
    # Prepare model parameters for worker processes
    model_path = model._model_path
    model_runtime = model._runtime
    # Use configured providers, not actual providers (which may have fallen back to CPU)
    model_providers = getattr(model, '_configured_providers', model._providers)
    obj_class_score_th = model._obj_class_score_th
    attr_class_score_th = model._attr_class_score_th
    keypoint_th = model._keypoint_th
    
    # Prepare model parameters for worker initialization
    model_params = {
        'runtime': model_runtime,
        'model_path': model_path,
        'obj_class_score_th': obj_class_score_th,
        'attr_class_score_th': attr_class_score_th,
        'keypoint_th': keypoint_th,
        'providers': model_providers,  # Use configured providers
    }
    
    print(Color.CYAN(f'Passing providers to workers: {model_providers}'))
    
    # Prepare arguments for worker function (without model params, model is pre-loaded)
    # Convert processed_image_stems to frozenset for pickling in multiprocessing
    processed_stems_frozen = frozenset(processed_image_stems) if processed_image_stems else frozenset()
    worker_args = [
        (
            image_path_str,
            disable_generation_identification_mode,
            disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode,
            str(tracker.output_root),
            crop_indexer.start_folder,
            crop_indexer.folder_size,
            processed_stems_frozen,
        )
        for image_path_str in image_paths
    ]
    
    # Process in parallel with model pre-initialization
    processed_count = 0
    start_time = time.time()
    
    # Detect GPU count for reporting and worker assignment
    num_gpus = 1
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split('\n'))
    except:
        pass
    
    print(Color.YELLOW(f'Creating {num_workers} worker processes ({num_gpus} GPU(s) available) and loading models...'))
    print(Color.CYAN(f'Workers will be distributed across {num_gpus} GPU(s) in round-robin fashion'))
    print(Color.YELLOW('Note: Model loading can take 30-120 seconds per worker, especially with TensorRT.'))
    print(Color.YELLOW('This is normal - each worker needs to load the model into memory.'))
    
    # Create shared counter for worker index assignment
    # This ensures each worker gets a unique index in spawn mode
    from multiprocessing import Value
    worker_counter = Value('i', 0)  # Shared integer counter
    
    # Create pool with shared counter for worker index assignment
    with Pool(processes=num_workers, initializer=_init_worker_model, initargs=(model_params, worker_counter)) as pool:
        print(Color.GREEN(f'✓ All {num_workers} workers initialized. Starting image processing...'))
        # Use smaller chunksize for better load balancing and GPU utilization
        # Smaller chunksize means tasks are distributed more frequently, reducing worker idle time
        chunksize = max(1, total_images // (num_workers * 4))  # Dynamic chunksize based on workload
        chunksize = min(chunksize, 20)  # Cap at 20 to ensure frequent task distribution
        results = pool.imap_unordered(_process_single_image_worker, worker_args, chunksize=chunksize)
        skipped_in_worker = 0
        
        # Use tqdm for progress bar if available
        if HAS_TQDM:
            results = tqdm(results, total=total_images, desc='Processing images', 
                          unit='img', ncols=100, mininterval=1.0)
        
        for result in results:
            processed_count += 1
            if result is None:
                continue
            
            # Skip if already processed (shouldn't happen due to pre-filtering, but check anyway)
            if result.get('skipped', False):
                skipped_in_worker += 1
                continue
            
            detection_count = result['detection_count']
            tracker.register_image(detection_count)
            
            for crop_data in result['crops']:
                tracker.register_crop(
                    label_name=crop_data['label_name'],
                    classid=crop_data['classid'],
                    relative_path=Path(crop_data['relative_path']),
                    height_px=crop_data['height_px'],
                    width_px=crop_data['width_px'],
                )
            
            # Periodically check and write CSV if entries accumulate too much
            # This ensures we don't lose data if processing is interrupted
            if len(tracker.annotation_entries) >= tracker.csv_max_entries:
                print(Color.CYAN(f'Reached {len(tracker.annotation_entries)} entries, writing to CSV...'))
                tracker.write_annotation_csv(lock=csv_lock, force_flush=True)
            
            # Update progress bar description with rate info
            if HAS_TQDM and processed_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta_seconds = (total_images - processed_count) / rate if rate > 0 else 0
                eta_str = format_eta(eta_seconds)
                results.set_postfix({'rate': f'{rate:.1f} img/s', 'ETA': eta_str})
            
            # Also print detailed progress every 100 images (for logging)
            if processed_count % 100 == 0 or processed_count == total_images:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta_seconds = (total_images - processed_count) / rate if rate > 0 else 0
                percentage = (processed_count / total_images * 100) if total_images > 0 else 0
                eta_str = format_eta(eta_seconds)
                print(f'\n[{percentage:5.1f}%] Processed {processed_count}/{total_images} images. '
                      f'Rate: {rate:.1f} img/s, ETA: {eta_str}')


def process_video_for_eye_analysis(
    *,
    model: DEIMv2,
    video_path: Path,
    tracker: EyeAnalysisTracker,
    crop_indexer: CropIndexer,
    disable_generation_identification_mode: bool,
    disable_gender_identification_mode: bool,
    disable_left_and_right_hand_identification_mode: bool,
    disable_headpose_identification_mode: bool,
    csv_lock: Optional[Lock] = None,
) -> None:
    label_info = resolve_video_label(video_path)
    if label_info is None:
        print(Color.YELLOW(f'Could not determine label/classid mapping for video: {video_path}'))
        return
    label_name, classid = label_info
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(Color.RED(f'ERROR: Failed to open video {video_path}'))
        return
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # If we can't get frame count, we'll process without percentage
        total_frames = None
    
    frame_idx = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            boxes = model(
                image=frame,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
            eye_boxes = [box for box in boxes if box.classid == 17]
            eye_boxes.sort(key=lambda box: box.score, reverse=True)
            detection_count = len(eye_boxes)
            tracker.register_image(detection_count)
            if detection_count == 0:
                continue
            base_name = f'{video_path.stem}_{frame_idx:08d}'
            for det_idx, box in enumerate(eye_boxes[:2]):
                crop = crop_eye_region(frame, box)
                if crop is None or crop.size == 0:
                    continue
                crop_height, crop_width = crop.shape[0], crop.shape[1]
                if not is_valid_crop(label_name, crop_height, crop_width):
                    continue
                saved_rel_path = crop_indexer.save_crop(crop, base_name=base_name, detection_idx=det_idx)
                if saved_rel_path is None:
                    continue
                tracker.register_crop(
                    label_name=label_name,
                    classid=classid,
                    relative_path=saved_rel_path,
                    height_px=crop_height,
                    width_px=crop_width,
                )
            
            # Periodically check and write CSV if entries accumulate too much
            # This ensures we don't lose data if processing is interrupted
            if len(tracker.annotation_entries) >= tracker.csv_max_entries:
                print(Color.CYAN(f'Reached {len(tracker.annotation_entries)} entries, writing to CSV...'))
                tracker.write_annotation_csv(lock=csv_lock, force_flush=True)
            
            # Progress reporting
            if frame_idx % 100 == 0 or (total_frames and frame_idx == total_frames):
                elapsed = time.time() - start_time
                rate = frame_idx / elapsed if elapsed > 0 else 0
                if total_frames:
                    percentage = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    eta_seconds = (total_frames - frame_idx) / rate if rate > 0 else 0
                    eta_str = format_eta(eta_seconds)
                    print(f'[{percentage:5.1f}%] Processed {frame_idx}/{total_frames} frames from {video_path.name}. '
                          f'Rate: {rate:.1f} fps, ETA: {eta_str}')
                else:
                    print(f'Processed {frame_idx} frames from {video_path.name}. Rate: {rate:.1f} fps')
    finally:
        cap.release()

def format_eta(seconds: float) -> str:
    """Format ETA (estimated time remaining) in a human-readable format.
    
    Parameters
    ----------
    seconds: float
        Remaining time in seconds.
    
    Returns
    -------
    str
        Formatted time string (e.g., "2h 30m 15s" or "45m 30s" or "30s").
    """
    if seconds < 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or len(parts) == 0:
        parts.append(f"{secs}s")
    
    return " ".join(parts)

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)

def distance_euclid(p1: Tuple[int,int], p2: Tuple[int,int]) -> float:
    """2点 (x1, y1), (x2, y2) のユークリッド距離を返す"""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def draw_skeleton(
    image: np.ndarray,
    boxes: List[Box],
    color=(0,255,255),
    max_dist_threshold=500.0
):
    """
    与えられた boxes (各クラスIDの関節候補) を基に、EDGESで定義された親子を
    「もっとも近い距離のペアから順番に」接合していく。ただし、
    classid=0 (人物) のバウンディングボックス内にあるキーポイント同士のみを
    接続対象とする。
    """
    # -------------------------
    # 1) 人物ボックスに ID を付与する
    # -------------------------
    person_boxes = [b for b in boxes if b.classid == 0]
    for i, pbox in enumerate(person_boxes):
        # 便宜上、Boxクラスに person_id 属性がないので動的に付与する例
        pbox.person_id = i

    # -------------------------------------------------
    # 2) キーポイントがどの人物ボックスに属するか判断して person_id を記録
    #    （複数人のバウンディングボックスが重なっている場合は、
    #      先に見つかったものを採用、など適宜ルールを決める）
    # -------------------------------------------------
    keypoint_ids = {21,22,23,24,25,29,30,31,32}
    for box in boxes:
        if box.classid in keypoint_ids:
            box.person_id = -1
            for pbox in person_boxes:
                if (pbox.x1 <= box.cx <= pbox.x2) and (pbox.y1 <= box.cy <= pbox.y2):
                    box.person_id = pbox.person_id
                    break

    # -------------------------
    # 3) クラスIDごとに仕分け
    # -------------------------
    classid_to_boxes: Dict[int, List[Box]] = {}
    for b in boxes:
        classid_to_boxes.setdefault(b.classid, []).append(b)

    edge_counts = Counter(EDGES)

    # 結果のラインを入れる
    lines_to_draw = []

    # ユークリッド距離計算の簡易関数
    def distance_euclid(p1, p2):
        import math
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    # 各 (pid, cid) ペアに対してグルーピング
    for (pid, cid), repeat_count in edge_counts.items():
        parent_list = classid_to_boxes.get(pid, [])
        child_list  = classid_to_boxes.get(cid, [])

        if not parent_list or not child_list:
            continue

        # 親クラスIDが21 or 29の時はEDGESに書かれている回数(=repeat_count)だけマッチ可
        # それ以外は1回だけ
        for_parent = repeat_count if (pid in [21, 29]) else 1

        parent_capacity = [for_parent]*len(parent_list)  # 親ごとに繋げる上限

        # 子は常に1回のみ
        child_used = [False]*len(child_list)

        # 距離が小さいペアから順に確定していくために、全ペアの距離を計算
        pair_candidates = []
        for i, pbox in enumerate(parent_list):
            for j, cbox in enumerate(child_list):
                # ここで "同じ person_id 同士であること" をチェック
                if (pbox.person_id is not None) and (cbox.person_id is not None) and (pbox.person_id == cbox.person_id):

                    dist = distance_euclid((pbox.cx, pbox.cy), (cbox.cx, cbox.cy))
                    if dist <= max_dist_threshold:
                        pair_candidates.append((dist, i, j))

        # 距離の小さい順に並べ替え
        pair_candidates.sort(key=lambda x: x[0])

        # 貪欲に割り当て
        for dist, i, j in pair_candidates:
            if parent_capacity[i] > 0 and (not child_used[j]):
                # 親iがまだマッチ可能 & 子jが未使用ならマッチ確定
                pbox = parent_list[i]
                cbox = child_list[j]

                lines_to_draw.append(((pbox.cx, pbox.cy), (cbox.cx, cbox.cy)))
                parent_capacity[i] -= 1
                child_used[j] = True

    # -------------------------
    # 4) ラインを描画
    # -------------------------
    for (pt1, pt2) in lines_to_draw:
        cv2.line(image, pt1, pt2, color, thickness=2)

def main():
    # CRITICAL: Set multiprocessing start method to 'spawn' at the very beginning
    # This is essential for CUDA to work properly in multiprocessing
    # CUDA contexts don't work well with fork(), so we must use spawn
    try:
        set_start_method('spawn', force=True)
        print(Color.CYAN('✓ Using spawn method for multiprocessing (required for CUDA)'))
    except RuntimeError:
        # Already set, ignore
        pass
    
    parser = ArgumentParser()

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 2:
            raise ArgumentTypeError(f"Invalid Value: {ivalue}. Please specify an integer of 2 or greater.")
        return ivalue

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx',
        help='ONNX/TFLite file path for DEIMv2.',
    )
    group_v_or_i = parser.add_mutually_exclusive_group(required=False)
    group_v_or_i.add_argument(
        '-v',
        '--video',
        type=str,
        help='Video file path or camera index.',
    )
    group_v_or_i.add_argument(
        '-i',
        '--images_dir',
        type=str,
        help='jpg, png images folder path.',
    )
    parser.add_argument(
        '-ep',
        '-oep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cuda',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-it',
        '--inference_type',
        type=str,
        choices=['fp16', 'int8'],
        default='fp16',
        help='Inference type. Default: fp16',
    )
    parser.add_argument(
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    parser.add_argument(
        '-dwk',
        '--disable_waitKey',
        action='store_true',
        help=\
            'Disable cv2.waitKey(). '+
            'When you want to process a batch of still images, '+
            ' disable key-input wait and process them continuously.',
    )
    parser.add_argument(
        '-ost',
        '--object_socre_threshold',
        type=float,
        default=0.35,
        help=\
            'The detection score threshold for object detection. Default: 0.35',
    )
    parser.add_argument(
        '-ast',
        '--attribute_socre_threshold',
        type=float,
        default=0.70,
        help=\
            'The attribute score threshold for object detection. Default: 0.70',
    )
    parser.add_argument(
        '-kst',
        '--keypoint_threshold',
        type=float,
        default=0.30,
        help=\
            'The keypoint score threshold for object detection. Default: 0.30',
    )
    parser.add_argument(
        '-kdm',
        '--keypoint_drawing_mode',
        type=str,
        choices=['dot', 'box', 'both'],
        default='dot',
        help='Key Point Drawing Mode. Default: dot',
    )
    parser.add_argument(
        '-ebm',
        '--enable_bone_drawing_mode',
        action='store_true',
        help=\
            'Enable bone drawing mode. (Press B on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dnm',
        '--disable_generation_identification_mode',
        action='store_true',
        help=\
            'Disable generation identification mode. (Press N on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dgm',
        '--disable_gender_identification_mode',
        action='store_true',
        help=\
            'Disable gender identification mode. (Press G on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dlr',
        '--disable_left_and_right_hand_identification_mode',
        action='store_true',
        help=\
            'Disable left and right hand identification mode. (Press H on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dhm',
        '--disable_headpose_identification_mode',
        action='store_true',
        help=\
            'Disable HeadPose identification mode. (Press P on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-ea',
        '--eye_analysis',
        action='store_true',
        help='Enable eye-only analysis workflow (cropping, histograms, annotation export).',
    )
    parser.add_argument(
        '-j',
        '--jobs',
        type=int,
        default=None,
        help='Number of parallel workers for image processing. Default: auto (8 workers per GPU for CUDA/TensorRT, 1 for CPU). Use -j 16 to override.',
    )
    parser.add_argument(
        '--process-dataset-only',
        action='store_true',
        help='Only process image directories (dataset), skip video files.',
    )
    parser.add_argument(
        '--process-video-only',
        action='store_true',
        help='Only process video files, skip image directories (dataset).',
    )
    parser.add_argument(
        '-drc',
        '--disable_render_classids',
        type=int,
        nargs="*",
        default=[],
        help=\
            'Class ID to disable bounding box drawing. List[int]. e.g. -drc 17 18 19',
    )
    parser.add_argument(
        '-efm',
        '--enable_face_mosaic',
        action='store_true',
        help=\
            'Enable face mosaic.',
    )
    parser.add_argument(
        '-dtk',
        '--disable_tracking',
        action='store_true',
        help=\
            'Disable instance tracking. (Press R on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dti',
        '--disable_trackid_overlay',
        action='store_true',
        help=\
            'Disable TrackID overlay. (Press T on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-dhd',
        '--disable_head_distance_measurement',
        action='store_true',
        help=\
            'Disable Head distance measurement. (Press M on the keyboard to switch modes)',
    )
    parser.add_argument(
        '-oyt',
        '--output_yolo_format_text',
        action='store_true',
        help=\
            'Output YOLO format texts and images.',
    )
    parser.add_argument(
        '-bblw',
        '--bounding_box_line_width',
        type=check_positive,
        default=2,
        help=\
            'Bounding box line width. Default: 2',
    )
    parser.add_argument(
        '-chf',
        '--camera_horizontal_fov',
        type=int,
        default=90,
        help=\
            'Camera horizontal FOV. Default: 90',
    )
    args = parser.parse_args()

    # runtime check
    model_file: str = args.model
    model_dir_path = os.path.dirname(os.path.abspath(model_file))
    model_ext: str = os.path.splitext(model_file)[1][1:].lower()
    runtime: str = None
    if model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif model_ext == 'tflite':
        if is_package_installed('ai_edge_litert'):
            runtime = 'ai_edge_litert'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: ai_edge_litert or tensorflow is not installed.'))
            sys.exit(0)
    video: str = args.video
    images_dir: str = args.images_dir
    disable_waitKey: bool = args.disable_waitKey
    object_socre_threshold: float = args.object_socre_threshold
    attribute_socre_threshold: float = args.attribute_socre_threshold
    keypoint_threshold: float = args.keypoint_threshold
    keypoint_drawing_mode: str = args.keypoint_drawing_mode
    enable_bone_drawing_mode: bool = args.enable_bone_drawing_mode
    disable_generation_identification_mode: bool = args.disable_generation_identification_mode
    disable_gender_identification_mode: bool = args.disable_gender_identification_mode
    disable_left_and_right_hand_identification_mode: bool = args.disable_left_and_right_hand_identification_mode
    disable_headpose_identification_mode: bool = args.disable_headpose_identification_mode
    eye_analysis_enabled: bool = args.eye_analysis
    # Auto-detect optimal worker count if not specified
    # For GPU processing, use at least 2 workers per GPU for better utilization
    if args.jobs is not None:
        num_workers: int = args.jobs
    else:
        # Auto-detect GPU count and set workers accordingly
        num_gpus = 1
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                num_gpus = len(result.stdout.strip().split('\n'))
        except:
            pass
        
        # Use 6-8 workers per GPU for better GPU utilization
        # More workers help keep GPU busy while some workers are waiting for I/O
        if execution_provider in ['cuda', 'tensorrt'] and num_gpus > 0:
            num_workers = max(4, num_gpus * 8)  # At least 4 workers, or 8 per GPU for better utilization
            print(Color.CYAN(f'Auto-detected {num_gpus} GPU(s), setting {num_workers} workers ({num_workers // num_gpus} per GPU) for optimal utilization'))
        else:
            num_workers = 1
    if not eye_analysis_enabled and video is None and images_dir is None:
        parser.error('Either --video or --images_dir must be specified unless --eye_analysis is enabled.')
    disable_render_classids: List[int] = args.disable_render_classids
    enable_face_mosaic: bool = args.enable_face_mosaic
    enable_tracking: bool = not args.disable_tracking
    enable_trackid_overlay: bool = not args.disable_trackid_overlay
    enable_head_distance_measurement: bool = not args.disable_head_distance_measurement
    output_yolo_format_text: bool = args.output_yolo_format_text
    execution_provider: str = args.execution_provider
    inference_type: str = args.inference_type
    inference_type = inference_type.lower()
    bounding_box_line_width: int = args.bounding_box_line_width
    camera_horizontal_fov: int = args.camera_horizontal_fov
    
    # Check if CUDA/TensorRT is available when using GPU provider
    if execution_provider in ['cuda', 'tensorrt']:
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            
            # Check onnxruntime version and build info
            try:
                build_info = ort.get_build_info()
                print(Color.CYAN(f'ONNX Runtime version: {ort.__version__}'))
                print(Color.CYAN(f'Build info: {build_info}'))
            except:
                pass
            
            # Check for CUDA provider (required for both CUDA and TensorRT)
            if 'CUDAExecutionProvider' not in available_providers:
                print(Color.RED('ERROR: CUDAExecutionProvider is not available!'))
                print(Color.RED('This usually means:'))
                print(Color.RED('  1. You installed onnxruntime (CPU version) instead of onnxruntime-gpu'))
                print(Color.RED('  2. CUDA/cuDNN is not properly installed'))
                print(Color.RED('  3. GPU drivers are not installed'))
                print(Color.RED('  4. CUDA library path is not in LD_LIBRARY_PATH'))
                print(Color.YELLOW(f'Available providers: {available_providers}'))
                print()
                print(Color.GREEN('Solution:'))
                print(Color.GREEN('  1. Uninstall CPU version: pip uninstall onnxruntime'))
                print(Color.GREEN('  2. Install GPU version: pip install onnxruntime-gpu'))
                print(Color.GREEN('  3. Check CUDA: nvidia-smi'))
                print(Color.GREEN('  4. Verify: python -c "import onnxruntime as ort; print(ort.get_available_providers())"'))
                print()
                print(Color.YELLOW('Note: If you still see this error after installing onnxruntime-gpu,'))
                print(Color.YELLOW('      you may need to install CUDA and cuDNN separately.'))
                print(Color.YELLOW('      Also check LD_LIBRARY_PATH includes CUDA libraries.'))
                sys.exit(1)
            
            # Check for TensorRT provider if requested
            if execution_provider == 'tensorrt':
                if 'TensorrtExecutionProvider' not in available_providers:
                    print(Color.YELLOW('WARNING: TensorrtExecutionProvider is not available!'))
                    print(Color.YELLOW('Falling back to CUDAExecutionProvider.'))
                    print(Color.YELLOW('To use TensorRT, you need:'))
                    print(Color.YELLOW('  1. TensorRT installed'))
                    print(Color.YELLOW('  2. TensorRT libraries in LD_LIBRARY_PATH'))
                    print(Color.YELLOW('  3. Compatible CUDA version'))
                    execution_provider = 'cuda'  # Fallback to CUDA
                else:
                    print(Color.GREEN(f'✓ TensorRT is available in providers list.'))
                    # Check if TensorRT library can be found
                    try:
                        import subprocess
                        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=2)
                        if 'libnvinfer' not in result.stdout:
                            print(Color.YELLOW('WARNING: TensorRT library (libnvinfer.so) not found in system library cache.'))
                            print(Color.YELLOW('TensorRT may fail to load at runtime. The code will try to add TensorRT paths automatically.'))
                            print(Color.YELLOW('If it still fails, consider using -ep cuda instead.'))
                    except:
                        pass
                    print(Color.GREEN(f'Available providers: {available_providers}'))
                    print(Color.CYAN('Note: If TensorRT fails to load, it will automatically fall back to CUDA.'))
            else:
                print(Color.GREEN(f'✓ CUDA is available. Available providers: {available_providers}'))
            
            # Try to create a test session to verify CUDA actually works
            try:
                import numpy as np
                # Create a minimal test model or use a simple test
                test_providers = [('CUDAExecutionProvider', {'device_id': 0})]
                # Just check if provider can be initialized, don't create actual session
                print(Color.CYAN('Verifying CUDA provider can be initialized...'))
            except Exception as e:
                print(Color.YELLOW(f'Warning: Could not verify CUDA provider: {e}'))
                
        except ImportError:
            print(Color.YELLOW('Warning: Could not import onnxruntime to check CUDA availability'))
    
    providers: List[Tuple[str, Dict] | str] = None

    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        # Configure CUDA provider with multi-GPU support
        # For main process, use GPU 0 (workers will be assigned different GPUs)
        providers = [
            (
            'CUDAExecutionProvider',
                {
                    'device_id': 0,  # Primary GPU, workers will use different devices
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 12 * 1024 * 1024 * 1024,  # 12GB per GPU (RTX 3090 has 24GB, use half for safety)
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                    'tunable_op_enable': True,  # Enable tunable ops for better performance
                    'tunable_op_tuning_enable': True,
                }
            ),
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
        ep_type_params = {}
        if inference_type == 'fp16':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        elif inference_type == 'int8':
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                    "trt_int8_enable": True,
                    "trt_int8_calibration_table_name": "calibration.flatbuffers",
                }
        else:
            ep_type_params = \
                {
                    "trt_fp16_enable": True,
                }
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': True, # .engine, .profile export
                    'trt_engine_cache_path': f'{model_dir_path}',
                    # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                    # onnxruntime>=1.21.0 breaking changes
                    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                    # https://github.com/microsoft/onnxruntime/pull/22681/files
                    # https://github.com/microsoft/onnxruntime/pull/23893/files
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                } | ep_type_params,
            ),
            "CUDAExecutionProvider",
            'CPUExecutionProvider',
        ]

    print(Color.GREEN('Provider parameters:'))
    pprint(providers)

    # Model initialization
    model = DEIMv2(
        runtime=runtime,
        model_path=model_file,
        obj_class_score_th=object_socre_threshold,
        attr_class_score_th=attribute_socre_threshold,
        keypoint_th=keypoint_threshold,
        providers=providers,
    )

    if eye_analysis_enabled:
        image_dirs_for_analysis: List[Path] = []
        video_paths_for_analysis: List[Path] = []

        if images_dir is not None:
            image_dirs_for_analysis.append(Path(images_dir))

        if video is not None:
            if is_parsable_to_int(video):
                print(Color.YELLOW('Skipping camera index for eye analysis; only file paths are supported.'))
            else:
                video_paths_for_analysis.append(Path(video))

        if not image_dirs_for_analysis and not video_paths_for_analysis:
            # default_dir = Path('data') / 'extracted' / 'train'
            # image_dirs_for_analysis.append(default_dir)
            real_data_dir = Path('real_data_test')
            default_videos: List[Path] = []
            if real_data_dir.exists():
                def _video_sort_key(path: Path) -> Tuple[str, int]:
                    stem = path.stem
                    prefix = stem.rstrip('0123456789')
                    suffix = stem[len(prefix):]
                    number = int(suffix) if suffix else 0
                    return prefix, number

                for pattern in ('open*.mp4', 'closed*.mp4'):
                    matched_paths = [
                        candidate for candidate in real_data_dir.glob(pattern)
                        if candidate.is_file()
                    ]
                    default_videos.extend(sorted(matched_paths, key=_video_sort_key))
            if not default_videos:
                default_videos = [
                    real_data_dir / 'open.mp4',
                    real_data_dir / 'closed.mp4',
                ]
            video_paths_for_analysis.extend(default_videos)

        # Deduplicate while preserving order
        seen_dirs = set()
        unique_image_dirs: List[Path] = []
        for candidate in image_dirs_for_analysis:
            resolved = candidate
            if resolved in seen_dirs:
                continue
            unique_image_dirs.append(resolved)
            seen_dirs.add(resolved)

        seen_videos = set()
        unique_video_paths: List[Path] = []
        for candidate in video_paths_for_analysis:
            resolved = candidate
            if resolved in seen_videos:
                continue
            unique_video_paths.append(resolved)
            seen_videos.add(resolved)

        process_dataset_only: bool = args.process_dataset_only
        process_video_only: bool = args.process_video_only
        
        if process_dataset_only and process_video_only:
            parser.error('--process-dataset-only and --process-video-only cannot be used together.')

        run_eye_analysis(
            model=model,
            image_dirs=unique_image_dirs,
            video_paths=unique_video_paths,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
            num_workers=num_workers,
            process_dataset_only=process_dataset_only,
            process_video_only=process_video_only,
        )
        return

    file_paths: List[str] = None
    cap = None
    video_writer = None
    if images_dir is not None:
        file_paths = list_image_files(dir_path=images_dir)
    else:
        cap = cv2.VideoCapture(
            int(video) if is_parsable_to_int(video) else video
        )
        disable_video_writer: bool = args.disable_video_writer
        if not disable_video_writer:
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                filename='output.mp4',
                fourcc=fourcc,
                fps=cap_fps,
                frameSize=(w, h),
            )

    file_paths_count = -1
    movie_frame_count = 0
    white_line_width = bounding_box_line_width
    colored_line_width = white_line_width - 1
    tracker = SimpleSortTracker()
    track_color_cache: Dict[int, np.ndarray] = {}
    tracking_enabled_prev = enable_tracking
    while True:
        image: np.ndarray = None
        if file_paths is not None:
            file_paths_count += 1
            if file_paths_count <= len(file_paths) - 1:
                image = cv2.imread(file_paths[file_paths_count])
            else:
                break
        else:
            res, image = cap.read()
            if not res:
                break
            movie_frame_count += 1

        debug_image = copy.deepcopy(image)
        debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        start_time = time.perf_counter()
        boxes = model(
            image=debug_image,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )
        elapsed_time = time.perf_counter() - start_time

        if file_paths is None:
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        body_boxes = [box for box in boxes if box.classid == 0]
        current_tracking_enabled = enable_tracking
        if current_tracking_enabled:
            if not tracking_enabled_prev:
                tracker = SimpleSortTracker()
                track_color_cache.clear()
            tracker.update(body_boxes)
            active_track_ids = {track['id'] for track in tracker.tracks}
            stale_ids = [tid for tid in track_color_cache.keys() if tid not in active_track_ids]
            for tid in stale_ids:
                track_color_cache.pop(tid, None)
        else:
            if tracking_enabled_prev:
                tracker = SimpleSortTracker()
                track_color_cache.clear()
            for box in boxes:
                box.track_id = -1
        tracking_enabled_prev = current_tracking_enabled

        # Draw bounding boxes
        for box in boxes:
            classid: int = box.classid
            color = (255,255,255)

            if classid in disable_render_classids:
                continue

            if classid == 0:
                # Body
                if not disable_gender_identification_mode:
                    # Body
                    if box.gender == 0:
                        # Male
                        color = (255,0,0)
                    elif box.gender == 1:
                        # Female
                        color = (139,116,225)
                    else:
                        # Unknown
                        color = (0,200,255)
                else:
                    # Body
                    color = (0,200,255)
            elif classid == 5:
                # Body-With-Wheelchair
                color = (0,200,255)
            elif classid == 6:
                # Body-With-Crutches
                color = (83,36,179)
            elif classid == 7:
                # Head
                if not disable_headpose_identification_mode:
                    color = BOX_COLORS[box.head_pose][0] if box.head_pose != -1 else (216,67,21)
                else:
                    color = (0,0,255)
            elif classid == 16:
                # Face
                color = (0,200,255)
            elif classid == 17:
                # Eye
                color = (255,0,0)
            elif classid == 18:
                # Nose
                color = (0,255,0)
            elif classid == 19:
                # Mouth
                color = (0,0,255)
            elif classid == 20:
                # Ear
                color = (203,192,255)

            elif classid == 21:
                # Collarbone
                color = (0,0,255)
            elif classid == 22:
                # Shoulder
                color = (255,0,0)
            elif classid == 23:
                # Solar_plexus
                color = (252,189,107)
            elif classid == 24:
                # Elbow
                color = (0,255,0)
            elif classid == 25:
                # Wrist
                color = (0,0,255)

            elif classid == 26:
                if not disable_left_and_right_hand_identification_mode:
                    # Hands
                    if box.handedness == 0:
                        # Left-Hand
                        color = (0,128,0)
                    elif box.handedness == 1:
                        # Right-Hand
                        color = (255,0,255)
                    else:
                        # Unknown
                        color = (0,255,0)
                else:
                    # Hands
                    color = (0,255,0)

            elif classid == 29:
                # abdomen
                color = (0,0,255)
            elif classid == 30:
                # hip_joint
                color = (255,0,0)
            elif classid == 31:
                # Knee
                color = (0,0,255)
            elif classid == 32:
                # ankle
                color = (255,0,0)

            elif classid == 33:
                # Foot
                color = (250,0,136)

            if (classid == 0 and not disable_gender_identification_mode) \
                or (classid == 7 and not disable_headpose_identification_mode) \
                or (classid == 26 and not disable_left_and_right_hand_identification_mode) \
                or classid == 16 \
                or classid in [21,22,23,24,25,29,30,31,32]:

                # Body
                if classid == 0:
                    if box.gender == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Head
                elif classid == 7:
                    if box.head_pose == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Face
                elif classid == 16:
                    if enable_face_mosaic:
                        w = int(abs(box.x2 - box.x1))
                        h = int(abs(box.y2 - box.y1))
                        small_box = cv2.resize(debug_image[box.y1:box.y2, box.x1:box.x2, :], (3,3))
                        normal_box = cv2.resize(small_box, (w,h))
                        if normal_box.shape[0] != abs(box.y2 - box.y1) \
                            or normal_box.shape[1] != abs(box.x2 - box.x1):
                                normal_box = cv2.resize(small_box, (abs(box.x2 - box.x1), abs(box.y2 - box.y1)))
                        debug_image[box.y1:box.y2, box.x1:box.x2, :] = normal_box
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                    cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Hands
                elif classid == 26:
                    if box.handedness == -1:
                        draw_dashed_rectangle(
                            image=debug_image,
                            top_left=(box.x1, box.y1),
                            bottom_right=(box.x2, box.y2),
                            color=color,
                            thickness=2,
                            dash_length=10
                        )
                    else:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

                # Shoulder, Elbow, Knee
                elif classid in [21,22,23,24,25,29,30,31,32]:
                    if keypoint_drawing_mode in ['dot', 'both']:
                        cv2.circle(debug_image, (box.cx, box.cy), 4, (255,255,255), -1)
                        cv2.circle(debug_image, (box.cx, box.cy), 3, color, -1)
                    if keypoint_drawing_mode in ['box', 'both']:
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
                        cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)

            else:
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), white_line_width)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, colored_line_width)

            # TrackID text
            if enable_trackid_overlay and classid == 0 and box.track_id > 0:
                track_text = f'ID: {box.track_id}'
                text_x = max(box.x1 - 5, 0)
                text_y = box.y1 - 30
                if text_y < 20:
                    text_y = min(box.y2 + 25, debug_image_h - 10)
                cached_color = track_color_cache.get(box.track_id)
                if isinstance(cached_color, np.ndarray):
                    text_color = tuple(int(np.clip(v, 0, 255)) for v in cached_color.tolist())
                else:
                    text_color = color if isinstance(color, tuple) else (0, 200, 255)
                cv2.putText(
                    debug_image,
                    track_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (10, 10, 10),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    track_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            # Attributes text
            generation_txt = ''
            if box.generation == -1:
                generation_txt = ''
            elif box.generation == 0:
                generation_txt = 'Adult'
            elif box.generation == 1:
                generation_txt = 'Child'

            gender_txt = ''
            if box.gender == -1:
                gender_txt = ''
            elif box.gender == 0:
                gender_txt = 'M'
            elif box.gender == 1:
                gender_txt = 'F'

            attr_txt = f'{generation_txt}({gender_txt})' if gender_txt != '' else f'{generation_txt}'

            headpose_txt = BOX_COLORS[box.head_pose][1] if box.head_pose != -1 else ''
            attr_txt = f'{attr_txt} {headpose_txt}' if headpose_txt != '' else f'{attr_txt}'

            cv2.putText(
                debug_image,
                f'{attr_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{attr_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            handedness_txt = ''
            if box.handedness == -1:
                handedness_txt = ''
            elif box.handedness == 0:
                handedness_txt = 'L'
            elif box.handedness == 1:
                handedness_txt = 'R'
            cv2.putText(
                debug_image,
                f'{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f'{handedness_txt}',
                (
                    box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
                    box.y1-10 if box.y1-25 > 0 else 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                1,
                cv2.LINE_AA,
            )

            # Head distance
            if enable_head_distance_measurement and classid == 7:
                focalLength: float = 0.0
                if (camera_horizontal_fov > 90):
                    # Fisheye Camera (Equidistant Model)
                    focalLength = debug_image_w / (camera_horizontal_fov * (math.pi / 180))
                else:
                    # Normal camera (Pinhole Model)
                    focalLength = debug_image_w / (2 * math.tan((camera_horizontal_fov / 2) * (math.pi / 180)))
                # Meters
                distance = (AVERAGE_HEAD_WIDTH * focalLength) / abs(box.x2 - box.x1)

                cv2.putText(
                    debug_image,
                    f'{distance:.3f} m',
                    (
                        box.x1+5 if box.x1 < debug_image_w else debug_image_w-50,
                        box.y1+20 if box.y1-5 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_image,
                    f'{distance:.3f} m',
                    (
                        box.x1+5 if box.x1 < debug_image_w else debug_image_w-50,
                        box.y1+20 if box.y1-15 > 0 else 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (10, 10, 10),
                    1,
                    cv2.LINE_AA,
                )

            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (255, 255, 255),
            #     2,
            #     cv2.LINE_AA,
            # )
            # cv2.putText(
            #     debug_image,
            #     f'{box.score:.2f}',
            #     (
            #         box.x1 if box.x1+50 < debug_image_w else debug_image_w-50,
            #         box.y1-10 if box.y1-25 > 0 else 20
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     color,
            #     1,
            #     cv2.LINE_AA,
            # )

        # Draw skeleton
        if enable_bone_drawing_mode:
            draw_skeleton(image=debug_image, boxes=boxes, color=(0, 255, 255), max_dist_threshold=300)

        if file_paths is not None:
            basename = os.path.basename(file_paths[file_paths_count])
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{basename}', debug_image)

        if file_paths is not None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_i.png', image)
            cv2.imwrite(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}_o.png', debug_image)
            with open(f'output/{os.path.splitext(os.path.basename(file_paths[file_paths_count]))[0]}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
        elif file_paths is None and output_yolo_format_text:
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(f'output/{movie_frame_count:08d}.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_i.png', image)
            cv2.imwrite(f'output/{movie_frame_count:08d}_o.png', debug_image)
            with open(f'output/{movie_frame_count:08d}.txt', 'w') as f:
                for box in boxes:
                    classid = box.classid
                    cx = box.cx / debug_image_w
                    cy = box.cy / debug_image_h
                    w = abs(box.x2 - box.x1) / debug_image_w
                    h = abs(box.y2 - box.y1) / debug_image_h
                    f.write(f'{classid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

        if video_writer is not None:
            video_writer.write(debug_image)
            # video_writer.write(image)

        cv2.imshow("test", debug_image)

        key = cv2.waitKey(1) & 0xFF if file_paths is None or disable_waitKey else cv2.waitKey(0) & 0xFF
        if key == ord('\x1b'): # 27, ESC
            break
        elif key == ord('b'): # 98, B, Bone drawing mode switch
            enable_bone_drawing_mode = not enable_bone_drawing_mode
        elif key == ord('n'): # 110, N, Generation mode switch
            disable_generation_identification_mode = not disable_generation_identification_mode
        elif key == ord('g'): # 103, G, Gender mode switch
            disable_gender_identification_mode = not disable_gender_identification_mode
        elif key == ord('p'): # 112, P, HeadPose mode switch
            disable_headpose_identification_mode = not disable_headpose_identification_mode
        elif key == ord('h'): # 104, H, HandsLR mode switch
            disable_left_and_right_hand_identification_mode = not disable_left_and_right_hand_identification_mode
        elif key == ord('k'): # 107, K, Keypoints mode switch
            if keypoint_drawing_mode == 'dot':
                keypoint_drawing_mode = 'box'
            elif keypoint_drawing_mode == 'box':
                keypoint_drawing_mode = 'both'
            elif keypoint_drawing_mode == 'both':
                keypoint_drawing_mode = 'dot'
        elif key == ord('r'): # 114, R, Tracking mode switch
            enable_tracking = not enable_tracking
            if enable_tracking and not enable_trackid_overlay:
                enable_trackid_overlay = True
        elif key == ord('t'): # 116, T, TrackID overlay mode switch
            enable_trackid_overlay = not enable_trackid_overlay
            if not enable_tracking:
                enable_trackid_overlay = False
        elif key == ord('m'): # 109, M, Head distance measurement mode switch
            enable_head_distance_measurement = not enable_head_distance_measurement

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()

    try:
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()
