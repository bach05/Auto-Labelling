import logging
import os
import json
from datetime import datetime
from typing import Callable, List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from torchvision import transforms as T


class CoupledCameraDataset(Dataset):
    """Dataset that yields synchronized images from multiple camera groups in a coupling JSON.

    Args:
      coupling_json: path to the coupling JSON produced by DataCoupler/save_matches
      groups: list of group ids (ints) to couple, or None to use all groups in the JSON
      cameras: list of camera types to filter, or None to use all available cameras
      transform: optional callable applied to PIL.Image -> tensor
      score_thres: float, if >0 drop pairs whose sim_score (when available) < score_thres
      date_from/date_to: optional ISO strings or datetime objects to filter pairs in a time window
      require_both_in_slot: if True both images must fall inside the window, otherwise at least one

    return:
        Each sample is a dict with keys for each group containing the transformed images, and a 'meta' dict with sim_score and other info.
        {
            'basler': [img_g1_basler_1, img_g2_basler_2],
            'ob_color': [img_g1_ob_color_1, img_g2_ob_color_2],
            ...
        }
    """

    def __init__(
            self,
            coupling_json: str,
            groups: Optional[List[int]] = None,
            cameras: Optional[List[str]] = None,
            transform: Optional[Callable] = None,
            score_thres: float = 0.0,
            date_from: Optional[Any] = None,
            date_to: Optional[Any] = None,
            require_both_in_slot: bool = True,
            root_override: Optional[str] = None,
            enable_pbar: bool = False,
    ):
        self.coupling_json = coupling_json
        self.groups = groups
        self.cameras = cameras
        self.transform = transform
        self.score_thres = float(score_thres or 0.0)
        self.require_both_in_slot = bool(require_both_in_slot)
        self.root_override = root_override

        # parse date_from/date_to to datetimes if provided
        self.date_from = self._parse_date(date_from)
        self.date_to = self._parse_date(date_to)

        # default transform if none provided
        if self.transform is None:
            self.transform = T.Compose([T.ToTensor()])

        # load coupling JSON
        with open(self.coupling_json, 'r') as f:
            payload = json.load(f)

        # Extract common root and data
        self.common_root = payload.get('common_root_path') if isinstance(payload, dict) else None
        if root_override:
            self.common_root = root_override

        raw_data = payload.get('data') if isinstance(payload, dict) else payload
        if raw_data is None:
            raise ValueError('No data field found in coupling JSON')

        # Handle groups
        self.groups = self.groups or sorted(list(payload.get('camera_group_description', {}).keys()))

        # Prepare the samples list with filtered data
        self.samples: List[Dict[str, Any]] = []
        for item in tqdm(raw_data, desc="Processing coupled data", disable=not enable_pbar):
            self._process_item(item)

    def _parse_date(self, v):
        """Helper to convert date string or datetime to datetime object."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except Exception:
                pass
        return None

    def _process_item(self, item: Dict[str, Any]):
        """Process each item to apply filters and store valid pairs."""
        # iterate over camera groups in each data entry
        group_data = {str(group): item.get(str(group)) for group in self.groups}
        if not any(group_data.values()):
            return

        sim_score = self._get_similarity_score(group_data)
        if self.score_thres > 0 and sim_score < self.score_thres:
            return

        # Date filtering
        if not self._check_dates(group_data):
            return

        # Build image paths for selected cameras and groups
        selected_data = {}
        for group, data in group_data.items():
            if data:
                selected_data[group] = {}
                for camera in self.cameras or data.get('available_cameras', []):
                    cam_path = data.get(camera)
                    if cam_path:
                        abs_path = os.path.join(self.common_root or '.', cam_path)
                        if os.path.exists(abs_path):
                            selected_data[group][camera] = abs_path

        # Only add the sample if valid cameras are found
        if selected_data:
            # create the output structure { camera: [img_g1_cam, img_g2_cam, ...], ... }
            out_data = {}
            for group, cameras in selected_data.items():
                for camera, img_path in cameras.items():
                    out_data.setdefault(camera, []).append(img_path)
            self.samples.append({'selected_data': out_data, 'sim_score': sim_score})

    def _get_similarity_score(self, group_data: Dict[str, Any]) -> float:
        """Retrieve similarity score from available data."""
        sim_score = None
        for group in group_data.values():
            if group and 'sim_score' in group:
                if group['sim_score'] is None:
                        continue
                sim_score = max(sim_score or -float('inf'), group['sim_score'])
        return sim_score or 0.0

    def _check_dates(self, group_data: Dict[str, Any]) -> bool:
        """Check if the samples fall within the given date range."""

        def in_date_range(dt):
            return (self.date_from is None or dt >= self.date_from) and (self.date_to is None or dt <= self.date_to)

        for group in group_data.values():
            if group and 'datetime' in group:
                dt = datetime.strptime(group['datetime'], "%Y-%m-%d %H:%M:%S")
                if not in_date_range(dt):
                    return False
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sample_data = sample['selected_data']

        # Load images and apply transformations
        transformed_data = {}
        for camera, img_list in sample_data.items():
            transformed_data[camera] = []
            for img_path in img_list:
                img = Image.open(img_path).convert('RGB')
                transformed_data[camera].append(self.transform(img))

        meta = {'sim_score': sample['sim_score']}
        return transformed_data, meta