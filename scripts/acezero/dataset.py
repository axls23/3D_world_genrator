import logging
import math
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage import color
from skimage import io
from skimage.transform import rotate, resize
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image

from ace_network import Regressor
import dataset_io

_logger = logging.getLogger(__name__)


class CamLocDataset(Dataset):
    """ACE dataset.

    Access to images, calibration and poses. Optionally, ground truth scene coordinates from depth.
    """

    def __init__(self,
                 rgb_files,
                 pose_files=None,
                 ace_pose_file=None,
                 ace_pose_file_conf_threshold=None,
                 pose_seed=-1,
                 depth_files=None,
                 calibration_files=None,
                 calibration_file_f_idx=0,
                 use_depth=False,
                 return_eye_coords=False,
                 augment=False,
                 aug_rotation=15,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_black_white=0.1,
                 aug_color=0.3,
                 image_short_size=480,
                 use_half=True,
                 use_heuristic_focal_length=False
                 ):
        """Constructor.

        Parameters:
            rgb_files: Glob pattern that matches rgb files.
            pose_files: Glob pattern that matches pose files associated with rgb files.
            ace_pose_file: Path to the ACE pose file that contains RGB file paths and pose, focal lengths and confidences.
            ace_pose_file_conf_threshold: Confidence threshold for ACE pose file. Ignore images below confidence.
            pose_seed: If set, only use a single image from the dataset. Float in [0, 1] that determines the image relative to the dataset size.
            depth_files: Glob pattern that matches depth files associated with rgb files.
            calibration_files: Glob pattern that matches calibration files associated with rgb files. Used to populate focal_lengths list.
            calibration_file_f_idx: Index of the item in each calibration file to be interpreted as focal length.
            use_depth: Use depth to generate target coordinates. Either from depth files or ZoeDepth.
            return_eye_coords: If True, target coordinates returned are eye (camera) coordinates, otherwise scene coordinates.
            augment: Use random data augmentation, note: not supported for mode = 2 (RGB-D) since pre-generated eye
                coordinates cannot be augmented
            aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions, degrees.
            aug_scale_min: Lower limit of image scale factor for uniform sampling
            aug_scale_min: Upper limit of image scale factor for uniform sampling
            aug_black_white: Max relative scale factor for image brightness/contrast sampling, e.g. 0.1 -> [0.9,1.1]
            aug_color: Max relative scale factor for image saturation/hue sampling, e.g. 0.1 -> [0.9,1.1]
            image_short_size: RGB images are rescaled such that the short side has this length (if augmentation is disabled, and in the range
                [aug_scale_min * image_short_size, aug_scale_max * image_short_size] otherwise).
            use_half: Enabled if training with half-precision floats.
            use_heuristic_focal_length: Use a heuristic focal length derived from the image dimensions if no focal length is provided.
        """

        self.use_half = use_half
        self.use_depth = use_depth
        self.return_eye_coords = return_eye_coords

        self.image_short_size = image_short_size

        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_black_white = aug_black_white
        self.aug_color = aug_color

        self.use_heuristic_focal_length = use_heuristic_focal_length
        # an external focal length can be provided using a setter function to overwrite the focal length
        self.external_focal_length = None

        if use_heuristic_focal_length:
            _logger.info(f"Overwriting focal length with heuristic derived from image dimensions.")

        # Loading dataset depending on what arguments are provided.
        if ace_pose_file is not None:
            _logger.info(f"Loading dataset from pose file: {ace_pose_file}")
            dataset_info = dataset_io.load_dataset_ace(
                pose_file=ace_pose_file, confidence_threshold=ace_pose_file_conf_threshold)

            self.rgb_files, self.poses, self.focal_lengths, files_kept = dataset_info
        else:
            _logger.info(f"Loading RGB files from: {rgb_files}")
            self.rgb_files = dataset_io.get_files_from_glob(rgb_files)
            self.poses = dataset_io.load_pose_files(pose_files) if pose_files is not None else []

            # Remove invalid poses and corresponding RGB files.
            self.rgb_files, self.poses, files_kept = dataset_io.remove_invalid_poses(self.rgb_files, self.poses)

            # Focal length can be set via an extra function call, or heuristic will be used
            self.focal_lengths = []

        # Load depth files if available.
        if depth_files is not None:
            self.depth_files = dataset_io.get_files_from_glob(depth_files)
            # check if files have been removed and remove depth files as well to keep them in sync
            self.depth_files = [df for df, keep in zip(self.depth_files, files_kept) if keep]
            # final check to make sure depth and rgb files are in sync
            assert len(self.depth_files) == len(self.rgb_files)
        else:
            self.depth_files = []

        # Load calibration files if available and extract focal lengths.
        if calibration_files is not None:
            calibration_files_list = dataset_io.get_files_from_glob(calibration_files)
            # check if files have been removed and remove calibration files as well to keep them in sync
            calibration_files_list = [cf for cf, keep in zip(calibration_files_list, files_kept) if keep]
            # final check to make sure calibration and rgb files are in sync
            assert len(calibration_files_list) == len(self.rgb_files)
            
            # Extract focal lengths from calibration files
            for calibration_file in calibration_files_list:
                with open(calibration_file, 'r') as f:
                    content = f.read()
                    # Split by spaces and newlines to get all items
                    items = content.split()
                    # Extract the nth item as focal length
                    if calibration_file_f_idx < len(items):
                        focal_length = float(items[calibration_file_f_idx])
                        self.focal_lengths.append(focal_length)
                    else:
                        raise ValueError(f"Calibration file {calibration_file} has only {len(items)} items, "
                                       f"but index {calibration_file_f_idx} was requested.")

        # Reduce dataset to single image if pose_seed is set.
        if pose_seed > -1:
            seed_index = int(pose_seed * len(self.rgb_files))

            _logger.info(f"Overwriting dataset with single image: {seed_index} - {self.rgb_files[seed_index]}")

            self.rgb_files = [self.rgb_files[seed_index]]
            self.poses = [torch.eye(4, 4)]

            if len(self.focal_lengths) > 0:
                self.focal_lengths = [self.focal_lengths[seed_index]]

            if len(self.depth_files) > 0:
                self.depth_files = [self.depth_files[seed_index]]
            else:
                # Check for cached depth file first (VRAM optimization)
                from pathlib import Path
                rgb_path = Path(self.rgb_files[0])  # Use [0] since list was already reduced to 1 element
                # Look in depths/ folder (separate from images/)
                depths_dir = rgb_path.parent.parent / 'depths'
                cached_depth = depths_dir / f"{rgb_path.stem}.npy"
                
                # Fallback: legacy location (in images folder)
                if not cached_depth.exists():
                    cached_depth = rgb_path.parent / f"{rgb_path.stem}.depth.npy"
                
                if cached_depth.exists():
                    _logger.info(f"Using cached depth: {cached_depth}")
                    self.depth_files = [str(cached_depth)]
                    self.depth_model = None  # No need to load ZoeDepth
                else:
                    # No cache, estimate depth with ZoeDepth
                    _logger.info(f"Using ZoeDepth for depth initialization.")
                    self.depth_model = dataset_io.get_depth_model()

        # If no poses are provided (e.g. during the reloc stage) fill up with dummy identity poses.
        if len(self.poses) == 0:
            _logger.info(f"No poses provided. Dataset will return identity poses.")
            self.poses = [torch.eye(4, 4)] * len(self.rgb_files)

        # At this stage, number of poses and number of images should match
        if len(self.poses) != len(self.rgb_files):
            raise ValueError(f"Number of poses ({len(self.poses)}) does not match number of images ({len(self.rgb_files)}).")

        # Create grid of 2D pixel positions used when generating scene coordinates from depth.
        if self.use_depth:
            self.prediction_grid = self._create_prediction_grid()
        else:
            self.prediction_grid = None

        # Image transformations. Excluding scale since that can vary batch-by-batch.
        if self.augment:
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=self.aug_black_white, contrast=self.aug_black_white),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                    std=[0.25]
                ),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                    std=[0.25]
                ),
            ])

        # We use this to iterate over all frames.
        self.valid_file_indices = np.arange(len(self.rgb_files))

        # Calculate mean camera center (using the valid frames only).
        self.mean_cam_center = self._compute_mean_camera_center()

    def set_external_focal_length(self, focal_length):
        self.external_focal_length = focal_length

    @staticmethod
    def _create_prediction_grid():
        # Assumes all input images have a resolution smaller than 5000x5000.
        prediction_grid = np.zeros((2,
                                    math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE),
                                    math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE)))

        for x in range(0, prediction_grid.shape[2]):
            for y in range(0, prediction_grid.shape[1]):
                prediction_grid[0, y, x] = x * Regressor.OUTPUT_SUBSAMPLE
                prediction_grid[1, y, x] = y * Regressor.OUTPUT_SUBSAMPLE

        return prediction_grid

    @staticmethod
    def _resize_image(image, short_size):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        # Will resize such that shortest side has short_size length in px.
        image = TF.resize(image, short_size)
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _compute_mean_camera_center(self):
        mean_cam_center = torch.zeros((3,))
        invalid_poses = 0

        for idx in self.valid_file_indices:
            pose = self.poses[idx].clone()

            if torch.any(torch.isnan(pose)) or torch.any(torch.isinf(pose)):
                invalid_poses += 1
                continue

            # Get the translation component.
            mean_cam_center += pose[0:3, 3]

        if invalid_poses > 0:
            _logger.warning(f"Ignored {invalid_poses} poses from mean computation.")

        # Avg.
        mean_cam_center /= (len(self) - invalid_poses)
        return mean_cam_center

    def _load_image(self, idx):
        image = io.imread(self.rgb_files[idx])

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image

    def get_image_size(self, idx):
        """
        This method is used to get the size of the image at the given index.
        Opens image in lazy mode to get the size without loading the whole image.

        Parameters:
        idx (int): The index of the image for which the size is to be obtained.

        Returns:
        tuple: The size of the image at the given index.
        """

        with Image.open(self.rgb_files[idx]) as img:
            return img.size

    def get_focal_length(self, idx):
        """
        This method is used to get the focal length of the camera used to capture the image at the given index.
        The focal length can be obtained in four ways (in order of priority):
        1. If an external focal length is set, it is used.
        2. If focal lengths are available from calibration files or pose files, they are used.
        3. If the heuristic focal length is enabled, it is calculated based on the image dimensions.
        4. Otherwise, error (should not happen with proper validation).

        Parameters:
        idx (int): The index of the image for which the focal length is to be obtained.

        Returns:
        float: The focal length of the camera used to capture the image.
        """

        if self.external_focal_length is not None:
            # use external focal length if set
            return self.external_focal_length
        elif len(self.focal_lengths) > 0:
            # use focal lengths from calibration files or pose files if available
            return self.focal_lengths[idx]
        elif self.use_heuristic_focal_length:
            # use heuristic focal length derived from image dimensions
            width, height = self.get_image_size(idx)

            # we use 70% of the diagonal as focal length
            return math.sqrt(width ** 2 + height ** 2) * 0.7
        else:
            raise ValueError("No focal length source available. This should not happen with proper validation.")

    def _get_single_item(self, idx, image_short_size):
        # Apply index indirection.
        idx = self.valid_file_indices[idx]

        # Load image.
        image = self._load_image(idx)

        # Load intrinsics.
        focal_length = self.get_focal_length(idx)

        # The image will be scaled to image_height, adjust focal length as well.
        f_scale_factor = image_short_size / min(image.shape[0], image.shape[1])
        focal_length *= f_scale_factor

        # Rescale image.
        image = self._resize_image(image, image_short_size)

        # Create mask of the same size as the resized image (it's a PIL image at this point).
        image_mask = torch.ones((1, image.size[1], image.size[0]))

        # Load ground truth scene coordinates, if needed.
        if self.use_depth:
            if len(self.depth_files) > 0:
                depth_file = self.depth_files[idx]
                # Check if it's a cached .npy file or original depth file
                if depth_file.endswith('.npy'):
                    # Load pre-computed depth from NPY cache
                    depth = np.load(depth_file)
                else:
                    # Read original depth map from disk
                    depth = io.imread(depth_file)
                    depth = depth.astype(np.float64)
                    depth /= 1000  # from millimeters to meters
            elif hasattr(self, 'depth_model') and self.depth_model is not None:
                # Estimate depth using ZoeDepth
                depth = dataset_io.estimate_depth(self.depth_model, image)
            else:
                # No depth source available
                raise ValueError(f"No depth source for image at index {idx}. Run precompute_depths.py first.")
        else:
            # set coords to all zeros as a default, training loop will catch this case
            coords = torch.zeros((
                3,
                math.ceil(image.size[0] / Regressor.OUTPUT_SUBSAMPLE),
                math.ceil(image.size[1] / Regressor.OUTPUT_SUBSAMPLE)))

        # Apply remaining transforms.
        image = self.image_transform(image)

        # Get pose.
        pose = self.poses[idx].clone()

        # Apply data augmentation if necessary.
        if self.augment:
            # Generate a random rotation angle.
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            # Rotate input image and mask.
            image = self._rotate_image(image, angle, 1, 'reflect')
            image_mask = self._rotate_image(image_mask, angle, order=1, mode='constant')

            # If we loaded the GT scene coordinates.
            if self.use_depth:
                # rotate and scale depth maps
                depth = resize(depth, image.shape[1:], order=0)
                depth = rotate(depth, angle, order=0, mode='constant')

            # Rotate ground truth camera pose as well.
            angle = angle * math.pi / 180.
            # Create a rotation matrix.
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle)
            pose_rot[0, 1] = -math.sin(angle)
            pose_rot[1, 0] = math.sin(angle)
            pose_rot[1, 1] = math.cos(angle)
        else:
            pose_rot = torch.eye(4)

        # Generate ground truth scene coordinates from depth.
        if self.use_depth:
            # generate initialization targets from depth map
            offsetX = int(Regressor.OUTPUT_SUBSAMPLE / 2)
            offsetY = int(Regressor.OUTPUT_SUBSAMPLE / 2)

            coords = torch.zeros((
                3,
                math.ceil(image.shape[1] / Regressor.OUTPUT_SUBSAMPLE),
                math.ceil(image.shape[2] / Regressor.OUTPUT_SUBSAMPLE)))

            # subsample to network output size
            depth = depth[offsetY::Regressor.OUTPUT_SUBSAMPLE, offsetX::Regressor.OUTPUT_SUBSAMPLE]

            # construct x and y coordinates of camera coordinate
            xy = self.prediction_grid[:, :depth.shape[0], :depth.shape[1]].copy()
            # add random pixel shift
            xy[0] += offsetX
            xy[1] += offsetY
            # substract principal point (assume image center)
            xy[0] -= image.shape[2] / 2
            xy[1] -= image.shape[1] / 2
            # reproject
            xy /= focal_length
            xy[0] *= depth
            xy[1] *= depth

            # assemble camera coordinates tensor
            eye = np.ndarray((4, depth.shape[0], depth.shape[1]))
            eye[0:2] = xy
            eye[2] = depth
            eye[3] = 1

            if self.return_eye_coords:
                # target coordinates are eye coordinates, do nothing
                sc = eye
            else:
                # target coordinates are scene coordinates, transform eye coordinates
                sc = np.matmul(pose.numpy() @ pose_rot.numpy(), eye.reshape(4, -1))
                sc = sc.reshape(4, depth.shape[0], depth.shape[1])

            # mind pixels with invalid depth
            sc[:, depth == 0] = 0
            sc[:, depth > 1000] = 0
            sc = torch.from_numpy(sc[0:3])

            coords[:, :sc.shape[1], :sc.shape[2]] = sc

        # Convert to half if needed.
        if self.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        # Invert the pose, we need world-to-camera during training.
        pose_inv = pose.inverse()
        pose_rot_inv = pose_rot.inverse()

        # Final check of poses before returning.
        if not dataset_io.check_pose(pose_inv) or not dataset_io.check_pose(pose_rot_inv):
            raise ValueError(f"Pose at index {idx} is invalid.")

        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = focal_length
        intrinsics[1, 1] = focal_length
        # Hardcode the principal point to the centre of the image.
        intrinsics[0, 2] = image.shape[2] / 2
        intrinsics[1, 2] = image.shape[1] / 2

        # Also need the inverse.
        intrinsics_inv = intrinsics.inverse()

        return image, image_mask, pose_inv, pose_rot_inv, intrinsics, intrinsics_inv, coords, str(self.rgb_files[idx]), idx

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        if self.augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
            # scale_factor = 1 / scale_factor #inverse scale sampling, not used for ACE mapping
        else:
            scale_factor = 1

        # Target image height. We compute it here in case we are asked for a full batch of tensors because we need
        # to apply the same scale factor to all of them.
        image_short_size = int(self.image_short_size * scale_factor)

        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i, image_short_size) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx, image_short_size)
