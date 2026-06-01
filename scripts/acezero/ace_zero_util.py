import sys
import subprocess
import logging
import numpy as np

_logger = logging.getLogger(__name__)

TRAINING_EXE = [sys.executable, "train_ace.py"]
REGISTER_EXE = [sys.executable, "register_mapping.py"]
INIT_POSES_EXE = [sys.executable, "init_all_poses.py"]

def run_cmd(cmd, raise_on_error=True, verbose=True):
    """
    Executes a command in a subprocess and prints its output to stdout.

    Args:
        cmd (list): The command to be executed, represented as a list of strings.
        raise_on_error (bool, optional): If True, raises a RuntimeError if the command returns a non-zero exit code.
                                          Defaults to True.
        verbose (bool, optional): If True, the output of the subprocess is printed to stdout. Defaults to True.

    Returns:
        int: The return code of the executed command.

    Raises:
        RuntimeError: If the command returns a non-zero exit code and raise_on_error is True.
    """

    # Convert each element of the command to a string
    cmd_str = [str(c) for c in cmd]

    # Start a subprocess with the command
    proc = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Continuously read and print the output of the subprocess to stdout
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        # If verbose is True, print the output of the subprocess to stdout
        if verbose:
            sys.stdout.write(line)
            sys.stdout.flush()

    # Wait for the subprocess to finish and get its return code
    returncode = proc.wait()

    # If the return code is non-zero and raise_on_error is True, raise a RuntimeError
    if returncode != 0 and raise_on_error:
        raise RuntimeError("Error running ACE0: \nCommand:\n" + " ".join(cmd_str))

    # Return the return code of the subprocess
    return returncode


def get_seed_id(seed_idx):
    return f"iteration0_seed{seed_idx}"


def get_render_path(out_dir):
    return out_dir / "renderings"


def _get_common_mapping_cmd(rgb_files, iteration_id, out_dir, opt):
    """Constructs the common base for a mapping command."""
    # Specify base mapping call with exe, dataset, and output map file
    mapping_cmd = [
        *TRAINING_EXE,
        rgb_files,
        out_dir / f"{iteration_id}.pt",
    ]

    # Add parameters common to all mapping commands
    mapping_cmd += [
        "--loss_structure", opt.loss_structure,
        "--prior_loss_type", opt.prior_loss_type,
        "--prior_loss_weight", opt.prior_loss_weight,
        "--prior_loss_bandwidth", opt.prior_loss_bandwidth,
        "--prior_loss_location", opt.prior_loss_location,
        "--prior_diffusion_start_step", opt.prior_diffusion_start_step,
        "--prior_diffusion_warmup_steps", opt.prior_diffusion_warmup_steps,
        "--prior_diffusion_subsample", opt.prior_diffusion_subsample,
        "--render_target_path", get_render_path(out_dir),
        "--render_marker_size", opt.render_marker_size,
        "--refinement_ortho", opt.refinement_ortho,
        "--ace_pose_file_conf_threshold", opt.registration_confidence,
        "--render_flipped_portrait", opt.render_flipped_portrait,
        "--image_resolution", opt.image_resolution,
        "--pose_refinement_lr", opt.pose_refinement_lr,
        "--num_head_blocks", opt.num_head_blocks,
        "--repro_loss_hard_clamp", opt.repro_loss_hard_clamp,
        "--repro_loss_soft_clamp", opt.repro_loss_soft_clamp,
        "--iterations_output", opt.iterations_output,
        "--max_dataset_passes", opt.max_dataset_passes,
        "--learning_rate_cooldown_iterations", opt.cooldown_iterations,
        "--learning_rate_cooldown_trigger_percent_threshold", opt.cooldown_threshold,
        "--aug_rotation", opt.aug_rotation,
        "--training_buffer_cpu", opt.training_buffer_cpu,
        "--render_depth_hist", opt.render_depth_hist,
    ]

    if opt.prior_diffusion_model_path is not None:
        mapping_cmd += ["--prior_diffusion_model_path", opt.prior_diffusion_model_path]

    return mapping_cmd


def get_refit_mapping_cmd(rgb_files, iteration_id, out_dir, opt):
    """
    Constructs the mapping command for the last refinement iteration with a given scene and iteration.
    """
    # Start with the common set of options
    mapping_cmd = _get_common_mapping_cmd(rgb_files, iteration_id, out_dir, opt)

    # Determine the repro_loss_type, with a special case for the "dsac*" loss structure
    repro_loss_type = opt.repro_loss_type
    if opt.loss_structure == "dsac*":
        repro_loss_type = "dyntanh"

    # Add options specific to the refit command
    mapping_cmd += [
        "--repro_loss_type", repro_loss_type,
        "--pose_refinement_wait", opt.final_refit_posewait,
        "--learning_rate_schedule", "circle",
        "--learning_rate_max", 0.005,
        "--iterations", opt.refit_iterations,
    ]

    return mapping_cmd


def get_base_mapping_cmd(rgb_files, iteration_id, out_dir, opt):
    """
    Constructs the base mapping command for a given scene and iteration.
    """
    # Start with the common set of options
    mapping_cmd = _get_common_mapping_cmd(rgb_files, iteration_id, out_dir, opt)

    # Add options specific to the base mapping command
    mapping_cmd += [
        "--repro_loss_type", opt.repro_loss_type,
        "--pose_refinement_wait", opt.pose_refinement_wait,
        "--learning_rate_schedule", opt.learning_rate_schedule,
        "--learning_rate_max", opt.learning_rate_max,
        "--iterations", str(opt.seed_iterations),
    ]

    return mapping_cmd


def get_registration_rates(pose_file, thresholds):
    """
    Calculates the registration rates for a given pose file and a list of confidence thresholds.

    Args:
        pose_file (str): The path to the pose file.
        thresholds (list): A list of confidence thresholds for which the registration rates are to be calculated.

    Returns:
        list: A list of registration rates for each threshold. The registration rate for a threshold is the proportion
              of confidences in the pose file that are greater than the threshold.
    """

    # Open the pose file and read its contents
    with open(pose_file, 'r') as f:
        data = f.readlines()

    # Extract the confidence values from the pose file
    confidences = [float(line.split()[-1]) for line in data]
    confidences = np.array(confidences)

    # Calculate the total number of entries in the pose file
    num_entries = confidences.shape[0]

    # Calculate and return the registration rates for each threshold
    return [(confidences > t).sum() / num_entries for t in thresholds]


def map_seed(args):
    """
    Maps and scores a seed image for a given scene.

    Args:
        args (tuple): A tuple containing the following parameters:
            seed_idx (int): The index of the seed.
            seed (int): The seed to be mapped.
            rgb_files (str): Glob pattern to match input RGB files.
            out_dir (Path): The directory where the output will be stored.
            opt (Namespace): An argparse.Namespace object containing various options for the mapping and scoring process.
            verbose (bool): If True, the output of the subprocess is printed to stdout.
            visualisation (bool): If True, visualisation is rendered during the mapping process.
            mapping_only (bool): If True, only mapping is performed and scoring is skipped.

    Returns:
        float: The registration rate of the seed image.
    """

    seed_idx, seed, rgb_files, out_dir, opt, verbose, visualisation, mapping_only = args

    _logger.info(f"Processing seed {seed_idx}: {seed}")

    iteration_id = get_seed_id(seed_idx)

    # get base mapping call
    mapping_cmd = get_base_mapping_cmd(rgb_files, iteration_id, out_dir, opt)

    # determine number of workers available for each seed
    num_seed_workers = opt.num_data_workers // opt.seed_parallel_workers
    mapping_cmd += ["--num_data_workers", num_seed_workers]

    # setting parameters for mapping seed
    mapping_cmd += ["--render_visualization", visualisation]

    use_heuristic_focal_length = opt.use_external_focal_length < 0
    mapping_cmd += [
        "--use_pose_seed", seed,
        "--iterations", opt.seed_iterations,
        "--use_heuristic_focal_length", use_heuristic_focal_length,
    ]
    if not use_heuristic_focal_length:
        mapping_cmd += ["--use_external_focal_length", opt.use_external_focal_length]

    if opt.depth_files is not None:
        mapping_cmd += ["--depth_files", opt.depth_files]

    # map the seed image
    run_cmd(mapping_cmd, verbose=verbose)

    if mapping_only:
        return

    # scoring the seed
    scoring_cmd = [
        *REGISTER_EXE,
        rgb_files,
        out_dir / f"{iteration_id}.pt",
        "--render_visualization", False, # no visualization for scoring
        "--render_target_path", get_render_path(out_dir),
        "--render_marker_size", opt.render_marker_size,
        "--render_flipped_portrait", opt.render_flipped_portrait,
        "--session", f"{iteration_id}_fastcheck",
        "--confidence_threshold", opt.registration_confidence,
        "--use_external_focal_length", opt.use_external_focal_length,
        "--hypotheses", opt.ransac_iterations,
        "--threshold", opt.ransac_threshold,
        "--max_estimates", 1000, # scoring using a subset of images for large datasets
        "--image_resolution", opt.image_resolution,
        "--num_data_workers", num_seed_workers,
        "--hypotheses_max_tries", 16
    ]
    run_cmd(scoring_cmd, verbose=verbose)

    # check the number of registered mapping images
    registration_rate = get_registration_rates(
        pose_file=out_dir / f"poses_{iteration_id}_fastcheck.txt",
        thresholds=[opt.registration_confidence])[0]

    _logger.info(f"Seed successfully registered {registration_rate * 100:.1f}% of mapping images.")
    return registration_rate
