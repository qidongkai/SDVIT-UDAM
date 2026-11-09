"""
Power measurement on the Snapdragon platform.

Follows Step 5 of the deployment pipeline in Section 3.5.5 of the thesis:
the average power consumption is measured over a 60-second continuous
inference at 30 FPS using the device's on-board power monitoring
interface.
"""
import subprocess

import yaml


def measure(cfg_path="configs/default.yaml",
            dlc_path="sdvit_int8.dlc",
            input_list="input_list.txt",
            duration=60,
            fps=30):
    """
    Run the model continuously on the device and read the power monitor.

    Args:
        cfg_path:   path to the YAML config file.
        dlc_path:   path to the quantized DLC file.
        input_list: text file listing the input tensors.
        duration:   duration of the measurement in seconds.
        fps:        target frames per second.

    Returns:
        The instantaneous power reading in microwatts (string).
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    snpe_root = cfg["deploy"]["snapdragon_snpe"]
    runner = snpe_root + "/bin/snpe-net-run"
    iterations = duration * fps

    print(f"Running {iterations} iterations ({duration}s at {fps} FPS)...")
    subprocess.run(
        [
            "adb", "shell",
            runner,
            "--container", dlc_path,
            "--input_list", input_list,
            "--iterations", str(iterations),
        ],
        check=True,
    )

    out = subprocess.run(
        ["adb", "shell", "cat", "/sys/class/power_supply/battery/power_now"],
        capture_output=True,
        text=True,
    )
    reading = out.stdout.strip()
    print(f"Instantaneous power: {reading} uW")
    return reading


if __name__ == "__main__":
    measure()