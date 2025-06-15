import numpy as np
from pathlib import Path


class BVHReader:
    """
    A class to read and parse BVH (Biovision Hierarchy) files.
    This class loads the skeleton hierarchy and motion data.
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.joints = {}
        self.joint_names_ordered = []
        self.root_name = None
        self.frames = []
        self.frame_time = 0.0
        self.num_frames = 0
        self._parse_bvh()

    def _parse_bvh(self):
        """Main parsing function that reads the file line by line."""
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "HIERARCHY":
                i = self._parse_hierarchy(lines, i + 1, parent=None)
            elif line == "MOTION":
                i = self._parse_motion(lines, i + 1)
            else:
                i += 1

    def _parse_hierarchy(self, lines, i, parent):
        """Recursively parses the HIERARCHY section to build the skeleton."""
        line = lines[i].strip()
        parts = line.split()
        joint_type = parts[0]
        name = parts[1] if joint_type != "End" else parent["name"] + "_EndSite"

        if joint_type == "End":
            name = f"{parent['name']}_EndSite"
            joint_type = "End Site"

        joint = {
            "name": name,
            "type": joint_type,
            "parent": parent,
            "children": [],
            "offset": np.zeros(3),
            "channels": [],
            "channel_indices": {},
            "global_channel_offset": 0,
        }
        self.joints[name] = joint
        # The order is crucial for mapping to the final numpy array
        self.joint_names_ordered.append(name)

        if parent:
            self.joints[parent["name"]]["children"].append(joint)
        else:
            self.root_name = name

        i += 1
        while i < len(lines):
            line = lines[i].strip()
            if line == "{":
                i += 1
            elif line.startswith("OFFSET"):
                offset_values = [float(x) for x in line.split()[1:]]
                joint["offset"] = np.array(offset_values)
                i += 1
            elif line.startswith("CHANNELS"):
                if joint_type == "End Site":
                    i += 1
                    continue
                channel_info = line.split()
                joint["channels"] = channel_info[2:]
                for idx, channel_name in enumerate(joint["channels"]):
                    joint["channel_indices"][channel_name] = idx
                i += 1
            elif line.startswith("JOINT") or line.startswith("End"):
                i = self._parse_hierarchy(lines, i, parent=joint)
            elif line == "}":
                i += 1
                return i
            else:
                i += 1
        return i

    def _parse_motion(self, lines, i):
        """Parses the MOTION section to get frame count, time, and data."""
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Frames:"):
                self.num_frames = int(line.split()[1])
            elif line.startswith("Frame Time:"):
                self.frame_time = float(line.split()[2])
            else:
                # Assuming the line after "Frame Time:" is the start of motion data
                break
            i += 1

        current_global_offset = 0
        for name in self.joint_names_ordered:
            joint = self.joints[name]
            joint["global_channel_offset"] = current_global_offset
            current_global_offset += len(joint["channels"])

        for k in range(self.num_frames):
            line = lines[i + k].strip()
            try:
                frame_data = [float(x) for x in line.split()]
                if len(frame_data) != current_global_offset:
                    print(f"\nWarning: Frame {k} has {len(frame_data)} channels, expected {current_global_offset}.")
                self.frames.append(np.array(frame_data))
            except (ValueError, IndexError) as e:
                print(f"\nError parsing motion data for frame {k}: {line}. Error: {e}")
                self.frames.append(np.zeros(current_global_offset))
        return i + self.num_frames

    def get_joint_channels_for_frame(self, frame_idx: int, joint_name: str):
        """Extracts motion data for a specific joint at a specific frame."""
        if frame_idx >= self.num_frames or frame_idx < 0:
            raise IndexError(f"Frame index {frame_idx} out of bounds.")
        if joint_name not in self.joints:
            raise ValueError(f"Joint '{joint_name}' not found.")

        joint = self.joints[joint_name]
        if joint["type"] == "End Site":
            return np.array([])

        start_idx = joint["global_channel_offset"]
        end_idx = start_idx + len(joint["channels"])

        if end_idx > len(self.frames[frame_idx]):
            return np.zeros(len(joint["channels"]))

        return self.frames[frame_idx][start_idx:end_idx]

    @staticmethod
    def _euler_to_rotation_matrix(rx, ry, rz):
        """Converts Euler angles (in degrees) to a 3x3 rotation matrix (XYZ order)."""
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
        R_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        R_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        R_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        # Standard rotation order is Z, then Y, then X. In matrix multiplication, this is R_x @ R_y @ R_z.
        return R_x @ R_y @ R_z

    def calculate_global_positions(self, frame_idx: int):
        """Calculates the world-space 3D position of each joint for a given frame."""
        global_positions = {}
        global_transforms = {}

        for joint_name in self.joint_names_ordered:
            joint = self.joints[joint_name]
            channels_data = self.get_joint_channels_for_frame(frame_idx, joint_name)

            # Create local transformation matrix
            local_transform = np.eye(4)
            local_transform[:3, 3] = joint["offset"]

            tx, ty, tz = 0.0, 0.0, 0.0
            rx, ry, rz = 0.0, 0.0, 0.0

            if joint["type"] != "End Site":
                channel_map = joint["channel_indices"]
                for channel_name, val_idx in channel_map.items():
                    val = channels_data[val_idx]
                    if channel_name == "Xposition":
                        tx = val
                    elif channel_name == "Yposition":
                        ty = val
                    elif channel_name == "Zposition":
                        tz = val
                    elif channel_name == "Xrotation":
                        rx = val
                    elif channel_name == "Yrotation":
                        ry = val
                    elif channel_name == "Zrotation":
                        rz = val

            # Root joint has translation, other joints' translation is from their offset
            if joint_name == self.root_name:
                local_transform[:3, 3] += np.array([tx, ty, tz])

            # Apply rotation for all non-End-Site joints
            if joint["type"] != "End Site":
                rotation_matrix = self._euler_to_rotation_matrix(rx, ry, rz)
                local_transform[:3, :3] = rotation_matrix

            # Chain transformations: parent_global * local
            if joint["parent"]:
                parent_transform = global_transforms[joint["parent"]["name"]]
                global_transform = parent_transform @ local_transform
            else:
                global_transform = local_transform

            global_transforms[joint_name] = global_transform
            global_positions[joint_name] = global_transform[:3, 3]

        return global_positions


def export_joint_convention(bvh_reader: BVHReader, output_filepath: Path):
    """
    Exports the skeleton's structure (joint names and bone connections)
    to a Python file.

    Args:
        bvh_reader (BVHReader): The parsed BVH reader instance.
        output_filepath (Path): The path to save the 'joint_convention.py' file.
    """
    print(f"\nExporting joint convention to: {output_filepath}")

    joint_names = bvh_reader.joint_names_ordered
    joint_to_idx = {name: i for i, name in enumerate(joint_names)}

    bones = []
    for joint_name, joint_data in bvh_reader.joints.items():
        if joint_name not in joint_to_idx:
            continue

        parent_idx = joint_to_idx[joint_name]
        for child_joint in joint_data["children"]:
            child_name = child_joint["name"]
            if child_name in joint_to_idx:
                child_idx = joint_to_idx[child_name]
                bones.append((parent_idx, child_idx))

    # Create the content of the Python file as a string
    file_content = f'''"""
This file was generated automatically. It contains the skeleton definition
for the associated animation data.
"""

# A list of joint names, where the index in this list corresponds to the
# joint's index in the pose data arrays.
JOINT_NAMES = [
'''
    for name in joint_names:
        file_content += f"    '{name}',\n"
    file_content += "]\n\n"

    file_content += """# A list of bones, connecting the joints. Each bone is a tuple of two indices,
# representing the (parent, child) connection.
BONES = [
"""
    for p_idx, c_idx in bones:
        file_content += f"    ({p_idx}, {c_idx}),  # {joint_names[p_idx]} -> {joint_names[c_idx]}\n"
    file_content += "]\n"

    # Write the string to the specified file
    with open(output_filepath, "w") as f:
        f.write(file_content)
    print("Joint convention file created successfully.")


def export_poses_to_npy(bvh_reader: BVHReader, output_dir: Path):
    """
    Exports the 3D joint positions for each frame to separate .npy files.
    The output is transformed to a Z-up coordinate system, centered on the
    root joint's XY position, and grounded at Z=0 for each frame.

    Args:
        bvh_reader (BVHReader): The parsed BVH reader instance.
        output_dir (Path): The directory where the .npy files will be saved.
    """
    print(f"\nExporting {bvh_reader.num_frames} frames to NPY format...")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory for poses: {output_dir}")
    print("Applying transformations: Z-up, XY-centering, Z-grounding.")

    joint_names = bvh_reader.joint_names_ordered
    num_joints = len(joint_names)
    # Get the index of the root joint to use for centering
    try:
        root_idx = joint_names.index(bvh_reader.root_name)
    except (ValueError, IndexError):
        print(f"Warning: Root joint '{bvh_reader.root_name}' not found in ordered list. Will not center on XY plane.")
        root_idx = -1

    for i in range(bvh_reader.num_frames):
        # 1. Calculate global positions for the current frame in the original coordinate system
        global_positions_dict = bvh_reader.calculate_global_positions(i)

        # 2. Create the (J, 3) numpy array for the current frame's pose
        pose_array = np.zeros((num_joints, 3))
        for j, joint_name in enumerate(joint_names):
            if joint_name in global_positions_dict:
                pose_array[j] = global_positions_dict[joint_name]

        # 3. Transform coordinates from Y-up (BVH standard) to Z-up.
        #    (x, y, z) -> (x, -z, y)
        transformed_pose = np.zeros_like(pose_array)
        transformed_pose[:, 0] = pose_array[:, 0]  # X remains X
        transformed_pose[:, 1] = -pose_array[:, 2]  # Y becomes -Z
        transformed_pose[:, 2] = pose_array[:, 1]  # Z becomes Y (new up-axis)

        # 4. Center the skeleton on the XY plane based on the root joint's position
        if root_idx != -1:
            root_xy_offset = transformed_pose[root_idx, :2].copy()
            transformed_pose[:, :2] -= root_xy_offset

        # 5. Ground the skeleton at z=0 for the current frame
        #    Find the lowest point on the new Z-axis and shift the whole skeleton up.
        min_z = transformed_pose[:, 2].min()
        transformed_pose[:, 2] -= min_z

        # 6. Define the output file path and save the transformed pose
        output_filename = f"{i:06d}.npy"
        output_filepath = output_dir / output_filename
        np.save(output_filepath, transformed_pose)

        # Print progress on a single line
        print(f"  - Saved frame {i + 1}/{bvh_reader.num_frames} to {output_filepath}", end="\r")

    print("\n\nNPY pose export complete! 🎉")


if __name__ == "__main__":
    # --- Instructions ---
    # 1. Set 'bvh_file_path' to point to your .bvh file using pathlib.Path.
    # 2. Run the script.
    # 3. It will create two outputs in the same directory as your BVH file:
    #    a) A 'poses' folder containing the .npy file for each frame. The pose
    #       data will be centered, grounded, and in a Z-up coordinate system.
    #    b) A 'joint_convention.py' file defining the skeleton structure.

    # This is the only path you should need to change.
    bvh_file_path = Path("data/recordings/20250526_Pose/Rokoko/Animation.bvh")

    if not bvh_file_path.exists():
        print(f"Error: BVH file not found at '{bvh_file_path}'")
        print("Please update 'bvh_file_path' in the script to point to a valid file.")
    else:
        try:
            # The output directory will be data/recordings/20250526_Pose/Rokoko/
            output_parent_dir = bvh_file_path.parent
            poses_output_dir = output_parent_dir / "poses"
            convention_output_file = output_parent_dir / "joint_convention.py"

            # Load and parse the BVH file
            print(f"Loading BVH file: {bvh_file_path}")
            reader = BVHReader(bvh_file_path)
            print("BVH file loaded successfully.")

            # --- EXPORT JOINT CONVENTION ---
            export_joint_convention(reader, convention_output_file)

            # --- EXPORT POSE DATA ---
            export_poses_to_npy(reader, poses_output_dir)

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
