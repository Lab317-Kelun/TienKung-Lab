import pickle
import numpy as np
import torch
import argparse
from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat  
from scipy.spatial.transform import Rotation 

def convert_pkl_to_custom(input_pkl, output_txt, fps):
    dt = 1.0 / fps

    # Handle NumPy version compatibility for pickle loading
    # If pickle file was created with NumPy 2.x but we're using NumPy 1.x
    try:
        with open(input_pkl, "rb") as f:
            motion_data = pickle.load(f)
    except (ModuleNotFoundError, AttributeError) as e:
        if "numpy" in str(e) or "_core" in str(e) or "multiarray" in str(e):
            # NumPy version mismatch: file created with NumPy 2.x, loading with NumPy 1.x
            # Create a custom unpickler that maps numpy._core to numpy
            import sys
            import warnings
            warnings.warn(
                f"NumPy version mismatch detected ({e}). "
                "Attempting to load with compatibility workaround...",
                UserWarning
            )
            
            # Workaround: create a custom unpickler that handles NumPy version mismatch
            class CompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    original_module = module
                    
                    # First, try to handle known NumPy module mappings
                    # Map numpy._core.* to numpy.* for NumPy 2.x -> 1.x compatibility
                    if module.startswith('numpy._core'):
                        module = module.replace('numpy._core', 'numpy')
                    # Map numpy.multiarray -> numpy.core.multiarray (NumPy 1.x path)
                    elif module == 'numpy.multiarray':
                        module = 'numpy.core.multiarray'
                    
                    # Try standard loading first
                    try:
                        return super().find_class(module, name)
                    except (ModuleNotFoundError, AttributeError) as e:
                        # Fallback: Direct access to numpy.core.multiarray
                        if 'multiarray' in original_module or 'multiarray' in module:
                            try:
                                # Direct access to numpy.core.multiarray
                                multiarray_module = np.core.multiarray
                                if hasattr(multiarray_module, name):
                                    return getattr(multiarray_module, name)
                            except (AttributeError, ModuleNotFoundError):
                                pass
                        
                        # Fallback: Manual module navigation
                        if module.startswith('numpy'):
                            try:
                                parts = module.split('.')
                                obj = np
                                for part in parts[1:]:
                                    if hasattr(obj, part):
                                        obj = getattr(obj, part)
                                    else:
                                        break
                                else:
                                    # All parts found, try to get the attribute
                                    if hasattr(obj, name):
                                        return getattr(obj, name)
                            except (AttributeError, ModuleNotFoundError):
                                pass
                        
                        # Fallback: Known NumPy functions
                        known_functions = {
                            '_reconstruct': lambda: getattr(np.core.multiarray, '_reconstruct', None),
                            'scalar': lambda: getattr(np.core.multiarray, 'scalar', None),
                        }
                        if name in known_functions:
                            try:
                                func = known_functions[name]()
                                if func is not None:
                                    return func
                            except (AttributeError, ModuleNotFoundError):
                                pass
                        
                        # Last resort: try importing the module directly
                        try:
                            import importlib
                            mod = importlib.import_module(module)
                            if hasattr(mod, name):
                                return getattr(mod, name)
                        except (ImportError, ModuleNotFoundError, AttributeError):
                            pass
                        
                        # If all strategies fail, raise informative error
                        raise ModuleNotFoundError(
                            f"No module named '{original_module}' (mapped to '{module}'). "
                            f"Failed to find '{name}'. "
                            f"This is likely a NumPy version compatibility issue. "
                            f"Original error: {e}"
                        ) from e
            
            with open(input_pkl, "rb") as f:
                motion_data = CompatUnpickler(f).load()
        else:
            raise

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

    q1_conj = quat_conjugate(root_rot_t[:-1])         
    dq = quat_mul(q1_conj, root_rot_t[1:])            
    axis_angle = axis_angle_from_quat(dq)             
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1],  
         root_lin_vel, root_ang_vel, dof_vel),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps)
