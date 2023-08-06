from typing import Any, Dict
import numpy as np
from PIL import Image


################################################################################################
#                                        Target config                                         #
################################################################################################
# features=tfds.features.FeaturesDict({
#     'steps': tfds.features.Dataset({
#         'observation': tfds.features.FeaturesDict({
#             'image': tfds.features.Image(
#                 shape=(128, 128, 3),
#                 dtype=np.uint8,
#                 encoding_format='jpeg',
#                 doc='Main camera RGB observation.',
#             ),
#         }),
#         'action': tfds.features.Tensor(
#             shape=(8,),
#             dtype=np.float32,
#             doc='Robot action, consists of [3x EEF position, '
#                 '3x EEF orientation yaw/pitch/roll, 1x gripper open/close position, '
#                 '1x terminate episode].',
#         ),
#         'discount': tfds.features.Scalar(
#             dtype=np.float32,
#             doc='Discount if provided, default to 1.'
#         ),
#         'reward': tfds.features.Scalar(
#             dtype=np.float32,
#             doc='Reward if provided, 1 on final step for demos.'
#         ),
#         'is_first': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on first step of the episode.'
#         ),
#         'is_last': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on last step of the episode.'
#         ),
#         'is_terminal': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on last step of the episode if it is a terminal step, True for demos.'
#         ),
#         'language_instruction': tfds.features.Text(
#             doc='Language Instruction.'
#         ),
#         'language_embedding': tfds.features.Tensor(
#             shape=(512,),
#             dtype=np.float32,
#             doc='Kona language embedding. '
#                 'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
#         ),
#     })
################################################################################################
#                                                                                              #
################################################################################################

def quat2euler(quat):
    x, y, z, w = quat

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.sqrt(1 + 2 * (w * y - x * z));
    cosp = np.sqrt(1 - 2 * (w * y - x * z));
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2;

    siny_cosp = 2 * (w * z + x * y);
    cosy_cosp = 1 - 2 * (y * y + z * z);
    yaw = np.arctan2(siny_cosp, cosy_cosp);

    return np.array([yaw, pitch, roll])


def transform_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Maps step from source dataset to target dataset config.
       Input is dict of numpy arrays."""
    img = Image.fromarray(step['observation']['image']).resize(
        (128, 128), Image.Resampling.LANCZOS)
    transformed_step = {
        'observation': {
            'image': np.array(img),
        },
        'action': np.concatenate(
            [
                step['action'][:3],
                quat2euler(step['action'][5:8]),
                step['action'][-1:],
                [1] if step['is_terminal'] else [0],
            ]
        ),
    }

    # copy over all other fields unchanged
    for copy_key in ['discount', 'reward', 'is_first', 'is_last', 'is_terminal',
                     'language_instruction', 'language_embedding']:
        transformed_step[copy_key] = step[copy_key]

    return transformed_step

