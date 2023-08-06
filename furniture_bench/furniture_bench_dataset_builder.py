from typing import Iterator, Tuple, Any

import glob
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class FurnitureBenchDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for FurnitureBench dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(35,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x eef position, '
                                '4x eef quaternion, 3x eef linear velocity, '
                                '3x eef angular velocity, 7x joint position, '
                                '7x joint velocity, 7x joint torque, '
                                '1x gripper width].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x eef pos velocities, '
                            '4x eef quat velocities, 1x gripper velocity].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='+1 reward for each two-part assembly.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                    'skill_completion': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='+1 skill completion reward; otherwise, 0.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'furniture': tfds.features.Text(
                        doc='Furniture model name.'
                    ),
                    'initial_randomness': tfds.features.Text(
                        doc='Randomness in furniture initial configuration.'
                            '[low, med, high]'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            # 'train': self._generate_examples(path='/shared/youngwoon/furniture_dataset/low/lamp/*.pkl'),
            # 'train': self._generate_examples(path='/shared/youngwoon/furniture_dataset/high/lamp/*.pkl'),
            'train': self._generate_examples(path='/shared/youngwoon/furniture_dataset/**/*.pkl'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            with open(episode_path, "rb") as f:
                data = pickle.load(f)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            episode_length = len(data["actions"])
            obs = data["observations"][:-1]
            actions = data["actions"]  # 3D eef vel, 4D eef quat vel, 1D gripper
            rewards = data["rewards"]  # +1 when two parts are assembled
            skills = data["skills"]  # sub-task completion annotated by operator
            language_instruction = f"assemble {data['furniture']}"
            # compute Kona language embedding
            language_embedding = self._embed([language_instruction])[0].numpy()

            for i, (ob, ac, rew, skill) in enumerate(zip(obs, actions, rewards, skills)):
                episode.append({
                    'observation': {
                        'image': ob['color_image2'],
                        'wrist_image': ob['color_image1'],
                        'state': np.concatenate(list(ob['robot_state'].values()), dtype=np.float32),
                    },
                    'action': ac.astype(np.float32),
                    'discount': 1.0,
                    'reward': rew,
                    'is_first': i == 0,
                    'is_last': i == (episode_length - 1),
                    'is_terminal': i == (episode_length - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                    'skill_completion': skill,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'furniture': data["furniture"],
                    'initial_randomness': episode_path.split("/")[-3],
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path, recursive=True)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

