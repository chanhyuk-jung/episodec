import os

import numpy as np
import pytest
from imagecodecs import jpeg_decode, jpeg_encode
from imagecodecs.zarr import Jpeg
from numpy.typing import NDArray

from .zarr import Image, ZarrDecoder, ZarrEncoder

N = 8
seed = 42


def random_image(
    length: int, width: int = 64, height: int = 64, channels: int = 3
) -> NDArray[np.uint8]:
    x = np.random.rand(length, height, width, channels) * 255
    img = x.astype(np.uint8)
    return img


@pytest.fixture
def episodes():
    episodes = []

    for _ in range(N):
        length = np.random.randint(low=4, high=16)
        print(length)

        episode = (
            {
                "qpos": np.random.randn(length, 7),
                "qvel": np.random.randn(length, 7),
            },
            {
                "front_img": random_image(length),
                "wrist_img": random_image(length),
            },
            {
                "qpos": np.random.randn(length, 7),
            },
            length,
        )

        episodes.append(episode)

    return tuple(episodes)


def test_encode_decode(tmp_path, episodes):
    zarr_path = os.path.join(tmp_path, "fake")

    encoder = ZarrEncoder(zarr_path, shard_size=4, buffer_size=2)

    for states, frames, actions, length in episodes:
        for i in range(length):
            q_t = {k: v[i] for k, v in states.items()}
            I_t = {
                k: Image.from_array(v[i], codec=Jpeg(level=90))
                for k, v in frames.items()
            }

            a_t = {k: v[i] for k, v in actions.items()}

            encoder.append(obs=q_t | I_t, actions=a_t)

        encoder.mark_last()

    encoder.flush()

    decoder = ZarrDecoder(zarr_path)

    episode_ends = decoder.get_episode_ends()
    episode_starts = [0, *episode_ends[:-1]]

    for i in range(N):
        s = episode_starts[i]
        e = episode_ends[i]

        decoded_obs = decoder.slice_obs(s, e)
        decoded_actions = decoder.slice_actions(s, e)

        states, frames, actions, length = episodes[i]

        assert e - s == length

        for k, v in states.items():
            np.testing.assert_allclose(decoded_obs[k], v)

        for k, v in frames.items():
            frames = []
            for img in v:
                buf = jpeg_encode(img, level=90)
                img = jpeg_decode(buf)
                frames.append(img)

            frames = np.stack(frames)

            np.testing.assert_allclose(decoded_obs[k], frames)

        for k, v in actions.items():
            np.testing.assert_allclose(decoded_actions[k], v)
