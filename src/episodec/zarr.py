import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import numpy as np
import zarr as _zarr
from imagecodecs.zarr import ArrayBytesCodec, Jpeg, register_codecs
from numpy.typing import ArrayLike, NDArray

register_codecs()


def makedirs(path: str | os.PathLike):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


@dataclass(frozen=True)
class Image:
    frame: NDArray[np.uint8]
    codec: ArrayBytesCodec

    @classmethod
    def from_array(cls, frame: ArrayLike, codec: ArrayBytesCodec | None = None) -> Self:
        img = np.asarray(frame)

        if img.dtype != np.uint8:
            raise ValueError("dtype must be uint8")

        if img.ndim != 3:
            raise ValueError("invalid image shape")

        if codec is None:
            codec = Jpeg(level=90)

        return cls(frame=img, codec=codec)


def array_keys(root: _zarr.Group) -> set[str]:
    array_keys = list(root.array_keys())
    group_keys = list(root.group_keys())

    while len(group_keys) > 0:
        new_keys = []
        for name in group_keys:
            g = root.get_group(name)

            array_keys.extend([f"{name}/{key}" for key in g.array_keys()])
            new_keys.extend([f"{name}/{key}" for key in g.group_keys()])

        group_keys = new_keys

    return set(array_keys)


class ZarrEncoder:
    def __init__(
        self, path: str | os.PathLike, *, shard_size: int = 32, buffer_size: int = 128
    ):
        root_dir = Path(path)

        makedirs(root_dir.parent)

        self._group = _zarr.open_group(path, mode="w")
        self._names = array_keys(self._group)

        self.shard_size = shard_size

        self._buffer = []
        self.buffer_size = buffer_size

        self._group.update_attributes({"length": 0, "episode_ends": []})

        self._length = 0

    def _check(self, data) -> None:
        if len(self._names) == 0:
            self._names.update(data.keys())

        if self._names != set(data.keys()):
            raise ValueError(f"some keys are missing: {self._names - set(data)}")

    def append(
        self,
        obs: dict[str, NDArray | Image],
        actions: dict[str, NDArray],
    ) -> None:
        data = flatten_dict({"observations": obs, "actions": actions})

        self._check(data)

        self._buffer.append(data)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def mark_last(self):
        self.flush()

        self._group.update_attributes({"length": self._length})

        ends = cast(list[int], self._group.attrs["episode_ends"])
        ends.append(self._length)

        self._group.update_attributes({"episode_ends": ends})

    def flush(self) -> None:
        if len(self._buffer) == 0:
            return

        length = self._flush_buffer()
        self._length += length

        self._buffer = []

    def _flush_buffer(self) -> int:
        for name in self._names:
            first = self._buffer[0][name]
            if isinstance(first, Image):
                x = np.stack([buf[name].frame for buf in self._buffer])
                serializer = first.codec
            else:
                x = np.stack([buf[name] for buf in self._buffer])
                serializer = "auto"

            length = x.shape[0]

            if name in self._group:
                z = self._group.get_array(name)
                z.append(x)
                continue

            z = self._group.create_array(
                name,
                shape=x.shape,
                dtype=x.dtype,
                chunks=[1, *x.shape[1:]],
                shards=[self.shard_size, *x.shape[1:]],
                compressors=None,
                serializer=serializer,
            )
            z[:] = x

        return length

    def prune(self) -> None:
        length = cast(int, self._group.attrs["length"])

        for name in self._names:
            z = self._group.get_array(name)

            if z.shape[0] > length:
                z.resize([length, *z.shape[1:]])


class ZarrDecoder:
    def __init__(self, path: str | os.PathLike):
        root_dir = Path(path)

        if not root_dir.exists():
            raise RuntimeError(f"{path} does not exist")

        self._group = _zarr.open_group(root_dir, mode="r")

        self.obs_names = array_keys(self._group.get_group("observations"))
        self.action_names = array_keys(self._group.get_group("actions"))

        self._length = cast(int, self._group.attrs.get("length", 0))

    def __getitem__(
        self, index: int | slice
    ) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        obs = {}
        for name in self.obs_names:
            x = self._group.get_array(f"observations/{name}")[index]
            obs[name] = np.asarray(x)

        actions = {}
        for name in self.action_names:
            x = self._group.get_array(f"actions/{name}")[index]
            actions[name] = np.asarray(x)

        return obs, actions

    def __len__(self) -> int:
        return self._length

    def slice_obs(
        self, start: int, stop: int, step: int | None = None
    ) -> dict[str, NDArray]:
        data = {}

        for name in self.obs_names:
            x = self._group.get_array(f"observations/{name}")[start:stop:step]
            data[name] = np.asarray(x)

        return data

    def slice_actions(
        self, start: int, stop: int, step: int | None = None
    ) -> dict[str, NDArray]:
        data = {}

        for name in self.action_names:
            x = self._group.get_array(f"actions/{name}")[start:stop:step]
            data[name] = np.asarray(x)

        return data

    def get_episode_ends(self) -> list[int]:
        ends = cast(list[int], self._group.attrs.get("episode_ends", []))
        return ends
