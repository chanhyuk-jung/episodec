import numpy as np
from imagecodecs import jpeg_decode, jpeg_encode

from .zarr import Image, ZarrDecoder, ZarrEncoder

N = 32


def test_encode_decode(tmp_path):
    encoder = ZarrEncoder(tmp_path)

    all_obs = []
    all_actions = []

    for _ in range(N):
        obs = {}
        obs["qpos"] = np.random.randn(7)

        img = np.random.rand(64, 64, 3) * 255
        img = img.astype(np.uint8)
        obs["img"] = img

        all_obs.append(obs)

        actions = {}
        actions["qpos"] = np.random.randn(7)

        all_actions.append(actions)

    for i, (o, a) in enumerate(zip(all_obs, all_actions)):
        obs = {}
        for k, v in o.items():
            if k == "img":
                obs[k] = Image.from_array(v)
                continue

            obs[k] = v

        if (i + 1) % (N // 4) == 0:
            encoder.append(obs, a, last=True)
        else:
            encoder.append(obs, a)

    encoder.flush()

    decoder = ZarrDecoder(tmp_path)

    for i in range(N):
        dec_obs, dec_actions = decoder[i]

        o = all_obs[i]
        a = all_actions[i]

        for k in o.keys():
            if k == "img":
                img = o[k]

                buf = jpeg_encode(img, level=90)
                img = jpeg_decode(buf)

                np.testing.assert_allclose(dec_obs[k], img)
                continue

            np.testing.assert_allclose(dec_obs[k], o[k])

        for k in a.keys():
            np.testing.assert_allclose(dec_actions[k], a[k])
