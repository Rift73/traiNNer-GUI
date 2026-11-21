import json
import os

import numpy as np
from numpy.random import Generator

from traiNNer.utils.config import Config

# Mixed with the seed for the codec routing generator, so that its stream is
# independent of the project generator's for the same seed.
_CODEC_ROUTE_TAG = 0x41505258


class RNG:
    _rng = None
    _codec_route_rng = None
    _seed: int | None = None
    _owner_pid: int | None = None

    @classmethod
    def init_rng(cls, seed: int) -> None:
        pid = os.getpid()
        # A forked worker inherits copies of the parent's streams until it
        # receives its own worker seed. Reject reseeding only in their owner.
        if cls._owner_pid == pid and (
            cls._rng is not None or cls._codec_route_rng is not None
        ):
            raise RuntimeError(
                f"RNG has already been initialized and must not be reseeded, got seed={seed}."
            )
        cls._rng = np.random.default_rng(seed)
        cls._codec_route_rng = None
        cls._seed = seed
        cls._owner_pid = pid

    @classmethod
    def _manual_seed(cls) -> int:
        seed = cls._seed if cls._seed is not None else Config.get_manual_seed()
        if seed is None:
            raise RuntimeError("Manual seed is not set.")
        return seed

    @classmethod
    def get_rng(cls) -> Generator:
        if cls._rng is None:
            cls._rng = np.random.default_rng(cls._manual_seed())
            cls._owner_pid = os.getpid()
        return cls._rng

    @classmethod
    def get_codec_route_rng(cls) -> Generator:
        """The generator that decides which eligible video codec calls run the
        GPU approximation (compress_video_approx_prob). It has the project
        generator's seed mixed with a fixed tag, so the project stream does not
        depend on the routing and the same seed routes the same calls. Like
        the project generator it is restored only when checkpoint_rng is enabled."""
        if cls._codec_route_rng is None:
            cls._codec_route_rng = np.random.default_rng(
                [cls._manual_seed(), _CODEC_ROUTE_TAG]
            )
            cls._owner_pid = os.getpid()
        return cls._codec_route_rng

    @classmethod
    def capture_streams(cls) -> tuple[str, str]:
        """JSON preserves PCG64's unsigned 128-bit integers in a weights-only state."""
        return (json.dumps(cls.get_rng().bit_generator.state),
                json.dumps(cls.get_codec_route_rng().bit_generator.state))

    @classmethod
    def restore_streams(cls, states: tuple[str, str]) -> None:
        cls.get_rng().bit_generator.state = json.loads(states[0])
        cls.get_codec_route_rng().bit_generator.state = json.loads(states[1])
