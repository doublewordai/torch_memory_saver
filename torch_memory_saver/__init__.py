from .hooks.mode_preload import configure_subprocess

__all__ = ["TorchMemorySaver", "configure_subprocess", "torch_memory_saver"]


def __getattr__(name):
    if name == "TorchMemorySaver":
        from .entrypoint import TorchMemorySaver as _TorchMemorySaver

        return _TorchMemorySaver
    if name == "torch_memory_saver":
        from .entrypoint import TorchMemorySaver as _TorchMemorySaver

        instance = _TorchMemorySaver()
        globals()["torch_memory_saver"] = instance
        return instance
    raise AttributeError(name)
