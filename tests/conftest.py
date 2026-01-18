from gguf_clone.gguf_path import configure_gguf_path

# Ensure vendored gguf is importable during tests.
configure_gguf_path()
