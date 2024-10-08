# https://github.com/carloscdias/whisper-cpp-python/pull/12#issuecomment-1803553186
import re
import importlib.util
from pathlib import Path

try:
    import whisper_cpp_python
except FileNotFoundError:
    regex = r"(\"darwin\":\n\s*lib_ext = \")\.so(\")"
    subst = "\\1.dylib\\2"

    package = importlib.util.find_spec("whisper_cpp_python")
    whisper_path = Path(package.origin)
    whisper_cpp_py = whisper_path.parent.joinpath("whisper_cpp.py")
    content = whisper_cpp_py.read_text()
    result = re.sub(regex, subst, content, 0, re.MULTILINE)
    whisper_cpp_py.write_text(result)

    import whisper_cpp_python
