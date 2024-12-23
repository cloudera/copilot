from ._version import __version__

# expose embedding model providers on the package root
from .embedding_providers import (
    BaseEmbeddingsProvider,
)
from .exception import store_exception
from .magics import AiMagics

# expose ClouderaCopilotPersona on the package root
# required by `jupyter-ai`.
from .models.persona import ClouderaCopilotPersona, Persona

# expose model providers on the package root
from .providers import (
    BaseProvider,
)


def load_ipython_extension(ipython):
    ipython.register_magics(AiMagics)
    ipython.set_custom_exc((BaseException,), store_exception)


def unload_ipython_extension(ipython):
    ipython.set_custom_exc((BaseException,), ipython.CustomTB)
