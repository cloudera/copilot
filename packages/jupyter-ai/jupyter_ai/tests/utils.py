from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, NamedTuple

from jupyter_ai_magics.providers import BaseProvider, BedrockChatProvider, BedrockProvider

class PromptType(Enum):
    """Enumerates the different types of prompts"""
    IDENTITY = 'identity'
    CODE_COMPLETION = 'code-completion'
    EXPLAIN_ERROR = 'explain-error'
    DEBUG_CODE = 'debug-code'

class PromptData(NamedTuple):
    """Holds the prompt data for each prompt along with the expected response"""
    prompt: str
    prompt_type: PromptType
    expected_response: str

# To add a new model provider, edit the following enum as well as the get_provider and process_chunk functions
class ModelProvider(Enum):
    """Enumerates the different model providers"""
    BEDROCK = 'bedrock'
    BEDROCK_CHAT = 'bedrock-chat'

def get_provider(model_provider: ModelProvider) -> type[BaseProvider]:
    """
    Map each model provider to a different JuPyteR Provider class.
    
    Args:
        model_provider (ModelProvider): The model provider enum value.
    
    Returns:
        type[BaseProvider]: The provider class.
    """
    match model_provider:
        case ModelProvider.BEDROCK:
            return BedrockProvider
        case ModelProvider.BEDROCK_CHAT:
            return BedrockChatProvider

def process_chunk(model_provider: ModelProvider, chunk: Any) -> str:
    """
    Extract the string from each streaming chunk depending on the model provider.
    
    Args:
        model_provider (ModelProvider): The model provider enum value.
        chunk (Any): The chunk of data to process.
    
    Returns:
        str: The processed chunk content.
    """
    match model_provider:
        case ModelProvider.BEDROCK:
            return chunk
        case ModelProvider.BEDROCK_CHAT:
            return chunk.content

@dataclass
class StreamingData:
    """
    Stores all the data that will be saved to the CSV file.
    
    This includes prompt details, the response, timing details, any encountered errors, and the user rating of the response.

    Note that there is no problem to manually change/set the user ratings directly within the CSV file after data appears there.
    """
    prompt: str
    prompt_type: PromptType
    model_id: str
    model_provider: ModelProvider
    generation: str = ''
    time_to_first_chunk_ns: int = -1
    total_response_time_ns: int = -1
    num_generation_chunks: int = 0
    rating: int = -1
    errors_in_jupyterai: bool = False
    error_message: str = ''
    # The following fields are generated based off the above fields
    total_generation_words: int = field(init=False)
    chunk_generation_rate: float = field(init=False)
    word_generation_rate: float = field(init=False)

    def populate(self):
        self.total_generation_words = len(self.generation.split())

        self.chunk_generation_rate = self.num_generation_chunks / self.total_response_time_ns * 1E9
        self.word_generation_rate = self.total_generation_words / self.total_response_time_ns * 1E9

        # Handle newlines in the generation field before saving to CSV
        self.prompt = self.prompt.encode('unicode_escape').decode()
        self.generation = self.generation.encode('unicode_escape').decode()
        self.error_message = self.error_message.encode('unicode_escape').decode()

    def get_prompt(self) -> str:
        """
        Get the prompt, decoding any unicode escapes.
        
        Returns:
            str: The decoded prompt.
        """
        return self.prompt.encode().decode('unicode_escape')
    
    def get_generation(self) -> str:
        """
        Get the generation, decoding any unicode escapes.
        
        Returns:
            str: The decoded generation.
        """
        return self.generation.encode().decode('unicode_escape')

class Gen:
    """
    Wrapper class for a generator that stores the return value after iteration.
    
    Attributes:
        gen (Generator[str, None, int]): The generator to wrap.
        value (int): The final value yielded by the generator.
    """
    def __init__(self, gen: Generator[str, None, int]):
        self.gen = gen

    def __iter__(self) -> Generator[str, None, int]:
        self.value = yield from self.gen
        return self.value

PROMPTS_TO_EVALUATE: list[PromptData] = [
    PromptData('Who are you?', PromptType.IDENTITY, 'It should tell you that it is the Cloudera Copilot. It should not say another name or start talking with itself. It may tell you about itself but only if it stays on topic. Unacceptable: talks with itself, Low Quality: Incorrect information, Helpful: Announces something correct (assistant, Claude), Human: Says it is Cloudera Copilot'),
    PromptData('''Could you help me find the problem in my code? It happens when I try creating a heatmap:

# Heatmap of correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()

And here is the error:

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[5], line 3
      1 # Heatmap of correlation matrix
      2 plt.figure(figsize=(10, 6))
----> 3 sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
      4 plt.title('Heatmap of Correlation Matrix')
      5 plt.show()

File ~/anaconda3/envs/jupyter-ai/lib/python3.11/site-packages/pandas/core/frame.py:11049, in DataFrame.corr(self, method, min_periods, numeric_only)
  11047 cols = data.columns
  11048 idx = cols.copy()
> 11049 mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11051 if method == "pearson":
  11052     correl = libalgos.nancorr(mat, minp=min_periods)

File ~/anaconda3/envs/jupyter-ai/lib/python3.11/site-packages/pandas/core/frame.py:1993, in DataFrame.to_numpy(self, dtype, copy, na_value)
   1991 if dtype is not None:
   1992     dtype = np.dtype(dtype)
-> 1993 result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994 if result.dtype is not dtype:
   1995     result = np.asarray(result, dtype=dtype)

File ~/anaconda3/envs/jupyter-ai/lib/python3.11/site-packages/pandas/core/internals/managers.py:1694, in BlockManager.as_array(self, dtype, copy, na_value)
   1692         arr.flags.writeable = False
   1693 else:
-> 1694     arr = self._interleave(dtype=dtype, na_value=na_value)
   1695     # The underlying data was copied within _interleave, so no need
   1696     # to further copy if copy=True or setting na_value
   1698 if na_value is lib.no_default:

File ~/anaconda3/envs/jupyter-ai/lib/python3.11/site-packages/pandas/core/internals/managers.py:1753, in BlockManager._interleave(self, dtype, na_value)
   1751     else:
   1752         arr = blk.get_values(dtype)
-> 1753     result[rl.indexer] = arr
   1754     itemmask[rl.indexer] = 1
   1756 if not itemmask.all():

ValueError: could not convert string to float: 'Iris-setosa'
<Figure size 1000x600 with 0 Axes>''', PromptType.DEBUG_CODE, """Correct response is either dropping the 'class' column, dropping any non-numeric columns, selecting numeric columns, or (advanced) encoding the categorical data into numerical data using something like pd.get_dummies(df). Human Level: Correct response, Helpful: suggests steps that may help uncover correct response, but does not contain correct response, Beyond: All of: correct response + code + detailed explanation / passes understanding. It should not say that there is a _column_ named Iris-setosa, rather a value."""),
    PromptData("""Please help me complete my code, replying with nothing but code to copy-paste:

# Bar plot of total sales by store and product
plt.figure(figsize=(12, 8))
sns.barplot(x='store', y='sales', hue='product', data=grouped_data, palette='viridis')
plt.title('Total Sales by Store and Product')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.show()

# Bar plot of average profit by store and product""", PromptType.CODE_COMPLETION, 'It should complete the code with no commentary, following a very similar pattern to before. It may either try calculating the average itself, or assume that the profit is already averaged. Beyond: includes ```python. Helpful: Includes extraneous text in addition to the correct response.'),
 PromptData("""Why is this happening?
            
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[1], line 2
      1 # Display basic information about the dataset
----> 2 stores.info()
      4 # Display summary statistics
      5 stores.describe()

NameError: name 'stores' is not defined""", PromptType.EXPLAIN_ERROR, 'It should explain to you that the stores dataset is not yet defined, and there is likely another cell I forgot to run first that defines it. Helpful: Gives advice on how to define stores without suggesting you forgot to run / add code, Human Level: Should suggest that you forgot to run some code. Beyond: Avoid excessive information and suggestions.'),
    PromptData("""I don't understand this error:
               
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: Can't get attribute 'worker' on <module '__main__' (built-in)>
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: Can't get attribute 'worker' on <module '__main__' (built-in)>
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: Can't get attribute 'worker' on <module '__main__' (built-in)>
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: Can't get attribute 'worker' on <module '__main__' (built-in)>
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/ofek.gila/anaconda3/lib/python3.11/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: Can't get attribute 'worker' on <module '__main__' (built-in)>""", PromptType.EXPLAIN_ERROR, "It should say something about how JupyterLab handles multiprocessing weirdly, so the python multiprocessing library won't work. It should instead recommend some other library. Helpful: Anything reasonable. Human Level+ should mention JupyterLab, or using Pool or something similar.")
]
