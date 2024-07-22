import csv
import dataclasses
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from jupyter_ai.history import BoundedChatHistory # type: ignore
from jupyter_ai.tests.get_rating import GetRating # type: ignore
from jupyter_ai.tests.utils import get_provider, ModelProvider, process_chunk, PROMPTS_TO_EVALUATE, PromptData, StreamingData # type: ignore
from jupyter_ai_magics.providers import BaseProvider, BedrockChatProvider, BedrockProvider
from langchain_core.runnables.history import RunnableWithMessageHistory

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

STREAMING_DATA_FILE_NAME = 'streaming-data'

AVAILABLE_BEDROCK_MODELS = BedrockProvider.chat_models()
AVAILABLE_BEDROCK_CHAT_MODELS = BedrockChatProvider.chat_models()

# Leave commented out. Only uncomment this if you want to test with only one (cheap model)
# AVAILABLE_BEDROCK_MODELS = ['amazon.titan-text-lite-v1']
# AVAILABLE_BEDROCK_CHAT_MODELS: list[str] = []

def get_streaming_file_path(timestamp: datetime) -> Path:
    """
    Generate the file path for saving streaming data based on the given timestamp.
    
    Args:
        timestamp (datetime): The timestamp to include in the file name.
    
    Returns:
        Path: The generated file path.
    """
    return DATA_DIR / f'{STREAMING_DATA_FILE_NAME}-{timestamp.strftime("%Y-%m-%d_%H-%M-%S")}.csv'

def save_streaming_data(streaming_data: StreamingData, timestamp: datetime):
    """
    Save streaming data to a CSV file. Create a new file if it doesn't exist.
    
    Args:
        streaming_data (StreamingData): The streaming data to save.
        timestamp (datetime): The timestamp to include in the file name.
    """
    file_path = get_streaming_file_path(timestamp)

    is_file_new = not file_path.exists()
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = [field.name for field in dataclasses.fields(StreamingData)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_file_new:
            writer.writeheader()
        writer.writerow(dataclasses.asdict(streaming_data))

def create_llm_chain(model_id: str, provider: type[BaseProvider]) -> tuple[RunnableWithMessageHistory, bool]:
    """
    Create a language model chain using the specified model and provider.
    
    Args:
        model_id (str): The identifier of the model.
        provider (type[BaseProvider]): The provider class for the model.
    
    Returns:
        tuple: A tuple containing the language model chain and a boolean indicating if streaming is supported.
    """
    llm = provider(model_id=model_id)
    prompt_template = llm.get_chat_prompt_template()
    runnable = prompt_template | llm # type: ignore

    # I would return the runnable directly, and pass in an empty history as the input
    # This is just to mimic the way jupyterai does it
    return RunnableWithMessageHistory(
        runnable=runnable, # type: ignore
        get_session_history=lambda *args: BoundedChatHistory(k=2),
        input_messages_key="input",
        history_messages_key="history",
    ), llm.supports_streaming

def collect_streaming_data(prompt_data: PromptData, model_id: str, model_provider: ModelProvider) -> StreamingData:
    """
    Collect streaming data by invoking the language model chain with the provided prompt.
    
    Args:
        prompt_data (PromptData): The prompt data to use.
        model_id (str): The identifier of the model.
        model_provider (ModelProvider): The provider of the model.
    
    Returns:
        StreamingData: The collected streaming data.
    """
    llm_chain, supports_streaming = create_llm_chain(model_id, get_provider(model_provider))
    data = StreamingData(
        prompt=prompt_data.prompt,
        prompt_type=prompt_data.prompt_type,
        model_id=model_id,
        model_provider=model_provider,
    )

    def collect_streaming(data: StreamingData):
        """Stream the response from the model, collecting streaming data."""
        start_time = time.time_ns()

        for chunk in llm_chain.stream( # type: ignore
            {'input': data.prompt},
            {'configurable': {'session_id': 'static_session'}},
        ):
            if data.num_generation_chunks == 0:
                data.time_to_first_chunk_ns = time.time_ns() - start_time
            
            data.generation += process_chunk(model_provider, chunk)
            data.num_generation_chunks += 1

        data.total_response_time_ns = time.time_ns() - start_time

    def collect_no_streaming(data: StreamingData):
        """Invoke models directly that don't support streaming, collecting streaming data."""
        start_time = time.time_ns()

        data.generation = llm_chain.invoke( # type: ignore
            {'input': data.prompt},
            {'configurable': {'session_id': 'static_session'}},
        )

        data.time_to_first_chunk_ns = data.total_response_time_ns = time.time_ns() - start_time
        data.num_generation_chunks = -1

    try:
        if supports_streaming:
            # jupyterai's supports_streaming is inaccurate for some models, e.g., for ai21.j2-ultra-v1
            # so if streaming fails despite being 'supported', we also try without streaming to see if it works
            try:
                collect_streaming(data)
            except Exception as e:
                data.errors_in_jupyterai = True
                data.error_message = str(e)
                collect_no_streaming(data)
        else:
            collect_no_streaming(data)
    except Exception as e:
        data.errors_in_jupyterai = True
        data.error_message = str(e)
        print(data.error_message)

    data.populate()

    return data

def update_rating(data: StreamingData, expected: str, index: int, total: int):
    """
    Update the rating of the streaming data based on user input.
    The user will be asked to rate the quality of the response, comparing the response with the expected response.
    
    Args:
        data (StreamingData): The streaming data to update.
        expected (str): The expected response.
        index (int): The index of the current evaluation.
        total (int): The total number of evaluations.
    """
    if data.get_generation() == '':
        return

    gr = GetRating(data, expected, index, total)

    data.rating = gr.start()


def evaluate_prompt(prompt_data: PromptData, timestamp: datetime):
    """
    Evaluate a given prompt using multiple models, prompt the user to rates the models, and then save the streaming data.
    
    Args:
        prompt_data (PromptData): The prompt data to evaluate.
        timestamp (datetime): The timestamp for file naming.
    """
    print(f'Started evaluating prompt:\n{prompt_data.prompt}')

    with ThreadPoolExecutor() as executor:
        # all the tasks in this list will be evaluated in parallel using a thread pool.
        # a vast majority of the time is spent waiting on the models to respond, so this should not hurt the quality of our timing data.
        futures = [
            executor.submit(collect_streaming_data, prompt_data, model_id, ModelProvider.BEDROCK) for model_id in AVAILABLE_BEDROCK_MODELS
        ] + [
            executor.submit(collect_streaming_data, prompt_data, model_id, ModelProvider.BEDROCK_CHAT) for model_id in AVAILABLE_BEDROCK_CHAT_MODELS
        ]

        # in whatever order the threads complete, we will prompt the user to provide rating data and save it.
        # while the user is rating (on the main thread), we will concurrently be collecting streaming data from the other models.
        for i, future in enumerate(as_completed(futures)):
            data = future.result()
            update_rating(data, prompt_data.expected_response, i, len(futures))
            save_streaming_data(data, timestamp)

if __name__ == '__main__':
    """
    Main execution block to evaluate all prompts and save the results.
    """
    timestamp = datetime.now()

    for prompt_data in PROMPTS_TO_EVALUATE:
        evaluate_prompt(prompt_data, timestamp)

    print(f'Finished evaluating all prompts, results are saved in: {get_streaming_file_path(timestamp)}')
    print('For a nice visualisation, create a copy of the JupyterAI Compare Models Google Sheets (https://docs.google.com/spreadsheets/d/1Zwfs3vUrZfYntik44FmhRqZgn-CK6iOn0quDZqtOYw4/edit?usp=sharing) and File -> Import the data into the csv tab')
