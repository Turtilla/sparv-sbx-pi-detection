from sparv.api import Annotation, Output, annotator, get_logger, Config
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json, langchain_ollama, torch, pydantic
from typing import List

logger = get_logger(__name__)

@annotator("Annotate the data with PI entities.", 
           language=["swe"],
           config = [
               Config("sbx_pi_detection.annotation_level", default="detailed_iob", description="Annotation type (and, consequently, model) to use."),
               Config("sbx_pi_detection.generative_model", default="None", description="Generative model to use for annotation. Requires Ollama installed. Default: None. Renders annotation_level irrelevant. Requires a specified prompt_path and batch_size."),
               Config("sbx_pi_detection.prompt_path", default="None", description="Path to a .txt file containing the system prompt to be used to obtain the annotation. Requires Ollama installed. Default: None. Renders annotation_level irrelevant. Requires a specified generative_model and batch_size."),
               Config("sbx_pi_detection.batch_size", default="None", description="Batch size for the LLM processing. Requires Ollama installed. Default: None. Renders annotation_level irrelevant. Requires a specified generative_model and prompt_path.")
           ]
           )  

def pi_detection(
    word: Annotation = Annotation("<token:word>"),
    out: Output = Output("<token>:sbx_pi_detection.pi", description="The type of personal information that the token has: O if it is not personal information or one of the tags defined by the model or prompt selected via the configuration."),
    annotation_level: str = Config("sbx_pi_detection.annotation_level"),
    generative_model: str = Config("sbx_pi_detection.generative_model"),
    prompt_path: str = Config("sbx_pi_detection.prompt_path"),
    batch_size: int = Config("sbx_pi_detection.batch_size")
    ):
    if generative_model != "None" and prompt_path != "None" and batch_size != "None":
        # get LLM predictions
        predictions = get_llm_preds([val for val in word.read()], generative_model, prompt_path, batch_size)
        # write output
        out.write(predictions)
    else:
        # get the model path when the model comes with the plugin
        # set up tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODELS[annotation_level])
            model = AutoModelForTokenClassification.from_pretrained(MODELS[annotation_level])
        except:
            logger.warning(f"The requested annotation type {annotation_level} does not exist. Defaulting to detailed_iob.")
            tokenizer = AutoTokenizer.from_pretrained(MODELS['detailed_iob'])
            model = AutoModelForTokenClassification.from_pretrained(MODELS['detailed_iob'])

        # chunk the input to max model length
        subsplits = trim_to_max_len([val for val in word.read()], tokenizer)  
        # get model predictions for subword tokens
        predictions = [get_preds(subsplit, tokenizer, model) for subsplit in subsplits]
        # flatten
        predictions = [pred for subsplit in predictions for pred in subsplit]
        # write predictions out
        out.write([p[1] for p in predictions])

def trim_to_max_len(
    text: list,
    tokenizer: AutoTokenizer,
    max_len: int = 512,
    ):
    max_len -= tokenizer.num_special_tokens_to_add()
    subsplits = []
    current_subsplit = []
    current_len = 0
    
    # sum lengths of tokenized tokens until the maximum, split
    for element in text:
        tokenized = tokenizer.tokenize(element)
        current_sublen = len(tokenized)
        if current_sublen == 0:
            continue
        elif (current_len + current_sublen) > max_len:
            subsplits.append(current_subsplit)
            current_subsplit = [element]
            current_len = current_sublen   
        else:  # it's within the length
            current_subsplit.append(element)
            current_len += current_sublen
    subsplits.append(current_subsplit)
       
    return subsplits

def get_preds(
    text: list,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
):
    # get model predictions
    inputs = tokenizer(text, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
    
    # realign predictions to words
    words_and_preds = []
    current_word_idx = 0
    current_preds = []
    for i, orig_word_idx in enumerate(inputs.word_ids()):
        if orig_word_idx == None:
            continue
        elif orig_word_idx == current_word_idx:
            current_preds.append(predicted_token_class[i])
        else:  # next word
            # close off the previous word
            words_and_preds.append((current_word_idx, current_preds))
            # next word
            current_word_idx = orig_word_idx
            current_preds = [predicted_token_class[i]]
    words_and_preds.append((current_word_idx, current_preds))
    
    # subword token predictions -> token predictions heuristic
    one_pred_per_word = []
    for item in zip(text, words_and_preds):
        word_predictions = list(set(item[1][1]))
        if len(word_predictions) > 1:
            if 'O' in word_predictions:
                word_predictions.remove('O')  # replace with another if needed
            word_predictions = [word_predictions[0]]  # the final prediction is the first one from the set left after removing O if multiple predictions per word are present, assuming that the first element is more "core" to the word's meaning
        one_pred_per_word.append((item[0], word_predictions[0]))
         
    return one_pred_per_word


def get_llm_preds(text: list, generative_model: str, prompt_path: str, batch_size: int):
    # define output shape
    AnnotationResponse = pydantic.RootModel[List[str]]
    # load prompt    
    with open(prompt_path) as f:
        system_prompt = f.read()
    # define LLM    
    llm = langchain_ollama.ChatOllama(
        model=generative_model,
        temperature=0,
        num_predict=2048,
        num_ctx=4096
    )
    structured_llm = llm.with_structured_output(AnnotationResponse)
    # set up variables    
    sent_len = len(text)
    all_anns = []
    # iterate over batches
    for i in range(0, sent_len, batch_size):
        # set up batch inputs
        inputs = text[i:i+batch_size]
        indexed_batch = [{"id": i + idx, "token": t} for idx, t in enumerate(inputs)]
        # set up prompt messages
        messages = [
                {
                    "role": "system",
                    "content": (
                        system_prompt,
                        f"You MUST generate exactly {len(inputs)} items in the output array. "
                        "Do not skip any tokens in this batch."
                    )
                },
                {
                    "role": "user",
                    "content": f"Annotate this batch: {json.dumps(indexed_batch)}"
                }
            ]    
        try:
            # get preds
            output = structured_llm.invoke(messages)
            anns = output.root
            # fill missing values
            if len(anns) < batch_size:
                anns.append(['ANNOTATION_ERROR']*(batch_size-len(anns)))
            all_anns.extend(anns[:batch_size])
            logger.info(f"Successfully processed tokens {i} to {i + len(inputs)}")
        except Exception as e:
            # fill missing values
            logger.warning(f"Error processing batch starting at index {i}: {e}")
            anns = ['ANNOTATION_ERROR']*batch_size
            all_anns.extend(anns)
    return all_anns[:sent_len]

MODELS = {
    'basic': 'sbx/KB-bert-base-swedish-cased_PI-detection-basic',
    'basic_iob': 'sbx/KB-bert-base-swedish-cased_PI-detection-basic-iob',
    'detailed': 'sbx/KB-bert-base-swedish-cased_PI-detection-detailed',
    'detailed_iob': 'sbx/KB-bert-base-swedish-cased_PI-detection-detailed-iob',
    'general': 'sbx/KB-bert-base-swedish-cased_PI-detection-general',
    'general_iob': 'sbx/KB-bert-base-swedish-cased_PI-detection-general-iob'
}