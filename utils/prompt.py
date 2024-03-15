import openai
import random 

def create_prompt_instance(symptoms, n): 
    """
    Creates a user prompt instance 

    symptoms: dict containing configuration of each symptom, describing the patient encounter
    n: the number of clinical notes requested

    returns: string to be used as user message in prompt
    """

    instr = f"Create {n} short clinical note(s) related to the following patient encounter. Not all symptoms must be explicitly mentioned. Do not include any suspicions of possible diagnoses in the clinical note. You can imagine additional context or details described by the patient.\n" 
    symptom_list = ""
    for sym, val in symptoms.items():
        symptom_list += f"- {sym}: {val}\n"
    
    return instr + symptom_list

def prompt_known_combo(req_symptoms, examples, n_req, mask_neg, temp, freq_pen, debug=False):
    """
    Creates the full prompt for a known symptom combination. 
    See example: https://platform.openai.com/playground/p/poSdvoy9dipYIwVPXepRrUyL?model=gpt-3.5-turbo&mode=chat

    req_symptoms: dict containing symptom configuration
    examples: list of clinical note examples related to this symptom configuration
    n_req: number of notes requested 
    mask_neg: whether to leave out the absent symptoms from the prompt
    temp: OpenAI temperature parameter
    freq_pen: OpenAI frequency penalty parameter

    returns
        messages: the prompt sent to OpenAI
        resp_split: list of clinical notes, obtained by splitting response
        response: full response without splitting (for inspection purposes)
    """

    messages = []

    system_message = {"role": "system", "content": "You are a general practitioner with little time, and need to summarize the patient encounter in a short clinical note."}
    messages.append(system_message)

    # first show 1 or more examples for requested symptom combination
    user_message = create_prompt_instance(req_symptoms, len(examples))
    messages.append({"role": "user", "content": user_message}) # example request
    assistant_message = ""
    for i in range(len(examples)): 
        ex = examples[i]
        assistant_message += f"{i+1}. {ex}\n" 
    messages.append({"role": "assistant", "content": assistant_message}) # example response

    if mask_neg: 
        pos_symptoms = {key:value for key, value in req_symptoms.items() if value not in ["no", "none"]}
        user_message = create_prompt_instance(pos_symptoms, n_req)
        messages.append({"role": "user", "content": user_message})
    else: 
        user_message = create_prompt_instance(req_symptoms, n_req)
        messages.append({"role": "user", "content": user_message})

    if debug:
        
        # for message in messages: 
        #     role = message["role"]
        #     content = message["content"]
        #     print(f"{role}: {content}")
        #     print("-------------")

        fake_response = ""
        for i in range(n_req):
            fake_response += f"{i}. bla{i}\n"

        return messages, [" ".join(r.split(" ")[1:]) for r in fake_response.split("\n")[:-1]], fake_response
    
    else:

        res = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        temperature = temp, 
        frequency_penalty=freq_pen)

        response = res["choices"][0]["message"]["content"]

        resp_split = [" ".join(r.split(" ")[1:]) for r in response.split("\n")]

        return messages, resp_split, response

def prompt_unknown_combo(req_symptoms, examples, n_req, mask_neg, temp, freq_pen, debug=False):
    """
    Creates the full prompt for an unknown symptom combination. 
    See example: https://platform.openai.com/playground/p/6KOm6pP6DXmMwDxUJWdGGHP1?model=gpt-3.5-turbo&mode=chat 

    req_symptoms: dict containing symptom configuration
    examples: random subset of dataframe of clinical note examples related to different symptom configurations
    n_req: number of notes requested 
    mask_neg: whether to leave out the absent symptoms from the prompt
    temp: OpenAI temperature parameter
    freq_pen: OpenAI frequency penalty parameter

    returns
        messages: the prompt sent to OpenAI
        resp_split: list of clinical notes, obtained by splitting response
        response: full response without splitting (for inspection purposes)
    """


    messages = []

    system_message = {"role": "system", "content": "You are a general practitioner with little time, and need to summarize the patient encounter in a short clinical note."}
    messages.append(system_message)

    # first show a couple examples for other symptom combinations
    for i in range(len(examples)):
        sympt = examples.iloc[i].drop(labels=["text"]).to_dict()
        user_message = create_prompt_instance(sympt, 1)
        messages.append({"role": "user", "content": user_message}) # example request
        assistant_message = "1. " + examples.iloc[i]["text"]
        messages.append({"role": "assistant", "content": assistant_message}) # example response

    if mask_neg: 
        pos_symptoms = {key:value for key, value in req_symptoms.items() if value not in ["no", "none"]}
        user_message = create_prompt_instance(pos_symptoms, n_req)
        messages.append({"role": "user", "content": user_message})
    else: 
        user_message = create_prompt_instance(req_symptoms, n_req)
        messages.append({"role": "user", "content": user_message})

    if debug:
        
        # for message in messages: 
        #     role = message["role"]
        #     content = message["content"]
        #     print(f"{role}: {content}")
        #     print("-------------")

        fake_response = ""
        for i in range(n_req):
            fake_response += f"{i}. unkbla{i}\n"

        return messages, [" ".join(r.split(" ")[1:]) for r in fake_response.split("\n")[:-1]], fake_response
    
    else:

        res = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        temperature = temp, 
        frequency_penalty=freq_pen)

        response = res["choices"][0]["message"]["content"]

        resp_split = [" ".join(r.split(" ")[1:]) for r in response.split("\n")]

        return messages, resp_split, response
    
def get_random_examples(df_examples, n_ex):
    """
    Selects random examples from the dataframe of example clinical notes. 

    df_examples: Dataframe containing symptom combinations, accompanied by clinical example notes
    n_ex: number of examples requested

    returns: dataframe with selected examples (subset of df_examples)
    """
    idx = list(range(len(df_examples)))
    sel_idx = random.sample(idx, k=n_ex)
    sel_examples = df_examples.iloc[sel_idx]
    return sel_examples

def prompt_unrelated_strat45(n_req, temp, freq_pen, mention_sympt, debug=False):
    """
    Prompt for unrelated clinical notes, without providing any examples. 
    See example (strategy 4, mention_sympt=False): https://platform.openai.com/playground/p/8KaVpr7chxLMHyeKz6Y2mMVE?model=gpt-3.5-turbo&mode=chat
    See example (strategy 5, mention_sympt=True): https://platform.openai.com/playground/p/eXDqpwMkqcvUA8wWiW1H06et?model=gpt-3.5-turbo&mode=chat 

    n_req: number of requested clinical notes
    temp: OpenAI temperature parameter
    freq_pen: OpenAI frequency penalty parameter
    mention_sympt: whether to mention the symptoms that are not experienced by the patient or not

    returns
        messages: the prompt sent to OpenAI
        resp_split: list of clinical notes, obtained by splitting response
        response: full response without splitting (for inspection purposes)
    """

    messages = []

    system_message = {"role": "system", "content": "You are a general practitioner with little time, and need to summarize the patient encounter in a short clinical note."}
    messages.append(system_message)

    user_message = f"Create {n_req} short clinical notes describing an encounter with a patient in a primary care setting. You may add additional context that seems relevant, as long as it fits a single sentence. Do not include any suspicions of possible diagnoses in the clinical note."
    if mention_sympt: 
        user_message += " The patient does not experience any of the following symptoms: dyspnea, cough, fever, chest pain / pain attributed to airways, sneezing / blocked nose."
    messages.append({"role": "user", "content": user_message}) # request

    if not debug: 
        res = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = messages,
            temperature = temp, 
            frequency_penalty=freq_pen
        )

        response = res["choices"][0]["message"]["content"]

        resp_split = [" ".join(r.split(" ")[1:]) for r in response.split("\n")]

        return messages, resp_split, response
    else: 
        return messages
    
def prompt_unrelated_strat13(n_req, df_examples, n_ex, mention_sympt, temp, freq_pen, debug=False):
    """
    Creates the full prompt for an unknown symptom combination. 
    See example (strategy 1, mention_sympt=True): https://platform.openai.com/playground/p/Zw9y4EZ8RRfGaZTgBCPu5DPT?model=gpt-3.5-turbo&mode=chat
    See example (strategy 3, mention_sympt=False): https://platform.openai.com/playground/p/Ikszv18Zbqr162kF0iSErbCT?model=gpt-3.5-turbo&mode=chat

    n_req: number of notes requested
    df_examples: clinical note examples related to all-negative symptom combination 
    n_ex: number of examples to sample from df_examples
    mention_sympt: whether to mention the symptoms that are not experienced by the patient or not
    temp: OpenAI temperature parameter
    freq_pen: OpenAI frequency penalty parameter

    returns
        messages: the prompt sent to OpenAI
        resp_split: list of clinical notes, obtained by splitting response
        response: full response without splitting (for inspection purposes)
    """

    messages = []

    system_message = {"role": "system", "content": "You are a general practitioner with little time, and need to summarize the patient encounter in a short clinical note."}
    messages.append(system_message)

    examples = get_random_examples(df_examples, n_ex)
    
    user_message = f"Create {n_ex} short clinical notes describing an encounter with a patient in a primary care setting. You may add additional context that seems relevant, as long as it fits a single sentence. Do not include any suspicions of possible diagnoses in the clinical note."
    if mention_sympt: 
        user_message += " The patient does not experience any of the following symptoms: dyspnea, cough, fever, chest pain / pain attributed to airways, sneezing / blocked nose."
    messages.append({"role": "user", "content": user_message}) # example request
    assistant_message = ""
    for i in range(len(examples)):
        assistant_message += f"{i+1}. " + examples.iloc[i]["text"] + "\n"
    messages.append({"role": "assistant", "content": assistant_message}) # example response

    user_message = f"Create {n_req} short clinical notes describing an encounter with a patient in a primary care setting. You may add additional context that seems relevant, as long as it fits a single sentence. Do not include any suspicions of possible diagnoses in the clinical note."
    if mention_sympt: 
        user_message += " The patient does not experience any of the following symptoms: dyspnea, cough, fever, chest pain / pain attributed to airways, sneezing / blocked nose."
    messages.append({"role": "user", "content": user_message}) # request

    if not debug: 
        res = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = messages,
            temperature = temp, 
            frequency_penalty=freq_pen
        )

        response = res["choices"][0]["message"]["content"]

        resp_split = [" ".join(r.split(" ")[1:]) for r in response.split("\n")]

        return messages, resp_split, response
    else: 
        return messages