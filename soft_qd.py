from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.emitters import EvolutionStrategyEmitter
from ribs.visualize import grid_archive_heatmap
import time
import transformers

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from pathlib import Path
from typing import Optional
import torch
import json
import pickle
import time
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from ribs.emitters import EvolutionStrategyEmitter

import matplotlib.pyplot as plt

from ribs.visualize import grid_archive_heatmap

import sys

import tqdm



THIS_DIR = Path(__file__).parent.resolve()
def parse_response(response, default=None):
    """Parse the respones for a JSON object. If not found, return None."""
    try:
        parse = json.loads(response)
        return parse
    except:
        pass
    
    try:
        left = response.find("{")
        right = response.find("}") + 1
        response = response[left:right]

        return json.loads(response)
    except:
        return default

def run_main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    key = "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"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=key, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        token=key,
    )

    # create the archive
    ranges = [(0, 10), (0, 10)]
    dims = (10, 10)
    total_itrs = 32
    num_emitters = 4
    batch_size = 4
    lr = 1
    sigma0 = 0.03# 0.01

    eval_times = 1 # how many times we should evaluate metrics for the solution

    use_random = False # if true, we will use a random solution instead of CMA-ES/ME/MAE solutions

    # misc data
    import datetime
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("results_32_%Y-%m-%d_%H-%M-%S")

    # logging data
    failed_parse = 0 # save the number of failed parses
    history = {
        "time": [],
        "num_elites": [],
        "coverage": [],
        "qd_score": [],
        "max_obj": [],
        "mean_obj": []
    }

    # create the x0
    inject = "Happy"
    full_message = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "create a haiku with theme:"},
        ]


    full_message = tokenizer.apply_chat_template(full_message, tokenize=False, add_generation_prompt=True)
    encoding = tokenizer(full_message, return_offsets_mapping=True)
    input_ids = encoding['input_ids']
    # offset_mapping = encoding['offset_mapping']
    print(full_message)

    # start_char = full_message.find("create a haiku with following theme:<|eot_id|><|start_header_id|>user<|end_header_id|>") + len("create a haiku with following theme:<|eot_id|><|start_header_id|>user<|end_header_id|>") +1
    # end_char = full_message.find("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
    # print(full_message[start_char])
    # print(full_message[end_char])
    # token_indices = []
    # for idx, (start, end) in enumerate(offset_mapping):
    #     if start >= start_char and end <= end_char:
    #         token_indices.append(idx)
    # print("Token Indices for inject:", token_indices)

    # Get the embeddings for inject_input_ids
    full_message_embeddings = model.get_input_embeddings()(torch.tensor(input_ids))
    inject_input_ids = tokenizer(inject, add_special_tokens=False)['input_ids']

    inject_embeddings = model.get_input_embeddings()(torch.tensor(inject_input_ids))

    print(inject_embeddings.size())
    print(full_message_embeddings.size())

    x0 = inject_embeddings.reshape(-1).float().cpu().detach().numpy()
    solution_dim = x0.shape[0]



    # add slight noise
    # soft_prompt = soft_prompt + torch.randn_like(soft_prompt) * 0.05

    archive = GridArchive(solution_dim=solution_dim,
                        dims=dims,
                        ranges=ranges,
                        learning_rate=lr)

    result_archive = GridArchive(solution_dim=solution_dim,
                                dims=dims,
                                ranges=ranges)

    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=x0,
            sigma0=sigma0,
            ranker="2imp",
            selection_rule="mu",
            restart_rule="basic",
            batch_size=batch_size,
        ) for _ in range(num_emitters)
    ]

    scheduler = Scheduler(archive, emitters, result_archive=result_archive)

    def eval(
        task,
        soft_prompt_embeding=None #an embeding
    ):
        #evaluate with huggingface. 
         
        if soft_prompt_embeding is not None:
            """evaluate embeding instead of prompts """
            
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            # Step 4: Generate text using the modified embeddings
            # input_tensor = torch.tensor(soft_prompt_embeding, dtype=torch.float16, device='cuda')
            print(soft_prompt_embeding.shape)
            outputs = model.generate(
                inputs_embeds=soft_prompt_embeding,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            # Step 5: Decode the generated response
            print(outputs)
            response = outputs[0]#[soft_prompt_embeding.shape[1]:]
            decoded_response = tokenizer.decode(response, skip_special_tokens=True)
            print("--------------poem------------")
            print(decoded_response)
            print("--------------poem------------")
            return decoded_response   

        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": task},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            decoded_response = tokenizer.decode(response, skip_special_tokens=True)
            print(decoded_response)
            return decoded_response  


    def quality(
        poem
    ):
        quality_str = f"""{poem}
Rate the quality of the above haiku on a scale from 1 to 10. Answer in JSON with the key 'quality'.
"""
        q_res = eval(quality_str)
        return parse_response(q_res, default={"quality": 0})
    
    def diversity(
        poem
    ):
        # parse diversity
        diversity_str = f"""{poem}
On a scale of 0 to 10, with 0 being tragic and 10 being happy, how would you rate the tone of the poem?
On a scale of 0 to 10, with 0 being romantic and 10 being horror, how would you rate the genre of the poem?

Respond in JSON with the keys 'tone' and 'genre'.
"""
        d_res = eval(diversity_str)
        return parse_response(d_res, default={"tone": 5, "genre": 5})

    start_time = time.time()
    for itr in tqdm.trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):
        solution_batch = scheduler.ask()

        objective_batch = []
        measure_batch = []
        # evaluate all the solutions given
        for s in solution_batch:
            # if random, we don't use the algorithm's embeddings, but instead use random embeddings
            if use_random:
                print("using a embedding with noise instead of CMA-ME")
                soft_prompt = torch.tensor(x0, device=model.device, dtype=model.dtype).reshape(1,  s.shape[0] // 4096, -1)
                soft_prompt = soft_prompt + (torch.randn_like(soft_prompt) * 0.05)
            else:

                 # soft_prompt = torch.tensor(s, device="cuda").reshape(1, 1, -1).to(torch.bfloat16)
                print("concatinating static embedding and inject")
                #concatenate inject and 
                print(s.shape)
                print(len(input_ids))
                soft_prompt = torch.tensor(
                    s, device=model.device, dtype=model.dtype
                ).reshape(1, s.shape[0] // 4096, -1)

            
            full_message_embeddings = torch.tensor(
                full_message_embeddings, device=model.device, dtype=model.dtype
            ).reshape(1,len(input_ids),-1)
            print("Soft prompt shape:", soft_prompt.shape)
            print("full size:",full_message_embeddings.shape)
            # Insert the mutated embeddings back into the current_message_embeddings
            print("concating newly muatated embedding: ")
            insertion_position = 23 # NOTE: this is based on the prompt.
            injected_input_embeddings = torch.cat([full_message_embeddings[:, :insertion_position, :], soft_prompt, full_message_embeddings[:, insertion_position:, :]],dim=1)

            # injected_input_embeddings = torch.cat([full_message_embeddings,soft_prompt], dim=1)
            print(injected_input_embeddings.shape)
            # if use_random:
            #     soft_prompt = soft_prompt + (torch.randn_like(soft_prompt) * 0.05)
            print("evaluating mutated embedding:")

            poem = eval(
                "",
                injected_input_embeddings
            )
            # get the quality of the poem
            q = 0
            t = 0
            g = 0

            success = 0
            
            for _ in range(eval_times):
                # parse quality
#                 quality_str = f"""{poem}
# Rate the quality of the above poem on a scale from 1 to 10. Answer in JSON with the key 'quality'.
# """
#                 q_res = eval([UserMessage(content=quality_str)])
#                 q_json, failed1 = parse_response(q_res, default={"quality": 0})

#                 # parse diversity
#                 diversity_str = f"""{poem}
# On a scale of 0 to 10, with 0 being tragic and 10 being happy, how would you rate the tone of the poem?
# On a scale of 0 to 10, with 0 being romantic and 10 being horror, how would you rate the genre of the poem?

# Respond in JSON with the keys 'tone' and 'genre'.
# """
#                 d_res = eval([UserMessage(content=diversity_str)])
#                 d_json, failed2 = parse_response(d_res, default={"tone": 5, "genre": 5})
                
                # ensure we get the quality and diversity
                q_json = quality(poem)


            
                d_json = diversity(poem)
                t += int(d_json["tone"])
                g += int(d_json["genre"])
                q += int(q_json["quality"])


                # if not (failed1 or failed2):
                #     t += int(d_json["tone"])
                #     g += int(d_json["genre"])
                #     q += int(q_json["quality"])

                #     success += 1
                # else:
                #     failed_parse += 1

            # take the average
            # if success != 0:
            #     q = q / success
            #     t = t / success
            #     g = g / success

            # add aggregation
            objective_batch.append(q)
            measure_batch.append((t, g))

        scheduler.tell(objective_batch, measure_batch)

        # get variables
        time_diff = time.time() - start_time
        num_elites = archive.stats.coverage
        coverage = archive.stats.coverage
        qd_score = archive.stats.qd_score

        max_obj = archive.stats.obj_max
        mean_obj = archive.stats.obj_mean

        # save the data to history
        history["time"].append(time_diff)
        history["num_elites"].append(num_elites)
        history["coverage"].append(coverage)
        history["qd_score"].append(qd_score)

        history["max_obj"].append(max_obj)
        history["mean_obj"].append(mean_obj)

        if itr % 25 == 0:
            # print the results
            tqdm.write(f"> {itr} itrs completed after {time_diff:.2f}s")
            tqdm.write(f"  - Size: {num_elites}")
            tqdm.write(f"  - Coverage: {coverage}")
            tqdm.write(f"  - QD Score: {qd_score}")

            tqdm.write(f"  - Max Obj: {max_obj}")
            tqdm.write(f"  - Mean Obj: {mean_obj}")


    # save contents to file
    with open(f"data/{file_name}.pkl", "wb") as f:
        pickle.dump({
            "archive": archive,
            "failed_parse": failed_parse,
            # "result_archive": result_archive,
        }, f)


def main():

    run_main()

if __name__ == "__main__":
    main()
