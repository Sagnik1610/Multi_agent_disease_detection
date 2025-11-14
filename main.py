#!/usr/bin/env python3
"""
Updated main.py with Krutrim integration and async-safe Supervisor.run handling.
Fixed to properly handle message conversion for Krutrim API.
"""

import os
import time
import json
import argparse
import os.path as osp
import asyncio
import inspect
from tqdm import tqdm

# autogen agent class
try:
    from autogen_agentchat.agents import AssistantAgent
except Exception:
    raise

# Krutrim model client adapter
from utils.krutrim_client import KrutrimModelClient

# helper utils from repo
from utils import (
    MedDataset,
    get_doc_system_message,
    get_supervisor_system_message,
    get_inital_message,
    prase_json,
)

# ---------- arg parsing ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Medagents Setting")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_list.json",
        help="the llm models' config file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="DeepSeek-R1-Llama-8B",
        choices=["x_gpt35_turbo", "x_gpt4_turbo", "x_gpt4o", "DeepSeek-R1-Llama-8B"],
        help="the llm models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="rare_disease_302",
        choices=["rare_disease_302"],
        help="choice different dataset",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="inital",
        choices=["inital", "follow_up"],
        help="choice different stages",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="choice different stages",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="log file",
    )
    parser.add_argument("--num_doctors", type=int, default=3, help="number of experts")
    parser.add_argument("--n_round", type=int, default=13, help="attempt_vote")

    args = parser.parse_args()
    return args


# ---------- small helpers ----------
def _safe_prompt_tokens_from_model_client(model_client):
    """
    Returns prompt_tokens from model_client.total_usage() safely (no exceptions).
    """
    try:
        usage = None
        if hasattr(model_client, "total_usage"):
            usage = model_client.total_usage()
        elif hasattr(model_client, "actual_usage"):
            usage = model_client.actual_usage()

        if usage is None:
            return 0
        
        # Try to get prompt_tokens
        try:
            return int(getattr(usage, "prompt_tokens", 0) or 0)
        except Exception:
            try:
                return int(usage.get("prompt_tokens", 0) or 0)
            except Exception:
                return 0
    except Exception:
        return 0


def _normalize_chat_result(chat_result):
    """
    Normalize various possible return shapes from AssistantAgent.run(...) into a simple object
    that has a .messages attribute (list of dicts with role/content).
    """
    from types import SimpleNamespace

    # If coroutine - shouldn't reach here; ensure awaited earlier
    if inspect.iscoroutine(chat_result):
        chat_result = asyncio.run(chat_result)

    # If object with .messages
    if hasattr(chat_result, "messages"):
        return chat_result

    # If dict with 'messages' key
    if isinstance(chat_result, dict) and "messages" in chat_result:
        return SimpleNamespace(messages=chat_result["messages"])

    # If it's a list (assume it's the messages list)
    if isinstance(chat_result, list):
        return SimpleNamespace(messages=chat_result)

    # If it's a string, wrap as a single assistant message
    if isinstance(chat_result, str):
        return SimpleNamespace(messages=[{"role": "assistant", "content": chat_result}])

    # Fallback: try to convert to string and wrap
    try:
        s = str(chat_result)
        return SimpleNamespace(messages=[{"role": "assistant", "content": s}])
    except Exception:
        return SimpleNamespace(messages=[])


# ---------- main case processing ----------
def process_single_case(args, dataset, idx, output_dir, model_config):
    """Process a single medical case through the multi-agent system."""
    case_cost = 0.0
    case_info = {}

    try:
        # Extract case information
        (
            case_type,
            case_name,
            case_crl,
            case_initial_presentation,
            case_follow_up_presentation,
        ) = dataset[idx]

        json_name = f"{case_crl}.json"
        conversation_name = f"{case_crl}_conversation.json"
        identify = f"{args.num_doctors}-{args.n_round}"

        output_dir_path = osp.join(
            output_dir,
            "MAC",
            args.stage,
            args.model_name,
            identify,
            str(args.times),
        )

        if not osp.exists(output_dir_path):
            os.makedirs(output_dir_path)

        # Check if already processed
        file_names = os.listdir(output_dir_path)
        json_files = [file for file in file_names if file.endswith(".json")]

        if json_name in json_files and conversation_name in json_files:
            print(f"Case {idx} already processed. Skipping.")
            return

        # Get case presentation based on stage
        if args.stage == "inital":
            case_presentation = case_initial_presentation
        elif args.stage == "follow_up":
            case_presentation = case_follow_up_presentation
        else:
            raise NotImplementedError(f"Stage {args.stage} not implemented")

        # Prepare model config
        model_config_copy = model_config.copy()
        model_config_copy.setdefault("model", args.model_name)
        model_config_copy.setdefault("base_url", "krutrim_cloud")

        # Create Krutrim client
        print(f"Processing case {idx}: {case_crl}")
        krutrim_client = KrutrimModelClient(config=model_config_copy)

        # Create doctor agents
        Docs = []
        for index in range(args.num_doctors):
            name = f"Doctor{index}"
            doc_system_message = get_doc_system_message(doctor_name=name, stage=args.stage)

            Doc = AssistantAgent(
                name=name,
                model_client=krutrim_client,
                system_message=doc_system_message,
            )
            Docs.append(Doc)

        # Create supervisor agent
        supervisor_system_message = get_supervisor_system_message(
            stage=args.stage, use_specialist=False
        )

        Supervisor = AssistantAgent(
            name="Supervisor",
            model_client=krutrim_client,
            system_message=supervisor_system_message,
        )

        # Get initial message
        inital_message = get_inital_message(
            patient_history=case_presentation, stage=args.stage
        )

        # Run the supervisor agent
        print(f"  Running multi-agent conversation for case {idx}...")
        try:
            run_call = Supervisor.run(task=inital_message)
            if inspect.iscoroutine(run_call):
                chat_result_raw = asyncio.run(run_call)
            else:
                chat_result_raw = run_call
        except Exception as e:
            raise RuntimeError(f"Supervisor.run failed: {str(e)}") from e

        # Normalize the result
        chat_result = _normalize_chat_result(chat_result_raw)

        # Calculate case cost
        for agent in Docs + [Supervisor]:
            try:
                mc = (
                    getattr(agent, "_model_client", None)
                    or getattr(agent, "model_client", None)
                    or getattr(agent, "client", None)
                )
                if mc is None:
                    continue
                case_cost += _safe_prompt_tokens_from_model_client(mc)
            except Exception:
                continue

        # Save conversation
        conversation_path = osp.join(output_dir_path, conversation_name)
        try:
            with open(conversation_path, "w", encoding='utf-8') as file:
                json.dump(chat_result.messages, file, indent=4, ensure_ascii=False)
            print(f"  Saved conversation to {conversation_name}")
        except Exception as e:
            print(f"Warning: Failed to write conversation file: {e}")

        # Extract critic output
        critic_output = [
            item
            for item in chat_result.messages
            if item.get("name") is None
            and '"Most Likely Diagnosis":' in item.get("content", "")
        ]

        if not critic_output:
            # Fallback: find any message with the diagnosis phrase
            for item in chat_result.messages:
                if '"Most Likely Diagnosis":' in item.get("content", ""):
                    critic_output.append(item)

        if not critic_output:
            print(f"Warning: No diagnosis output found for case {idx}")
            syn_report = ""
        else:
            syn_report = critic_output[-1]["content"]

        # Parse JSON output
        json_output = {}
        if syn_report:
            try:
                json_output = prase_json(syn_report)
            except Exception as e:
                print(f"Warning: Failed to parse JSON: {e}")
                json_output = {}

        # Build case info
        case_info["Type"] = case_type
        case_info["Crl"] = case_crl
        case_info["Cost"] = case_cost
        case_info["Presentation"] = case_presentation
        case_info["Name"] = case_name
        case_info["Most Likely"] = json_output.get("Most Likely Diagnosis")
        case_info["Other Possible"] = json_output.get("Differential") or json_output.get(
            "Differential Diagnosis"
        )

        if args.stage == "inital":
            case_info["Recommend Tests"] = json_output.get(
                "Recommend Tests"
            ) or json_output.get("Recommended Tests")

        # Save case info
        recorder_path = osp.join(output_dir_path, json_name)
        try:
            with open(recorder_path, "w", encoding='utf-8') as file:
                json.dump(case_info, file, indent=4, ensure_ascii=False)
            print(f"  Saved case info to {json_name}")
        except Exception as e:
            print(f"Warning: Failed to write case info file: {e}")

        print(f"Successfully processed case {idx}")

    except Exception as e:
        print(f"Error processing case {idx}: {str(e)}")
        raise


# ---------- main ----------
def main():
    args = parse_args()

    # Build model config
    model_config = {
        "model": args.model_name,
        "base_url": "krutrim_cloud",
    }

    # Check for API key
    if not (os.getenv("KRUTRIM_CLOUD_API_KEY") or os.getenv("KRUTRIM_API_KEY")):
        print(
            "WARNING: KRUTRIM_CLOUD_API_KEY (or KRUTRIM_API_KEY) not found in environment."
        )
        print("Please set the environment variable before running:")
        print("  export KRUTRIM_CLOUD_API_KEY='your-api-key-here'")
        print("\nContinuing anyway, but API calls may fail...\n")

    # Load dataset
    dataset = MedDataset(dataname=args.dataset_name)
    data_len = len(dataset)
    output_dir = args.output_dir

    print(f"Starting processing of {data_len} cases...")
    print(f"Model: {args.model_name}")
    print(f"Stage: {args.stage}")
    print(f"Number of doctors: {args.num_doctors}")
    print(f"Number of rounds: {args.n_round}")
    print("-" * 60)

    # Process each case
    success_count = 0
    failure_count = 0

    for idx in tqdm(range(data_len), desc="Processing cases"):
        try:
            process_single_case(args, dataset, idx, output_dir, model_config)
            success_count += 1
        except Exception as e:
            print(f"\nFailed to process case {idx}: {str(e)}\n")
            failure_count += 1
            continue

    print("-" * 60)
    print(f"Processing complete!")
    print(f"Successful: {success_count}/{data_len}")
    print(f"Failed: {failure_count}/{data_len}")


if __name__ == "__main__":
    main()
