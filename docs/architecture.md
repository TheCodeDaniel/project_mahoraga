# Architecture

## Critic

The Critic component is responsible for evaluating claims made by the AI. Given an input claim, it investigates whether the claim is factually accurate by consulting live or cached sources. It returns a structured verdict including a boolean correctness flag, a confidence score, a list of supporting sources, and an optional correction string if the claim is found to be wrong. This component is the entry point of the adaptation pipeline.

## Trigger

The Trigger component decides whether the adaptation pipeline should activate based on the Critic's output and contextual signals. It considers factors such as the confidence score returned by the Critic, whether the user has manually flagged the response, and any configured thresholds. Its output is a simple boolean gate that controls whether the Orchestrator should open a case and proceed with adaptation.

## Adapter

The Adapter component applies the correction produced by the Critic to the underlying model. Using parameter-efficient fine-tuning techniques (such as LoRA via PEFT), it updates the model weights or adapter layers so that the same mistake is less likely to recur. It receives a correction dictionary and returns a boolean indicating whether the update was successfully applied.

## Orchestrator

The Orchestrator (MaestroHook) coordinates the full adaptation lifecycle. When the Trigger fires, the Orchestrator opens a case, assigns it a unique ID, and manages the flow of data between the Critic, Trigger, and Adapter components. It acts as the central nervous system of Mahoraga, ensuring that each detected failure is tracked, investigated, and resolved in a structured and auditable way.
