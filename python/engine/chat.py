"""Interactive multi-turn chat with ChatML formatting."""

import time

import nl_hecate
from engine.generation import (
    generate, chatml_encode_turn, chatml_encode_prompt, IM_END,
)


def run_chat(
    params, cfg, tokenizer, gpu_model,
    max_tokens: int, temperature: float, top_k: int, stateless: bool,
    learn: bool = False, learn_kwargs: dict | None = None,
):
    """
    Interactive multi-turn chat with ChatML formatting.

    Stateful mode (default): CMS memory carries conversation context.
      Only the new user message is fed each turn. Memory accumulates state.
      Prompt size: constant per turn.

    Stateless mode (--stateless): Full conversation history re-sent each turn.
      No CMS memory persistence. Traditional transformer-style chat.
      Prompt size: grows linearly with conversation length.
    """
    seq_len = cfg.seq_len
    history_tokens: list[int] = []  # accumulated token history (stateless mode)
    turn_count = 0

    # Persistent CMS state for stateful mode
    conductor = None
    context = None
    if not stateless:
        conductor = nl_hecate.Conductor(
            cfg.k,
            list(cfg.chunk_sizes) if hasattr(cfg, "chunk_sizes") else [1] * cfg.k,
        )
        if gpu_model is None:
            context = nl_hecate.ContextState(cfg.k, cfg.d_model)
            if params is not None and getattr(cfg, 'projection_kind', 'static') == 'adaptive':
                context.seed_self_ref(params)

    mode_label = "stateless (full history)" if stateless else "stateful (CMS memory)"
    if learn:
        mode_label += " + learning"
    print(f"\n{'\u2500' * 60}")
    print(f"  NL-Hecate Chat")
    print(f"  Mode: {mode_label}")
    print(f"  temp={temperature}, top_k={top_k}, max_tokens={max_tokens}")
    print(f"{'\u2500' * 60}")
    print("  Commands: /quit  /clear  /mode  /stats")
    print(f"{'\u2500' * 60}\n")

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        # ── Slash commands ──
        if stripped.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if stripped.lower() == "/clear":
            history_tokens.clear()
            turn_count = 0
            if not stateless:
                conductor = nl_hecate.Conductor(
                    cfg.k,
                    list(cfg.chunk_sizes) if hasattr(cfg, "chunk_sizes") else [1] * cfg.k,
                )
                if gpu_model is None:
                    context = nl_hecate.ContextState(cfg.k, cfg.d_model)
                    if params is not None and getattr(cfg, 'projection_kind', 'static') == 'adaptive':
                        context.seed_self_ref(params)
            print("  [conversation cleared]\n")
            continue

        if stripped.lower() == "/mode":
            print(f"  Mode: {mode_label}")
            print(f"  History: {len(history_tokens)} tokens, {turn_count} turns")
            if not stateless:
                print(f"  CMS: memory persists across turns (constant prompt size)")
            else:
                print(f"  No memory: full history re-sent each turn")
            print()
            continue

        if stripped.lower() == "/stats":
            print(f"  Turns: {turn_count}")
            print(f"  History tokens: {len(history_tokens)}")
            if not stateless and gpu_model is not None and hasattr(gpu_model, "gate_biases"):
                biases = gpu_model.gate_biases()
                for i, (ba, bt, be) in enumerate(biases):
                    print(f"  Level {i}: b_alpha={ba:.2f} b_theta={bt:.2f} b_eta={be:.2f}")
            print()
            continue

        # ── Build prompt tokens ──
        user_turn = chatml_encode_turn(tokenizer, "user", stripped)
        assistant_start = chatml_encode_prompt(tokenizer, "assistant")

        if stateless:
            history_tokens.extend(user_turn)
            prompt_tokens = history_tokens + assistant_start
            if len(prompt_tokens) > seq_len:
                prompt_tokens = prompt_tokens[-seq_len:]
        else:
            prompt_tokens = user_turn + assistant_start

        # ── Generate ──
        t0 = time.perf_counter()
        if stateless:
            output = generate(
                params, cfg, prompt_tokens, max_tokens, temperature,
                top_k=top_k, stop_token=IM_END, gpu_model=gpu_model,
                learn=learn, learn_kwargs=learn_kwargs,
            )
        else:
            output = generate(
                params, cfg, prompt_tokens, max_tokens, temperature,
                top_k=top_k, stop_token=IM_END, gpu_model=gpu_model,
                conductor=conductor, context=context,
                learn=learn, learn_kwargs=learn_kwargs,
            )
        t1 = time.perf_counter()

        gen_tokens = output[len(prompt_tokens):]
        response_text = tokenizer.decode(gen_tokens).strip()

        if stateless:
            response_turn = chatml_encode_turn(tokenizer, "assistant", response_text)
            history_tokens.extend(response_turn)

        turn_count += 1
        gen_count = len(gen_tokens)
        tps = gen_count / (t1 - t0) if (t1 - t0) > 0 else 0
        prompt_size = len(prompt_tokens)

        print(f"\033[1;32mAssistant:\033[0m {response_text}")
        print(f"  \033[2m[{gen_count} tokens, {tps:.0f} tok/s, "
              f"prompt={prompt_size} tokens]\033[0m\n")
