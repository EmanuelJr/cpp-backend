from pydantic import Field

openai_v1_tag = "OpenAI V1"
cpp_tag = "cpp Backend"

# General Settings
model_field = Field(
    default=None, description="Specifies the model to use for generating completions."
)

max_tokens_field = Field(
    default=16, ge=1, description="The maximum number of tokens to generate."
)

min_tokens_field = Field(
    default=0,
    ge=0,
    description=(
        "The minimum number of tokens to generate. Fewer tokens may be returned "
        "if conditions like max_tokens or stop are met."
    ),
)

stop_field = Field(
    default=None,
    description="A list of tokens where generation will stop. If None, no stop tokens are applied.",
)

stream_field = Field(
    default=False,
    description="If True, streams the results as they are generated. Useful for applications like chatbots.",
)

grammar_field = Field(
    default=None, description="A CBNF grammar string used to format the model's output."
)

# Sampling Parameters
temperature_field = Field(
    default=0.8,
    description=(
        "Controls the randomness of the generated text. Higher values (e.g., 1.5) produce "
        "more random and creative output, while lower values (e.g., 0.5) make the output "
        "more focused and deterministic. A value of 0 selects the most likely token at each step."
    ),
)

top_p_field = Field(
    default=0.95,
    ge=0.0,
    le=1.0,
    description=(
        "Limits next-token selection to a subset with cumulative probability of top_p. "
        "Higher values (e.g., 0.95) allow for more diverse text, while lower values "
        "(e.g., 0.5) focus on more probable tokens."
    ),
)

top_k_field = Field(
    default=40,
    ge=0,
    description=(
        "Limits next-token selection to the top_k most probable tokens. Higher values "
        "(e.g., 100) increase diversity, while lower values (e.g., 10) make the output more focused."
    ),
)

min_p_field = Field(
    default=0.05,
    ge=0.0,
    le=1.0,
    description=(
        "Sets a minimum probability threshold for token selection relative to the most probable token. "
        "Tokens with probability below min_p * (max token probability) are filtered out."
    ),
)

# Penalty Parameters
repeat_penalty_field = Field(
    default=1.1,
    ge=0.0,
    description=(
        "Applies a penalty to tokens that have already been generated, discouraging repetition. "
        "Higher values penalize repetitions more strongly."
    ),
)

presence_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description=(
        "Penalizes tokens based on their presence in the text so far. Positive values encourage "
        "the model to introduce new topics."
    ),
)

frequency_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description=(
        "Penalizes tokens based on their frequency in the text so far. Positive values reduce "
        "the likelihood of repeating the same lines."
    ),
)

# Mirostat Parameters
mirostat_mode_field = Field(
    default=0,
    ge=0,
    le=2,
    description="Enables the Mirostat algorithm for constant perplexity (1 or 2). Set to 0 to disable.",
)

mirostat_tau_field = Field(
    default=5.0,
    ge=0.0,
    le=10.0,
    description=(
        "Sets the Mirostat target entropy (tau), corresponding to the target perplexity. "
        "Lower values produce more focused text; higher values increase diversity."
    ),
)

mirostat_eta_field = Field(
    default=0.1, ge=0.001, le=1.0, description="Sets the Mirostat learning rate (eta)."
)
