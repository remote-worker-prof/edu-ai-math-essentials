"""Общие константы контрактов для проверок notebook-ов EAIME."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NOTEBOOKS = {
    "themes/00-Foundations/examples/01_numpy_sequence_basics.ipynb",
    "themes/00-Foundations/examples/02_minimal_keras_sequence_classifier.ipynb",
    "themes/00-Foundations/examples/03_tokenization_padding_masking.ipynb",
    "themes/00-Foundations/examples/04_attention_heatmap_toy.ipynb",
    "themes/00-Foundations/showcases/01_imdb_many_to_one_showcase.ipynb",
    "themes/00-Foundations/showcases/02_jena_climate_lstm_gru_showcase.ipynb",
    "themes/00-Foundations/showcases/03_spa_eng_seq2seq_attention_showcase.ipynb",
    "themes/01-RNN/lab/01_simple_rnn_many_to_one_toy.ipynb",
    "themes/01-RNN/lab/02_lstm_many_to_many_toy.ipynb",
    "themes/01-RNN/lab/03_gru_seq2seq_reverse_toy.ipynb",
    "themes/01-RNN/lab/solutions/01_simple_rnn_many_to_one_toy_solution.ipynb",
    "themes/01-RNN/lab/solutions/02_lstm_many_to_many_toy_solution.ipynb",
    "themes/01-RNN/lab/solutions/03_gru_seq2seq_reverse_toy_solution.ipynb",
    "themes/02-Attention/lab/01_gru_seq2seq_attention_reverse_toy.ipynb",
    "themes/02-Attention/lab/solutions/01_gru_seq2seq_attention_reverse_toy_solution.ipynb",
    "themes/03-Transformer/lab/01_transformer_encoder_order_toy.ipynb",
    "themes/03-Transformer/lab/02_transformer_encoder_imdb.ipynb",
    "themes/03-Transformer/lab/solutions/01_transformer_encoder_order_toy_solution.ipynb",
    "themes/03-Transformer/lab/solutions/02_transformer_encoder_imdb_solution.ipynb",
    "themes/04-Autoregression/lab/01_decoder_only_causal_toy.ipynb",
    "themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare.ipynb",
    "themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare_gpu.ipynb",
    "themes/04-Autoregression/lab/solutions/01_decoder_only_causal_toy_solution.ipynb",
    "themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb",
    "themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_solution.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_acceptance/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_acceptance/02_decoder_only_tiny_shakespeare_gpu_solution.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_g635_demo/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_g635_demo/02_decoder_only_tiny_shakespeare_gpu_solution.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_no_gpu_expected_fail/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick/02_decoder_only_tiny_shakespeare_gpu_solution.attempt2.gpu_60m_boost.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.attempt2.gpu_60m_boost.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5_tf.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5_tf2.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5_tf3.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5_tf4.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_stability_a/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_tf221_local_gpu_demo/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_tf221_local_gpu_demo/02_decoder_only_tiny_shakespeare_gpu_solution.attempt2.gpu_60m_boost.executed.ipynb",
    "themes/04-Autoregression/lab/solutions/runs_tf221_local_gpu_demo/02_decoder_only_tiny_shakespeare_gpu_solution.executed.ipynb",
    "themes/05-Full-Transformer/lab/01_full_transformer_tiny_shakespeare.ipynb",
    "themes/05-Full-Transformer/lab/solutions/01_full_transformer_wikitext2_solution.ipynb",
    "themes/05-Full-Transformer/lab/solutions/runs_gpu_full/01_full_transformer_wikitext2_solution.gpu_full_allow_errors.executed.ipynb",
    "themes/05-Full-Transformer/lab/solutions/runs_gpu_full_tuned/01_full_transformer_wikitext2_solution.gpu_full_tuned.executed.ipynb",
    "themes/05-Full-Transformer/lab/solutions/runs_gpu_full_tuned2/01_full_transformer_wikitext2_solution.gpu_full_tuned2.executed.ipynb",
    "themes/05-Full-Transformer/lab/solutions/runs_gpu_full_tuned3/01_full_transformer_wikitext2_solution.gpu_full_tuned3.executed.ipynb",
}

STARTER_SOLUTION_PAIRS = (
    (
        "themes/01-RNN/lab/01_simple_rnn_many_to_one_toy.ipynb",
        "themes/01-RNN/lab/solutions/01_simple_rnn_many_to_one_toy_solution.ipynb",
    ),
    (
        "themes/01-RNN/lab/02_lstm_many_to_many_toy.ipynb",
        "themes/01-RNN/lab/solutions/02_lstm_many_to_many_toy_solution.ipynb",
    ),
    (
        "themes/01-RNN/lab/03_gru_seq2seq_reverse_toy.ipynb",
        "themes/01-RNN/lab/solutions/03_gru_seq2seq_reverse_toy_solution.ipynb",
    ),
    (
        "themes/02-Attention/lab/01_gru_seq2seq_attention_reverse_toy.ipynb",
        "themes/02-Attention/lab/solutions/01_gru_seq2seq_attention_reverse_toy_solution.ipynb",
    ),
    (
        "themes/03-Transformer/lab/01_transformer_encoder_order_toy.ipynb",
        "themes/03-Transformer/lab/solutions/01_transformer_encoder_order_toy_solution.ipynb",
    ),
    (
        "themes/03-Transformer/lab/02_transformer_encoder_imdb.ipynb",
        "themes/03-Transformer/lab/solutions/02_transformer_encoder_imdb_solution.ipynb",
    ),
    (
        "themes/04-Autoregression/lab/01_decoder_only_causal_toy.ipynb",
        "themes/04-Autoregression/lab/solutions/01_decoder_only_causal_toy_solution.ipynb",
    ),
    (
        "themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare.ipynb",
        "themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_solution.ipynb",
    ),
    (
        "themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare_gpu.ipynb",
        "themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb",
    ),
    (
        "themes/05-Full-Transformer/lab/01_full_transformer_tiny_shakespeare.ipynb",
        "themes/05-Full-Transformer/lab/solutions/01_full_transformer_wikitext2_solution.ipynb",
    ),
)

RUN_SYNC_MAP = {
    "themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb": [
        rel
        for rel in EXPECTED_NOTEBOOKS
        if rel.startswith("themes/04-Autoregression/lab/solutions/runs_")
    ],
    "themes/05-Full-Transformer/lab/solutions/01_full_transformer_wikitext2_solution.ipynb": [
        rel
        for rel in EXPECTED_NOTEBOOKS
        if rel.startswith("themes/05-Full-Transformer/lab/solutions/runs_")
    ],
}

RUNTIME_GPU_NOTEBOOKS = (
    "themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare_gpu.ipynb",
    "themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb",
    "themes/05-Full-Transformer/lab/01_full_transformer_tiny_shakespeare.ipynb",
    "themes/05-Full-Transformer/lab/solutions/01_full_transformer_wikitext2_solution.ipynb",
)
